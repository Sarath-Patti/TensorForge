"""Dedicated Inference Runtime for executing TensorForge models without training overhead."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np

from tensorforge.autograd.engine import no_grad
from tensorforge.backend.dispatcher import backend_context, get_backend, get_last_backend, set_backend
from tensorforge.inference.loader import ModelLoader
from tensorforge.nn.linear import Linear
from tensorforge.nn.module import Module
from tensorforge.nn.sequential import Sequential
from tensorforge.quantization.quantize import qmatmul, quantize
from tensorforge.quantization.quantized_tensor import QuantizedTensor
from tensorforge.tensor.dtype import float32
from tensorforge.tensor.tensor import Tensor
from tensorforge.utils.validation import TensorForgeError


class InferenceRuntime:
    """A high-performance, standalone inference execution engine for TensorForge models.

    Loads serialized .tfmodel artifacts, reconstructs the network graph, restores weights,
    and executes forward predictions in eval mode with no_grad guarantees across CPU NumPy
    and Native C++ acceleration backends.

    Args:
        model: Reconstructed Module instance in evaluation mode.
        metadata: Model archive metadata.
        is_quantized: Whether the model parameters are stored in INT8 low precision.
        state_dict: State dictionary containing raw or quantized parameters.
        backend: Optional backend override ('numpy' or 'native').
    """

    def __init__(
        self,
        model: Module,
        metadata: Dict[str, Any],
        is_quantized: bool = False,
        state_dict: Optional[Dict[str, Any]] = None,
        backend: Optional[str] = None,
    ) -> None:
        self._model: Module = model
        self._metadata: Dict[str, Any] = metadata
        self._is_quantized: bool = is_quantized
        self._state_dict: Dict[str, Any] = state_dict or {}
        self._backend: Optional[str] = backend

        self._model.eval()

        # Infer input and output dimensions
        self._input_shape: Optional[Tuple[int, ...]] = None
        self._output_shape: Optional[Tuple[int, ...]] = None
        self._infer_shapes()

    def _infer_shapes(self) -> None:
        """Infer input and output feature shapes from model structure."""
        if isinstance(self._model, Linear):
            self._input_shape = (self._model.in_features,)
            self._output_shape = (self._model.out_features,)
        elif isinstance(self._model, Sequential) and len(self._model) > 0:
            for mod in self._model:
                if isinstance(mod, Linear) and self._input_shape is None:
                    self._input_shape = (mod.in_features,)
            for mod in reversed(list(self._model)):
                if isinstance(mod, Linear) and self._output_shape is None:
                    self._output_shape = (mod.out_features,)

    @classmethod
    def load(
        cls,
        filepath: str,
        backend: Optional[str] = None,
        strict: bool = True,
    ) -> InferenceRuntime:
        """Load a .tfmodel artifact and construct a ready-to-use InferenceRuntime.

        Args:
            filepath: Path to the .tfmodel archive.
            backend: Optional backend override ('numpy' or 'native').
            strict: Whether to enforce strict parameter key matching.

        Returns:
            Configured InferenceRuntime instance.
        """
        model, state_dict, metadata, is_quantized = ModelLoader.load(filepath, strict=strict)
        return cls(
            model=model,
            metadata=metadata,
            is_quantized=is_quantized,
            state_dict=state_dict,
            backend=backend,
        )

    @property
    def model(self) -> Module:
        """Access the underlying reconstructed neural network Module."""
        return self._model

    @property
    def backend(self) -> str:
        """Return the active compute backend (configured or global)."""
        return self._backend if self._backend is not None else get_backend()

    @property
    def is_quantized(self) -> bool:
        """Whether the runtime is operating in INT8 low-precision mode."""
        return self._is_quantized

    @property
    def metadata(self) -> Dict[str, Any]:
        """Dictionary of model metadata loaded from the .tfmodel container."""
        return self._metadata

    @property
    def input_shape(self) -> Optional[Tuple[int, ...]]:
        """Inferred expected input feature shape."""
        return self._input_shape

    @property
    def output_shape(self) -> Optional[Tuple[int, ...]]:
        """Inferred expected output feature shape."""
        return self._output_shape

    def predict(
        self,
        input_data: Union[Tensor, np.ndarray, Sequence[Any]],
    ) -> Tensor:
        """Execute inference prediction on the given input sample or batch.

        Guarantees that inference runs in `eval` mode with `no_grad` active,
        producing detached tensors with no autograd graph overhead.

        Args:
            input_data: Input tensor, NumPy array, or nested sequence.

        Returns:
            Output Tensor representing prediction results.
        """
        if isinstance(input_data, Tensor):
            x = input_data
        elif isinstance(input_data, np.ndarray):
            x = Tensor(input_data, dtype=float32, copy=False)
        else:
            x = Tensor(np.asarray(input_data, dtype=np.float32), dtype=float32)

        target_backend = self.backend
        with backend_context(target_backend):
            with no_grad():
                if self._is_quantized:
                    output = self._predict_quantized(x)
                else:
                    output = self._model(x)

        return output.detach()

    def predict_batch(
        self,
        batch_data: Union[Tensor, np.ndarray, Sequence[Any]],
    ) -> Tensor:
        """Execute batched inference on multi-sample inputs (alias for predict)."""
        return self.predict(batch_data)

    def _predict_quantized(self, x: Tensor) -> Tensor:
        """Execute quantized INT8 forward inference path."""
        current: Tensor = x

        if isinstance(self._model, Sequential):
            for idx, layer in enumerate(self._model):
                w_key = f"{idx}.weight"
                b_key = f"{idx}.bias"

                if w_key in self._state_dict and isinstance(self._state_dict[w_key], QuantizedTensor):
                    w_q = self._state_dict[w_key]
                    w_q_t = quantize(w_q.dequantize().transpose(), scheme="symmetric")
                    x_q = quantize(current, scheme="symmetric") if not isinstance(current, QuantizedTensor) else current
                    h = qmatmul(x_q, w_q_t)

                    if b_key in self._state_dict:
                        bias_val = self._state_dict[b_key]
                        bias_t = bias_val.dequantize() if isinstance(bias_val, QuantizedTensor) else bias_val
                        h = h + bias_t
                    current = h
                else:
                    current = layer(current)
        elif isinstance(self._model, Linear):
            w_key = "weight"
            b_key = "bias"
            if w_key in self._state_dict and isinstance(self._state_dict[w_key], QuantizedTensor):
                w_q = self._state_dict[w_key]
                w_q_t = quantize(w_q.dequantize().transpose(), scheme="symmetric")
                x_q = quantize(current, scheme="symmetric") if not isinstance(current, QuantizedTensor) else current
                h = qmatmul(x_q, w_q_t)
                if b_key in self._state_dict:
                    bias_val = self._state_dict[b_key]
                    bias_t = bias_val.dequantize() if isinstance(bias_val, QuantizedTensor) else bias_val
                    h = h + bias_t
                current = h
            else:
                current = self._model(current)
        else:
            current = self._model(current)

        return current

    def summary(self) -> Dict[str, Any]:
        """Generate a diagnostic summary of the loaded model and runtime environment.

        Returns:
            Dictionary with architecture details, parameter counts, memory consumption,
            backend configuration, and input/output shapes.
        """
        from tensorforge.serialization.checkpoint import compute_model_size

        size_stats = compute_model_size(self._state_dict if self._is_quantized else self._model)

        return {
            "model_type": type(self._model).__name__,
            "architecture": repr(self._model),
            "is_quantized": self._is_quantized,
            "backend": self.backend,
            "last_dispatch": get_last_backend(),
            "input_shape": self._input_shape,
            "output_shape": self._output_shape,
            "num_parameters": size_stats["num_parameters"],
            "total_bytes": size_stats["total_bytes"],
            "size_kb": size_stats["size_kb"],
            "format_version": self._metadata.get("format_version", "1.0"),
            "tensorforge_version": self._metadata.get("tensorforge_version", "0.9.0"),
        }

    def __repr__(self) -> str:
        q_str = " (INT8 Quantized)" if self._is_quantized else ""
        return (
            f"InferenceRuntime(\n"
            f"  backend='{self.backend}',\n"
            f"  is_quantized={self._is_quantized},\n"
            f"  input_shape={self._input_shape},\n"
            f"  output_shape={self._output_shape},\n"
            f"  model={repr(self._model)}\n"
            f")"
        )
