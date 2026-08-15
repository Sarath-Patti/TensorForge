"""Model loading and architecture reconstruction for the TensorForge Inference Runtime."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from tensorforge.nn.activations import ReLU, Sigmoid, Softmax, Tanh
from tensorforge.nn.linear import Linear
from tensorforge.nn.module import Module
from tensorforge.nn.sequential import Sequential
from tensorforge.quantization.quantized_tensor import QuantizedTensor
from tensorforge.serialization.format import read_tfmodel_container
from tensorforge.tensor.dtype import to_dtype
from tensorforge.tensor.tensor import Tensor
from tensorforge.utils.validation import SerializationError


class ModelLoader:
    """Utility class for reconstructing neural network architectures and loading parameters from .tfmodel files."""

    @staticmethod
    def reconstruct_architecture(architecture_config: Dict[str, Any]) -> Module:
        """Reconstruct a Module hierarchy from an architecture configuration dictionary.

        Supported layer types:
            - Sequential
            - Linear
            - ReLU
            - Sigmoid
            - Tanh
            - Softmax

        Args:
            architecture_config: Dictionary describing module hierarchy and hyperparameters.

        Returns:
            Reconstructed TensorForge Module instance.
        """
        if not isinstance(architecture_config, dict) or "type" not in architecture_config:
            raise SerializationError(f"Invalid architecture configuration: {architecture_config}")

        module_type = architecture_config["type"]

        if module_type == "Linear":
            in_features = architecture_config["in_features"]
            out_features = architecture_config["out_features"]
            bias = architecture_config.get("bias", True)
            dtype_str = architecture_config.get("dtype", "float32")
            return Linear(in_features=in_features, out_features=out_features, bias=bias, dtype=to_dtype(dtype_str))

        elif module_type == "ReLU":
            return ReLU()

        elif module_type == "Sigmoid":
            return Sigmoid()

        elif module_type == "Tanh":
            return Tanh()

        elif module_type == "Softmax":
            dim = architecture_config.get("dim", -1)
            return Softmax(dim=dim)

        elif module_type == "Sequential":
            layers_config = architecture_config.get("layers", [])
            submodules: List[Module] = []
            for layer_entry in layers_config:
                sub_config = layer_entry.get("module") if "module" in layer_entry else layer_entry
                submodules.append(ModelLoader.reconstruct_architecture(sub_config))
            return Sequential(*submodules)

        else:
            raise SerializationError(f"Unsupported module type in architecture configuration: '{module_type}'")

    @staticmethod
    def infer_architecture_from_state_dict(state_dict: Dict[str, Any]) -> Module:
        """Heuristically reconstruct a Sequential model when architecture metadata is absent.

        Args:
            state_dict: Dictionary mapping parameter names to Tensors.

        Returns:
            Inferred Sequential Module.
        """
        # Find linear layer keys (e.g. '0.weight', '2.weight' or 'weight')
        weight_keys = sorted([k for k in state_dict.keys() if k.endswith("weight")])
        if not weight_keys:
            raise SerializationError("Cannot infer architecture: no weight parameters found in state_dict")

        if len(weight_keys) == 1 and weight_keys[0] == "weight":
            w = state_dict["weight"]
            has_bias = "bias" in state_dict
            out_f, in_f = w.shape
            return Linear(in_features=in_f, out_features=out_f, bias=has_bias, dtype=w.dtype)

        layers: List[Module] = []
        for i, wk in enumerate(weight_keys):
            prefix = wk.rsplit(".", 1)[0]
            bk = f"{prefix}.bias"
            w = state_dict[wk]
            has_bias = bk in state_dict
            out_f, in_f = w.shape
            layers.append(Linear(in_features=in_f, out_features=out_f, bias=has_bias, dtype=w.dtype))
            # Insert ReLU between linear layers except after the last one
            if i < len(weight_keys) - 1:
                layers.append(ReLU())

        return Sequential(*layers)

    @classmethod
    def load(
        cls,
        filepath: str,
        strict: bool = True,
    ) -> Tuple[Module, Dict[str, Any], Dict[str, Any], bool]:
        """Load a .tfmodel file, reconstruct its architecture, and populate parameters.

        Args:
            filepath: Path to the .tfmodel archive.
            strict: Whether to enforce strict parameter key matching.

        Returns:
            Tuple of (model, state_dict, metadata, is_quantized).
        """
        state_dict, metadata = read_tfmodel_container(filepath)

        # Detect if model contains quantized parameters
        is_quantized = any(isinstance(v, QuantizedTensor) for v in state_dict.values())
        if not is_quantized and metadata.get("user_metadata", {}).get("is_quantized"):
            is_quantized = True

        arch_config = metadata.get("architecture")
        if arch_config:
            model = cls.reconstruct_architecture(arch_config)
        else:
            model = cls.infer_architecture_from_state_dict(state_dict)

        # For unquantized models, load parameters into physical storage
        if not is_quantized:
            model.load_state_dict(state_dict, strict=strict)

        model.eval()
        return model, state_dict, metadata, is_quantized
