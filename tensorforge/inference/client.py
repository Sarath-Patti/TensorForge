"""Production-Facing Inference Client API & Request Contract for TensorForge v2.0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from tensorforge.inference.reliability import RetryConfig
from tensorforge.inference.scheduler import InferenceFuture
from tensorforge.inference.server import InferenceServer
from tensorforge.tensor.tensor import Tensor
from tensorforge.utils.validation import TensorForgeInputError


@dataclass
class InferenceRequestContract:
    """Standardized request contract specifying input payload, target model, and SLA options."""

    model: str
    inputs: Union[Tensor, np.ndarray, Sequence[Any]]
    version: Optional[str] = None
    timeout_ms: Optional[float] = None
    retry_config: Optional[RetryConfig] = None


class InferenceClient:
    """High-level, production-facing client interface abstraction wrapping an InferenceServer instance.

    Provides application developers with a clean, ergonomic API for prediction, batching,
    asynchronous submission, health monitoring, and diagnostics.
    """

    def __init__(self, server: InferenceServer) -> None:
        if not isinstance(server, InferenceServer):
            raise TensorForgeInputError(f"InferenceClient requires a valid InferenceServer instance, got: {type(server)}")
        self._server: InferenceServer = server

    @property
    def server(self) -> InferenceServer:
        """Underlying InferenceServer instance."""
        return self._server

    def predict(
        self,
        model: str,
        inputs: Union[Tensor, np.ndarray, Sequence[Any]],
        version: Optional[str] = None,
        timeout_ms: Optional[float] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> Tensor:
        """Synchronously execute an inference prediction request.

        Args:
            model: Target registered model name.
            inputs: Input tensor data (Tensor, NumPy array, or nested sequence).
            version: Optional target version string.
            timeout_ms: Optional maximum request deadline in milliseconds.
            retry_config: Optional explicit request-level retry policy.

        Returns:
            Output Tensor representing prediction results.
        """
        return self._server.predict(
            model=model,
            inputs=inputs,
            version=version,
            timeout_ms=timeout_ms,
            retry_config=retry_config,
        )

    def execute_contract(self, contract: InferenceRequestContract) -> Tensor:
        """Execute prediction using a standardized InferenceRequestContract specification."""
        return self.predict(
            model=contract.model,
            inputs=contract.inputs,
            version=contract.version,
            timeout_ms=contract.timeout_ms,
            retry_config=contract.retry_config,
        )

    def predict_batch(
        self,
        model: str,
        inputs_list: Sequence[Union[Tensor, np.ndarray, Sequence[Any]]],
        version: Optional[str] = None,
        timeout_ms: Optional[float] = None,
    ) -> List[Tensor]:
        """Execute predictions for a batch/sequence of inputs asynchronously and join results.

        Args:
            model: Target registered model name.
            inputs_list: Sequence of input payloads.
            version: Optional target version string.
            timeout_ms: Optional deadline in milliseconds.

        Returns:
            List of output Tensor prediction results in order.
        """
        futures = [
            self.submit(model=model, inputs=inp, version=version, timeout_ms=timeout_ms)
            for inp in inputs_list
        ]
        return [f.result() for f in futures]

    def submit(
        self,
        model: str,
        inputs: Union[Tensor, np.ndarray, Sequence[Any]],
        version: Optional[str] = None,
        timeout_ms: Optional[float] = None,
    ) -> InferenceFuture:
        """Asynchronously submit an inference request to the server queue.

        Args:
            model: Target registered model name.
            inputs: Input tensor data.
            version: Optional target version string.
            timeout_ms: Optional deadline in milliseconds.

        Returns:
            InferenceFuture handle for asynchronous result retrieval.
        """
        return self._server.submit(
            model=model,
            inputs=inputs,
            version=version,
            timeout_ms=timeout_ms,
        )

    def health(self) -> Dict[str, Any]:
        """Query server-wide and model operational health status."""
        return self._server.health()

    def models(self) -> List[Dict[str, Any]]:
        """List metadata summaries for all registered models on the server."""
        return self._server.models()

    def performance_snapshot(self) -> Dict[str, Any]:
        """Retrieve unified diagnostic and performance snapshot across all server models."""
        return self._server.performance_snapshot()

    def stats(self) -> Dict[str, Any]:
        """Query server-wide aggregate operational statistics."""
        return self._server.stats()

    def close(self, timeout_ms: Optional[float] = 5000.0) -> None:
        """Close the underlying InferenceServer cleanly."""
        self._server.close(timeout_ms=timeout_ms)

    def __enter__(self) -> InferenceClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"InferenceClient(server={self._server})"
