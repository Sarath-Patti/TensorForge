"""Production Inference Serving Subsystem for TensorForge.

Provides a clean, in-process multi-model inference serving abstraction
(InferenceServer) above the existing InferenceScheduler, InferenceRuntime,
and MetricsCollector layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from tensorforge.inference.limits import RuntimeLimits
from tensorforge.inference.observability import (
    MetricsCollector,
    PerformanceSnapshot,
)
from tensorforge.inference.reliability import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    HealthState,
    RetryConfig,
    compute_backoff_delay_sec,
)
from tensorforge.inference.runtime import InferenceRuntime
from tensorforge.inference.scheduler import (
    InferenceFuture,
    InferenceScheduler,
    SchedulerConfig,
)
from tensorforge.serialization.format import LIBRARY_VERSION
from tensorforge.tensor.tensor import Tensor
from tensorforge.utils.validation import (
    CircuitBreakerOpenError,
    ModelAlreadyLoadedError,
    ModelDegradedError,
    ModelLoadError,
    ModelNotFoundError,
    ModelNotReadyError,
    ModelVersionNotFoundError,
    RequestCancelledError,
    RequestDeadlineExceededError,
    RetryLimitExceededError,
    ServerClosedError,
    ServerError,
    ServerLimitError,
    TensorForgeInputError,
)


class ModelLifecycleState(str, Enum):
    """Lifecycle states for a registered model instance."""

    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    READY = "READY"
    DRAINING = "DRAINING"
    FAILED = "FAILED"


class ServerLifecycleState(str, Enum):
    """Lifecycle states for the InferenceServer."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    DRAINING = "DRAINING"
    CLOSED = "CLOSED"


@dataclass
class ServerConfig:
    """Configuration options for InferenceServer."""

    max_loaded_models: Optional[int] = None
    max_total_pending_requests: Optional[int] = None
    auto_start: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "max_loaded_models": self.max_loaded_models,
            "max_total_pending_requests": self.max_total_pending_requests,
            "auto_start": self.auto_start,
        }


@dataclass
class ModelEntry:
    """Registry entry representing a loaded model version."""

    name: str
    version: str
    path: str
    state: ModelLifecycleState = ModelLifecycleState.UNLOADED
    runtime: Optional[InferenceRuntime] = None
    scheduler: Optional[InferenceScheduler] = None
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    health_state: HealthState = HealthState.HEALTHY
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    loaded_at_timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = False
    error_message: Optional[str] = None
    owner: bool = True

    @property
    def key(self) -> Tuple[str, str]:
        """Unique key for the model entry."""
        return (self.name, self.version)

    def to_dict(self) -> Dict[str, Any]:
        """Return read-only metadata dictionary representation."""
        backend_name = "unknown"
        is_compiled = False
        is_quantized = False
        input_shape = None
        output_shape = None

        if self.runtime is not None:
            backend_name = self.runtime.backend
            is_compiled = self.runtime.is_compiled
            is_quantized = self.runtime.is_quantized
            input_shape = self.runtime.input_shape
            output_shape = self.runtime.output_shape

        return {
            "name": self.name,
            "version": self.version,
            "path": self.path,
            "state": self.state.value,
            "health_state": self.health_state.value,
            "circuit_breaker": self.circuit_breaker.to_dict(),
            "is_active": self.is_active,
            "backend": backend_name,
            "is_compiled": is_compiled,
            "is_quantized": is_quantized,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "loaded_at": self.loaded_at_timestamp,
            "metadata": dict(self.metadata),
            "error_message": self.error_message,
        }


class ModelRegistry:
    """Thread-safe model registry managing loaded model versions and routing resolution."""

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._models: Dict[Tuple[str, str], ModelEntry] = {}
        self._active_versions: Dict[str, str] = {}

    def register(self, entry: ModelEntry, set_active: bool = True, overwrite: bool = False) -> None:
        """Register a model entry.

        Args:

            entry: ModelEntry to register.
            set_active: Whether to set this version as the active default for the model name.
            overwrite: Whether to overwrite existing registration.

        Raises:
            ModelAlreadyLoadedError: If duplicate version exists and overwrite is False.
        """
        with self._lock:
            key = entry.key
            if key in self._models and not overwrite:
                raise ModelAlreadyLoadedError(
                    f"Model '{entry.name}' version '{entry.version}' is already registered."
                )

            # If this is the first version registered for the model name, set active automatically
            if entry.name not in self._active_versions or set_active:
                # Clear previous active flag for other versions of this model name
                for (m_name, m_ver), m_entry in self._models.items():
                    if m_name == entry.name:
                        m_entry.is_active = False
                entry.is_active = True
                self._active_versions[entry.name] = entry.version
            else:
                entry.is_active = False

            self._models[key] = entry

    def unregister(self, name: str, version: str) -> Optional[ModelEntry]:
        """Remove a model version from the registry."""
        with self._lock:
            key = (name, version)
            entry = self._models.pop(key, None)
            if entry is not None:
                entry.is_active = False
                if self._active_versions.get(name) == version:
                    # Resolve new active version if available
                    remaining_versions = [v for (n, v) in self._models.keys() if n == name]
                    if remaining_versions:
                        new_active_ver = remaining_versions[-1]
                        self._active_versions[name] = new_active_ver
                        self._models[(name, new_active_ver)].is_active = True
                    else:
                        self._active_versions.pop(name, None)
            return entry

    def get(self, name: str, version: Optional[str] = None) -> ModelEntry:
        """Retrieve a ModelEntry by name and optional version.

        If version is None, resolves the configured active version for name.

        Raises:
            ModelNotFoundError: If model name is not registered.
            ModelVersionNotFoundError: If specified version is not found.
        """
        with self._lock:
            if version is None:
                if name not in self._active_versions:
                    raise ModelNotFoundError(f"No active model found registered for name '{name}'.")
                version = self._active_versions[name]

            key = (name, version)
            if key not in self._models:
                if (name, self._active_versions.get(name)) in self._models or any(n == name for (n, v) in self._models):
                    raise ModelVersionNotFoundError(f"Version '{version}' not found for model '{name}'.")
                raise ModelNotFoundError(f"Model '{name}' is not registered.")

            return self._models[key]

    def set_active_version(self, name: str, version: str) -> None:
        """Set the active default version for a model name.

        Raises:
            ModelNotFoundError: If model name is not registered.
            ModelVersionNotFoundError: If specified version is not found.
        """
        with self._lock:
            key = (name, version)
            if key not in self._models:
                if any(n == name for (n, v) in self._models):
                    raise ModelVersionNotFoundError(f"Version '{version}' not found for model '{name}'.")
                raise ModelNotFoundError(f"Model '{name}' is not registered.")

            for (m_name, m_ver), entry in self._models.items():
                if m_name == name:
                    entry.is_active = (m_ver == version)

            self._active_versions[name] = version

    def get_active_version(self, name: str) -> str:
        """Get the active version string for a model name."""
        with self._lock:
            if name not in self._active_versions:
                raise ModelNotFoundError(f"Model '{name}' is not registered.")
            return self._active_versions[name]

    def has_model(self, name: str, version: Optional[str] = None) -> bool:
        """Check if a model (and optional version) exists in the registry."""
        with self._lock:
            if version is None:
                return name in self._active_versions
            return (name, version) in self._models

    def list_models(self) -> List[ModelEntry]:
        """List all registered model entries."""
        with self._lock:
            return list(self._models.values())

    def clear(self) -> List[ModelEntry]:
        """Clear registry and return list of all previous entries."""
        with self._lock:
            entries = list(self._models.values())
            self._models.clear()
            self._active_versions.clear()
            return entries


class InferenceServer:
    """Production Multi-Model Inference Server for TensorForge.

    Orchestrates multiple named model instances, manages model versioning, routing,
    lifecycle states, failure isolation, and aggregate observability.
    """

    def __init__(self, config: Optional[ServerConfig] = None) -> None:
        self._config: ServerConfig = config if config is not None else ServerConfig()
        self._registry: ModelRegistry = ModelRegistry()
        self._lock: threading.RLock = threading.RLock()
        self._state: ServerLifecycleState = ServerLifecycleState.CREATED

        if self._config.auto_start:
            self.start()

    @property
    def config(self) -> ServerConfig:
        """Access server configuration."""
        return self._config

    @property
    def registry(self) -> ModelRegistry:
        """Access underlying ModelRegistry instance."""
        return self._registry

    @property
    def state(self) -> ServerLifecycleState:
        """Current server lifecycle state."""
        with self._lock:
            return self._state

    @property
    def lifecycle_state(self) -> ServerLifecycleState:
        """Alias for state property."""
        return self.state

    def start(self) -> InferenceServer:
        """Start the InferenceServer."""
        with self._lock:
            if self._state == ServerLifecycleState.CLOSED:
                raise ServerClosedError("Cannot start InferenceServer: Server has been closed.")
            if self._state == ServerLifecycleState.CREATED:
                self._state = ServerLifecycleState.RUNNING
        return self

    def load_model(
        self,
        name: str,
        path: str,
        version: str = "1",
        active: bool = True,
        scheduler_config: Optional[SchedulerConfig] = None,
        runtime_limits: Optional[RuntimeLimits] = None,
        num_threads: Optional[int] = None,
        backend: str = "numpy",
        compile_input_shape: Optional[Tuple[int, ...]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> ModelEntry:
        """Load a serialized TensorForge model into the server.

        Args:
            name: Model identifier name.
            path: Filepath to serialized .tfmodel artifact.
            version: Model version identifier (default: "1").
            active: Set as default active version for the model name.
            scheduler_config: Optional dynamic batching scheduler configuration.
            runtime_limits: Optional execution limits for the runtime.
            num_threads: Optional CPU thread pool worker count.
            backend: Target backend ("numpy" or "native").
            compile_input_shape: Optional input shape to trigger graph compilation.
            metadata: Optional user-defined metadata dictionary.
            overwrite: Replace existing model version if registered.

        Returns:
            ModelEntry representing the successfully loaded model version.

        Raises:
            ServerClosedError: If server is closed or draining.
            ServerLimitError: If max_loaded_models limit is exceeded.
            ModelAlreadyLoadedError: If duplicate model version exists without overwrite.
            ModelLoadError: If loading or runtime creation fails.
        """
        with self._lock:
            if self._state != ServerLifecycleState.RUNNING:
                raise ServerClosedError(f"Cannot load model: InferenceServer is {self._state.value}.")

            if self._config.max_loaded_models is not None:
                loaded_count = len(self._registry.list_models())
                if not overwrite and loaded_count >= self._config.max_loaded_models:
                    raise ServerLimitError(
                        f"Server capacity reached ({loaded_count}/{self._config.max_loaded_models} loaded models)."
                    )

            if self._registry.has_model(name, version) and not overwrite:
                raise ModelAlreadyLoadedError(
                    f"Model '{name}' version '{version}' is already loaded."
                )

        entry = ModelEntry(
            name=name,
            version=version,
            path=path,
            state=ModelLifecycleState.LOADING,
            metadata=dict(metadata or {}),
            is_active=active,
            owner=True,
        )

        try:
            # 1. Load InferenceRuntime using authoritative loader
            runtime = InferenceRuntime.load(
                path,
                num_threads=num_threads,
                backend=backend,
                limits=runtime_limits,
            )

            # 2. Trigger compilation if requested
            if compile_input_shape is not None:
                runtime.compile(input_shape=compile_input_shape)

            # 3. Instantiate model-specific InferenceScheduler
            sched_cfg = scheduler_config if scheduler_config is not None else SchedulerConfig()
            scheduler = InferenceScheduler(runtime, config=sched_cfg)

            entry.runtime = runtime
            entry.scheduler = scheduler
            entry.state = ModelLifecycleState.READY

            # 4. Register into ModelRegistry
            self._registry.register(entry, set_active=active, overwrite=overwrite)
            return entry

        except Exception as e:
            entry.state = ModelLifecycleState.FAILED
            entry.error_message = str(e)
            raise ModelLoadError(f"Failed to load model '{name}:{version}': {e}") from e

    def register_runtime(
        self,
        name: str,
        runtime: InferenceRuntime,
        version: str = "1",
        active: bool = True,
        scheduler_config: Optional[SchedulerConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> ModelEntry:
        """Register an existing InferenceRuntime instance into the server.

        Args:
            name: Model identifier name.
            runtime: Active InferenceRuntime instance.
            version: Version string.
            active: Set as default active version.
            scheduler_config: Optional scheduler configuration.
            metadata: Optional metadata dict.
            overwrite: Overwrite existing registration.

        Returns:
            Registered ModelEntry.
        """
        with self._lock:
            if self._state != ServerLifecycleState.RUNNING:
                raise ServerClosedError(f"Cannot register model: InferenceServer is {self._state.value}.")

        sched_cfg = scheduler_config if scheduler_config is not None else SchedulerConfig()
        scheduler = InferenceScheduler(runtime, config=sched_cfg)

        entry = ModelEntry(
            name=name,
            version=version,
            path="in-memory",
            state=ModelLifecycleState.READY,
            runtime=runtime,
            scheduler=scheduler,
            metadata=dict(metadata or {}),
            is_active=active,
            owner=False,
        )
        self._registry.register(entry, set_active=active, overwrite=overwrite)
        return entry

    def unload_model(self, name: str, version: Optional[str] = None, force: bool = False) -> bool:
        """Unload a model version and release its scheduler and runtime resources.

        Args:
            name: Model name.
            version: Specific version string (or None for active version).
            force: Force immediate resource cleanup.

        Returns:
            True if model was successfully unloaded.
        """
        entry = self._registry.get(name, version)

        with self._lock:
            entry.state = ModelLifecycleState.DRAINING

        try:
            if entry.scheduler is not None:
                entry.scheduler.close()
            if entry.runtime is not None and entry.owner:
                entry.runtime.close()
        finally:
            entry.state = ModelLifecycleState.UNLOADED
            self._registry.unregister(entry.name, entry.version)

        return True

    def reload_model(
        self,
        name: str,
        path: str,
        version: Optional[str] = None,
        active: bool = True,
        scheduler_config: Optional[SchedulerConfig] = None,
        runtime_limits: Optional[RuntimeLimits] = None,
        num_threads: Optional[int] = None,
        backend: str = "numpy",
        compile_input_shape: Optional[Tuple[int, ...]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelEntry:
        """Safely reload a model version with zero downtime.

        Loads new runtime into a temporary state; on success, atomically swaps
        the registry entry and drains the old version.
        """
        target_version = version if version is not None else self._registry.get_active_version(name)

        # 1. Load new model version into temporary entry
        try:
            new_entry = self.load_model(
                name=name,
                path=path,
                version=f"{target_version}-new",
                active=False,
                scheduler_config=scheduler_config,
                runtime_limits=runtime_limits,
                num_threads=num_threads,
                backend=backend,
                compile_input_shape=compile_input_shape,
                metadata=metadata,
                overwrite=True,
            )
        except Exception as e:
            # Atomic reload failure recovery: target version remains READY and untouched
            raise ModelLoadError(
                f"Failed atomic model reload for '{name}:{target_version}'. Existing version remains active. Error: {e}"
            ) from e

        # 2. Swap key to target version
        self._registry.unregister(name, f"{target_version}-new")
        old_entry = None
        if self._registry.has_model(name, target_version):
            old_entry = self._registry.get(name, target_version)

        new_entry.version = target_version
        self._registry.register(new_entry, set_active=active, overwrite=True)

        # 3. Cleanly drain old entry if present
        if old_entry is not None and old_entry is not new_entry:
            old_entry.state = ModelLifecycleState.DRAINING
            try:
                if old_entry.scheduler is not None:
                    old_entry.scheduler.close()
                if old_entry.runtime is not None and old_entry.owner:
                    old_entry.runtime.close()
            except Exception:
                pass
            old_entry.state = ModelLifecycleState.UNLOADED

        return new_entry

    def set_active_version(self, name: str, version: str) -> None:
        """Set active version for model name."""
        self._registry.set_active_version(name, version)

    def get_active_version(self, name: str) -> str:
        """Get active version for model name."""
        return self._registry.get_active_version(name)

    def predict(
        self,
        model: str,
        inputs: Union[Tensor, np.ndarray, Sequence[Any]],
        version: Optional[str] = None,
        timeout: Optional[float] = None,
        timeout_ms: Optional[float] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> Tensor:
        """Synchronously execute inference request through model's scheduler.

        Args:
            model: Registered model name.
            inputs: Input tensor data.
            version: Optional target version string.
            timeout: Maximum execution wait timeout in seconds.
            timeout_ms: Optional maximum request deadline in milliseconds.
            retry_config: Optional explicit request retry configuration.

        Returns:
            Output Tensor prediction result.
        """
        with self._lock:
            if self._state != ServerLifecycleState.RUNNING:
                raise ServerClosedError(f"Cannot execute predict(): InferenceServer is {self._state.value}.")

            if self._config.max_total_pending_requests is not None:
                total_pending = sum(
                    e.scheduler.queue_depth
                    for e in self._registry.list_models()
                    if e.scheduler is not None
                )
                if total_pending >= self._config.max_total_pending_requests:
                    raise ServerLimitError(
                        f"Server pending request limit reached ({total_pending}/{self._config.max_total_pending_requests})."
                    )

        entry = self._registry.get(model, version)

        if entry.state != ModelLifecycleState.READY or entry.scheduler is None:
            raise ModelNotReadyError(
                f"Model '{entry.name}' version '{entry.version}' is not in READY state (current: {entry.state.value})."
            )

        if not entry.circuit_breaker.allow_request():
            raise CircuitBreakerOpenError(
                f"Circuit breaker for model '{entry.name}:{entry.version}' is OPEN."
            )

        effective_timeout_ms = timeout_ms if timeout_ms is not None else (timeout * 1000.0 if timeout is not None else None)
        effective_retry_config = retry_config if retry_config is not None else entry.retry_config
        start_monotonic = time.monotonic()
        deadline = (start_monotonic + (effective_timeout_ms / 1000.0)) if effective_timeout_ms is not None else None

        attempts = 0
        max_retries = effective_retry_config.max_retries if effective_retry_config else 0

        while True:
            try:
                rem_timeout_ms = None
                if deadline is not None:
                    rem_sec = deadline - time.monotonic()
                    if rem_sec <= 0.0:
                        raise RequestDeadlineExceededError(f"Request deadline exceeded for model '{entry.name}:{entry.version}'.")
                    rem_timeout_ms = rem_sec * 1000.0

                result = entry.scheduler.predict(inputs, timeout_ms=rem_timeout_ms)
                entry.circuit_breaker.record_success()
                return result

            except Exception as e:
                # Do not record failures or retry client input validation errors
                if isinstance(e, (TensorForgeInputError, ValueError, TypeError)):
                    raise

                entry.circuit_breaker.record_failure()
                entry.health_state = HealthState.DEGRADED

                if attempts < max_retries and effective_retry_config.is_retryable(e):
                    attempts += 1
                    delay_sec = compute_backoff_delay_sec(attempts - 1, effective_retry_config)
                    if deadline is not None and time.monotonic() + delay_sec >= deadline:
                        raise RequestDeadlineExceededError(
                            f"Request deadline exceeded during retry backoff for model '{entry.name}:{entry.version}'."
                        ) from e
                    time.sleep(delay_sec)
                else:
                    raise

    def submit(
        self,
        model: str,
        inputs: Union[Tensor, np.ndarray, Sequence[Any]],
        version: Optional[str] = None,
        timeout_ms: Optional[float] = None,
    ) -> InferenceFuture:
        """Asynchronously submit inference request to model's scheduler.

        Args:
            model: Registered model name.
            inputs: Input tensor data.
            version: Optional target version string.
            timeout_ms: Optional deadline in milliseconds.

        Returns:
            InferenceFuture object for asynchronous result retrieval.
        """
        with self._lock:
            if self._state != ServerLifecycleState.RUNNING:
                raise ServerClosedError(f"Cannot submit request: InferenceServer is {self._state.value}.")

            if self._config.max_total_pending_requests is not None:
                total_pending = sum(
                    e.scheduler.queue_depth
                    for e in self._registry.list_models()
                    if e.scheduler is not None
                )
                if total_pending >= self._config.max_total_pending_requests:
                    raise ServerLimitError(
                        f"Server pending request limit reached ({total_pending}/{self._config.max_total_pending_requests})."
                    )

        entry = self._registry.get(model, version)

        if entry.state != ModelLifecycleState.READY or entry.scheduler is None:
            raise ModelNotReadyError(
                f"Model '{entry.name}' version '{entry.version}' is not in READY state (current: {entry.state.value})."
            )

        if not entry.circuit_breaker.allow_request():
            raise CircuitBreakerOpenError(
                f"Circuit breaker for model '{entry.name}:{entry.version}' is OPEN."
            )

        return entry.scheduler.submit(inputs, timeout_ms=timeout_ms)

    def health(self) -> Dict[str, Any]:
        """Perform a lightweight operational health check on the InferenceServer."""
        models_list = self._registry.list_models()
        ready_count = sum(1 for m in models_list if m.state == ModelLifecycleState.READY)
        failed_count = sum(1 for m in models_list if m.state == ModelLifecycleState.FAILED)
        total_pending = sum(m.scheduler.queue_depth for m in models_list if m.scheduler is not None)

        model_health_map = {}
        for m in models_list:
            model_key = f"{m.name}:{m.version}"
            sched_health = m.scheduler.health() if m.scheduler is not None else {}
            model_health_map[model_key] = {
                "name": m.name,
                "version": m.version,
                "state": m.state.value,
                "health_state": m.health_state.value,
                "circuit_state": m.circuit_breaker.state.value,
                "is_active": m.is_active,
                "circuit_breaker": m.circuit_breaker.to_dict(),
                "scheduler_health": sched_health,
            }

        with self._lock:
            state_val = self._state.value

        return {
            "status": "healthy" if state_val == "RUNNING" else "degraded",
            "server_state": state_val,
            "loaded_models_count": len(models_list),
            "ready_models_count": ready_count,
            "failed_models_count": failed_count,
            "total_pending_requests": total_pending,
            "models": model_health_map,
        }

    def models(self) -> List[Dict[str, Any]]:
        """List metadata summaries for all registered models."""
        return [entry.to_dict() for entry in self._registry.list_models()]

    def stats(self) -> Dict[str, Any]:
        """Aggregate statistical diagnostic report across all loaded models."""
        models_list = self._registry.list_models()
        total_sub = sum(m.scheduler.stats()["submitted_requests"] for m in models_list if m.scheduler)
        total_comp = sum(m.scheduler.stats()["completed_requests"] for m in models_list if m.scheduler)
        total_fail = sum(m.scheduler.stats()["failed_requests"] for m in models_list if m.scheduler)
        total_rej = sum(m.scheduler.stats()["rejected_requests"] for m in models_list if m.scheduler)
        total_batches = sum(m.scheduler.stats()["batches_formed"] for m in models_list if m.scheduler)
        total_samples = sum(m.scheduler.stats()["total_samples_processed"] for m in models_list if m.scheduler)

        per_model_stats = {}
        for m in models_list:
            key = f"{m.name}:{m.version}"
            per_model_stats[key] = m.scheduler.stats() if m.scheduler else {}

        return {
            "server_state": self.state.value,
            "config": self._config.to_dict(),
            "total_models": len(models_list),
            "submitted_requests": total_sub,
            "completed_requests": total_comp,
            "failed_requests": total_fail,
            "rejected_requests": total_rej,
            "batches_formed": total_batches,
            "total_samples_processed": total_samples,
            "models": per_model_stats,
            "tensorforge_version": LIBRARY_VERSION,
        }

    def performance_snapshot(self) -> Dict[str, Any]:
        """Generate an aggregated PerformanceSnapshot across all registered models."""
        models_list = self._registry.list_models()
        per_model_snapshots = {}
        collector = MetricsCollector()

        for m in models_list:
            if m.scheduler is not None:
                snap = m.scheduler.performance_snapshot()
                key = f"{m.name}:{m.version}"
                per_model_snapshots[key] = snap.to_dict()

                # Merge into aggregate metrics collector
                collector.record_request_submitted()
                if snap.requests.completed > 0:
                    collector.record_request_completed(
                        queue_wait_ms=snap.latency.queue.mean_ms,
                        exec_ms=snap.latency.execution.mean_ms,
                        e2e_ms=snap.latency.end_to_end.mean_ms,
                        samples=snap.batches.samples_processed,
                    )

        server_snapshot = collector.snapshot().to_dict()
        return {
            "server": {
                "state": self.state.value,
                "loaded_models": len(models_list),
                "aggregate_metrics": server_snapshot,
            },
            "models": per_model_snapshots,
            "tensorforge_version": LIBRARY_VERSION,
        }

    def export_metrics(self, filepath: str, indent: int = 2) -> None:
        """Export server performance snapshot to a JSON file."""
        import json
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.performance_snapshot(), f, indent=indent)

    def reset_metrics(self) -> InferenceServer:
        """Reset metrics on all loaded model schedulers."""
        for m in self._registry.list_models():
            if m.scheduler is not None:
                m.scheduler.reset_metrics()
        return self

    def close(self, timeout_ms: Optional[float] = 5000.0) -> None:
        """Shutdown the InferenceServer cleanly within a monotonic timeout."""
        with self._lock:
            if self._state == ServerLifecycleState.CLOSED:
                return
            self._state = ServerLifecycleState.DRAINING

        deadline = (time.monotonic() + (timeout_ms / 1000.0)) if timeout_ms is not None else None

        entries = self._registry.clear()
        for entry in entries:
            entry.state = ModelLifecycleState.DRAINING
            try:
                if entry.scheduler is not None:
                    rem_sec = max(0.0, deadline - time.monotonic()) if deadline is not None else None
                    entry.scheduler.close(drain=(rem_sec is None or rem_sec > 0.0))
                if entry.runtime is not None and entry.owner:
                    entry.runtime.close()
            except Exception:
                pass
            entry.state = ModelLifecycleState.UNLOADED

        with self._lock:
            self._state = ServerLifecycleState.CLOSED

    def __enter__(self) -> InferenceServer:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        models_list = self._registry.list_models()
        return (
            f"InferenceServer(\n"
            f"  state='{self.state.value}',\n"
            f"  loaded_models={len(models_list)},\n"
            f"  active_models={[m.name + ':' + m.version for m in models_list if m.is_active]}\n"
            f")"
        )
