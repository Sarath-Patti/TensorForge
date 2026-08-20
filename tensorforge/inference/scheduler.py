"""Production Inference Scheduling & Dynamic Batching Subsystem for TensorForge."""

from __future__ import annotations

import collections
from enum import Enum
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from tensorforge.inference.limits import RuntimeLimits
from tensorforge.inference.observability import (
    MetricsCollector,
    PerformanceSnapshot,
    SchedulerMetrics,
)
from tensorforge.inference.runtime import InferenceRuntime
from tensorforge.tensor.dtype import float32
from tensorforge.tensor.tensor import Tensor
from tensorforge.utils.validation import (
    SchedulerClosedError,
    SchedulerError,
    SchedulerQueueFullError,
    TensorForgeError,
    TensorForgeInputError,
)


class SchedulingPolicy(str, Enum):
    """Scheduling policies for dynamic batch formation."""

    FIFO = "FIFO"
    LARGEST_BATCH_FIRST = "LARGEST_BATCH_FIRST"


class SchedulerLifecycleState(str, Enum):
    """Lifecycle states for the InferenceScheduler."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    DRAINING = "DRAINING"
    CLOSED = "CLOSED"


class SchedulerConfig:
    """Configuration specification for the InferenceScheduler.

    Args:
        max_batch_size: Maximum dynamic batch size formed per inference step.
        max_queue_size: Maximum number of pending requests permitted in the queue.
        batch_timeout_ms: Maximum time in milliseconds to wait for a full batch before dispatching.
        max_pending_requests: Optional alias/limit for queue capacity.
        policy: Request selection policy (FIFO or LARGEST_BATCH_FIRST).
        drain_on_close: Whether to process already-enqueued requests prior to shutdown.

    Raises:
        ValueError: If any parameter is invalid (non-positive or unsupported policy).
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_queue_size: int = 128,
        batch_timeout_ms: float = 2.0,
        max_pending_requests: Optional[int] = None,
        policy: Union[SchedulingPolicy, str] = SchedulingPolicy.FIFO,
        drain_on_close: bool = True,
    ) -> None:
        if not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError(f"max_batch_size must be a positive integer, got {max_batch_size}")

        if max_pending_requests is not None:
            if not isinstance(max_pending_requests, int) or max_pending_requests < 1:
                raise ValueError(f"max_pending_requests must be a positive integer, got {max_pending_requests}")
            effective_queue_size = max_pending_requests
        else:
            if not isinstance(max_queue_size, int) or max_queue_size < 1:
                raise ValueError(f"max_queue_size must be a positive integer, got {max_queue_size}")
            effective_queue_size = max_queue_size

        if not isinstance(batch_timeout_ms, (int, float)) or batch_timeout_ms < 0:
            raise ValueError(f"batch_timeout_ms must be non-negative, got {batch_timeout_ms}")

        if isinstance(policy, str):
            try:
                policy = SchedulingPolicy(policy.upper())
            except ValueError:
                raise ValueError(f"Unsupported scheduling policy '{policy}'. Supported: FIFO, LARGEST_BATCH_FIRST")

        self.max_batch_size: int = max_batch_size
        self.max_queue_size: int = effective_queue_size
        self.batch_timeout_ms: float = float(batch_timeout_ms)
        self.policy: SchedulingPolicy = policy
        self.drain_on_close: bool = drain_on_close

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the configuration."""
        return {
            "max_batch_size": self.max_batch_size,
            "max_queue_size": self.max_queue_size,
            "batch_timeout_ms": self.batch_timeout_ms,
            "policy": self.policy.value,
            "drain_on_close": self.drain_on_close,
        }

    def copy(self) -> SchedulerConfig:
        """Create a deep copy of this SchedulerConfig instance."""
        return SchedulerConfig(
            max_batch_size=self.max_batch_size,
            max_queue_size=self.max_queue_size,
            batch_timeout_ms=self.batch_timeout_ms,
            policy=self.policy,
            drain_on_close=self.drain_on_close,
        )

    def __repr__(self) -> str:
        return (
            f"SchedulerConfig(max_batch_size={self.max_batch_size}, "
            f"max_queue_size={self.max_queue_size}, "
            f"batch_timeout_ms={self.batch_timeout_ms}, "
            f"policy='{self.policy.value}', "
            f"drain_on_close={self.drain_on_close})"
        )


class InferenceRequest:
    """Internal thread-safe representation of an inference request in the scheduler pipeline.

    Args:
        request_id: Unique request identifier.
        input_tensor: 2D or 1D Tensor to predict.
        submission_time_ns: Monotonic timestamp of request enqueue.
    """

    def __init__(
        self,
        request_id: str,
        input_tensor: Tensor,
        submission_time_ns: int,
    ) -> None:
        self.request_id: str = request_id
        self.input_tensor: Tensor = input_tensor
        self.submission_time_ns: int = submission_time_ns

        # Shape characteristics
        shape = input_tensor.shape
        if len(shape) == 1:
            self.batch_size: int = 1
            self.feature_shape: Tuple[int, ...] = (shape[0],)
            self.is_1d: bool = True
        else:
            self.batch_size = shape[0]
            self.feature_shape = shape[1:]
            self.is_1d = False

        self.dtype = input_tensor.dtype

        # Synchronization & completion state
        self._lock: threading.Lock = threading.Lock()
        self._event: threading.Event = threading.Event()
        self._result: Optional[Tensor] = None
        self._exception: Optional[Exception] = None
        self._is_cancelled: bool = False
        self._is_completed: bool = False
        self._is_batched: bool = False

    @property
    def is_cancelled(self) -> bool:
        with self._lock:
            return self._is_cancelled

    @property
    def is_completed(self) -> bool:
        with self._lock:
            return self._is_completed

    @property
    def is_batched(self) -> bool:
        with self._lock:
            return self._is_batched

    def mark_batched(self) -> bool:
        """Mark request as currently being processed in a batch (prevents late cancellation)."""
        with self._lock:
            if self._is_cancelled or self._is_completed:
                return False
            self._is_batched = True
            return True

    def cancel(self) -> bool:
        """Cancel request if still pending in the queue and not yet batched."""
        with self._lock:
            if self._is_batched or self._is_completed:
                return False
            self._is_cancelled = True
            self._is_completed = True
            self._event.set()
            return True

    def set_result(self, result: Tensor) -> None:
        """Deliver completed result tensor and notify waiting clients."""
        with self._lock:
            if self._is_completed:
                return
            self._result = result
            self._is_completed = True
            self._event.set()

    def set_exception(self, exc: Exception) -> None:
        """Deliver execution exception and notify waiting clients."""
        with self._lock:
            if self._is_completed:
                return
            self._exception = exc
            self._is_completed = True
            self._event.set()

    def wait(self, timeout: Optional[float] = None) -> Tensor:
        """Wait for request execution to complete and return the output tensor.

        Args:
            timeout: Maximum time in seconds to wait.

        Returns:
            Output Tensor representing prediction results.

        Raises:
            TimeoutError: If execution exceeds wait timeout.
            SchedulerError: If request was cancelled.
            Exception: Re-raises any exception encountered during runtime execution.
        """
        finished = self._event.wait(timeout=timeout)
        if not finished:
            raise TimeoutError(f"InferenceRequest '{self.request_id}' timed out after {timeout} seconds.")

        with self._lock:
            if self._is_cancelled:
                raise SchedulerError(f"InferenceRequest '{self.request_id}' was cancelled before execution.")
            if self._exception is not None:
                raise self._exception
            if self._result is not None:
                return self._result
            raise SchedulerError(f"InferenceRequest '{self.request_id}' completed with no result.")


class InferenceFuture:
    """Lightweight handle returned by asynchronous `InferenceScheduler.submit()` calls."""

    def __init__(self, request: InferenceRequest) -> None:
        self._request: InferenceRequest = request

    @property
    def request_id(self) -> str:
        """Unique identifier of the underlying request."""
        return self._request.request_id

    def result(self, timeout: Optional[float] = None) -> Tensor:
        """Wait for completion and return the resulting prediction Tensor."""
        return self._request.wait(timeout=timeout)

    def done(self) -> bool:
        """Return True if the prediction has completed, failed, or was cancelled."""
        return self._request.is_completed

    def exception(self, timeout: Optional[float] = None) -> Optional[Exception]:
        """Return the exception encountered during execution, or None if successful."""
        try:
            self._request.wait(timeout=timeout)
            return None
        except Exception as e:
            return e

    def cancel(self) -> bool:
        """Attempt to cancel request before dynamic batch assembly."""
        return self._request.cancel()

    def __repr__(self) -> str:
        status = "completed" if self._request.is_completed else "pending"
        return f"InferenceFuture(id='{self.request_id}', status='{status}')"


class InferenceScheduler:
    """Production request scheduler and dynamic batching manager for InferenceRuntime.

    Wraps an InferenceRuntime instance to aggregate individual client prediction calls
    into larger, hardware-efficient dynamic batches with bounded queue backpressure
    and millisecond-level batch timeout flushing.

    Args:
        runtime: Underlying InferenceRuntime instance.
        max_batch_size: Maximum batch size formed per execution.
        max_queue_size: Maximum pending requests before backpressure rejection.
        batch_timeout_ms: Maximum time to wait for a full batch before dispatching.
        config: Optional pre-configured SchedulerConfig.
        owns_runtime: If True, closing the scheduler also closes the underlying runtime.
    """

    def __init__(
        self,
        runtime: InferenceRuntime,
        max_batch_size: Optional[int] = None,
        max_queue_size: Optional[int] = None,
        batch_timeout_ms: Optional[float] = None,
        config: Optional[SchedulerConfig] = None,
        owns_runtime: bool = False,
    ) -> None:
        self._runtime: InferenceRuntime = runtime
        self._owns_runtime: bool = owns_runtime

        if config is not None:
            self._config: SchedulerConfig = config.copy()
        else:
            self._config = SchedulerConfig(
                max_batch_size=max_batch_size if max_batch_size is not None else 32,
                max_queue_size=max_queue_size if max_queue_size is not None else 128,
                batch_timeout_ms=batch_timeout_ms if batch_timeout_ms is not None else 2.0,
            )

        # Queue & synchronization state
        self._queue: collections.deque[InferenceRequest] = collections.deque()
        self._lock: threading.Lock = threading.Lock()
        self._cv: threading.Condition = threading.Condition(self._lock)
        self._state: SchedulerLifecycleState = SchedulerLifecycleState.RUNNING

        # Telemetry and counters
        self._stats_lock: threading.RLock = threading.RLock()
        self._metrics: MetricsCollector = MetricsCollector()
        self._request_counter: int = 0
        self._submitted_requests: int = 0
        self._completed_requests: int = 0
        self._failed_requests: int = 0
        self._rejected_requests: int = 0
        self._cancelled_requests: int = 0
        self._batches_formed: int = 0
        self._total_samples_processed: int = 0
        self._max_batch_size_observed: int = 0
        self._peak_queue_depth: int = 0
        self._total_queue_wait_ns: int = 0
        self._total_execution_ns: int = 0

        # Background worker thread
        self._worker_thread: threading.Thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="TensorForgeSchedulerWorker",
        )
        self._worker_thread.start()

    @property
    def runtime(self) -> InferenceRuntime:
        """Access the underlying InferenceRuntime instance."""
        return self._runtime

    @property
    def config(self) -> SchedulerConfig:
        """Return a copy of the active SchedulerConfig."""
        return self._config.copy()

    @property
    def lifecycle_state(self) -> str:
        """Current scheduler lifecycle state ('RUNNING', 'DRAINING', or 'CLOSED')."""
        with self._lock:
            return self._state.value

    @property
    def is_running(self) -> bool:
        """Whether the scheduler is running and actively accepting requests."""
        with self._lock:
            return self._state == SchedulerLifecycleState.RUNNING

    @property
    def is_closed(self) -> bool:
        """Whether the scheduler is closed."""
        with self._lock:
            return self._state == SchedulerLifecycleState.CLOSED

    @property
    def queue_depth(self) -> int:
        """Current number of pending requests in the scheduler queue."""
        with self._lock:
            return len(self._queue)

    def __enter__(self) -> InferenceScheduler:
        """Context manager support for InferenceScheduler."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with graceful shutdown."""
        self.close()

    def submit(
        self,
        input_data: Union[Tensor, np.ndarray, Sequence[Any]],
    ) -> InferenceFuture:
        """Asynchronously enqueue an inference request for dynamic batching.

        Args:
            input_data: Input tensor, NumPy array, or nested sequence.

        Returns:
            InferenceFuture handle for retrieving prediction results.

        Raises:
            SchedulerClosedError: If the scheduler is closed or draining.
            SchedulerQueueFullError: If the queue is at maximum capacity.
            TensorForgeInputError: If input data is malformed.
        """
        with self._lock:
            if self._state != SchedulerLifecycleState.RUNNING:
                raise SchedulerClosedError(
                    f"Cannot submit request: InferenceScheduler is {self._state.value}."
                )

            if len(self._queue) >= self._config.max_queue_size:
                with self._stats_lock:
                    self._rejected_requests += 1
                raise SchedulerQueueFullError(
                    f"Scheduler queue is full ({len(self._queue)}/{self._config.max_queue_size} pending requests)."
                )

        # Validate and prepare tensor
        if isinstance(input_data, Tensor):
            x = input_data
        elif isinstance(input_data, np.ndarray):
            x = Tensor(input_data, dtype=float32, copy=False)
        else:
            try:
                arr = np.asarray(input_data, dtype=np.float32)
                x = Tensor(arr, dtype=float32)
            except Exception as e:
                raise TensorForgeInputError(f"Failed to convert input to Tensor: {e}") from e

        if len(x.shape) == 0:
            raise TensorForgeInputError("0-dimensional scalar tensor cannot be queued for inference.")

        # Ensure 2D tensor for standardized row concatenation
        if len(x.shape) == 1:
            x_2d = x.reshape((1, x.shape[0]))
        else:
            x_2d = x

        t_now = time.perf_counter_ns()
        with self._lock:
            # Re-check state inside lock
            if self._state != SchedulerLifecycleState.RUNNING:
                raise SchedulerClosedError(
                    f"Cannot submit request: InferenceScheduler is {self._state.value}."
                )

            if len(self._queue) >= self._config.max_queue_size:
                with self._stats_lock:
                    self._rejected_requests += 1
                self._metrics.record_request_rejected("queue_full")
                raise SchedulerQueueFullError(
                    f"Scheduler queue is full ({len(self._queue)}/{self._config.max_queue_size} pending requests)."
                )

            with self._stats_lock:
                self._request_counter += 1
                req_id = f"sched-req-{self._request_counter}"
                self._submitted_requests += 1

            req = InferenceRequest(
                request_id=req_id,
                input_tensor=x_2d,
                submission_time_ns=t_now,
            )
            self._queue.append(req)

            q_len = len(self._queue)
            self._metrics.record_request_submitted(queue_depth=q_len)
            with self._stats_lock:
                self._peak_queue_depth = max(self._peak_queue_depth, q_len)

            # Wake worker immediately if batch capacity reached
            if q_len >= self._config.max_batch_size:
                self._cv.notify_all()
            else:
                self._cv.notify()

        return InferenceFuture(req)

    def predict(
        self,
        input_data: Union[Tensor, np.ndarray, Sequence[Any]],
        timeout: Optional[float] = None,
    ) -> Tensor:
        """Synchronously enqueue a request, wait for dynamic batch execution, and return results.

        Args:
            input_data: Input tensor, NumPy array, or nested sequence.
            timeout: Optional maximum wait time in seconds.

        Returns:
            Output Tensor representing prediction results.
        """
        future = self.submit(input_data)
        return future.result(timeout=timeout)

    def flush(self) -> None:
        """Explicitly notify worker to immediately process all currently pending requests."""
        with self._cv:
            self._cv.notify_all()

    def close(self, drain: Optional[bool] = None) -> None:
        """Gracefully shut down the scheduler.

        Args:
            drain: Whether to process pending requests before closing (defaults to config.drain_on_close).
        """
        should_drain = drain if drain is not None else self._config.drain_on_close

        with self._cv:
            if self._state == SchedulerLifecycleState.CLOSED:
                return

            if should_drain and len(self._queue) > 0:
                self._state = SchedulerLifecycleState.DRAINING
            else:
                self._state = SchedulerLifecycleState.CLOSED

            self._cv.notify_all()

        # Wait for worker thread to finish processing
        if self._worker_thread.is_alive() and threading.current_thread() != self._worker_thread:
            self._worker_thread.join(timeout=5.0)

        with self._cv:
            self._state = SchedulerLifecycleState.CLOSED
            # Cancel any remaining unprocessed requests
            while self._queue:
                req = self._queue.popleft()
                req.set_exception(
                    SchedulerClosedError("Scheduler closed before request could be processed.")
                )

        if self._owns_runtime:
            self._runtime.close()

    def _worker_loop(self) -> None:
        """Dedicated scheduler background loop forming and dispatching dynamic batches."""
        while True:
            batch_to_run: List[InferenceRequest] = []

            with self._cv:
                # 1. Clean up cancelled requests at head of queue
                while self._queue and self._queue[0].is_cancelled:
                    _ = self._queue.popleft()
                    with self._stats_lock:
                        self._cancelled_requests += 1

                # 2. Check shutdown conditions
                if self._state == SchedulerLifecycleState.CLOSED:
                    if not self._queue:
                        break

                if self._state == SchedulerLifecycleState.DRAINING:
                    if not self._queue:
                        self._state = SchedulerLifecycleState.CLOSED
                        break

                # 3. Wait for requests if queue is empty
                if not self._queue:
                    self._cv.wait(timeout=0.1)
                    continue

                # 4. Form batch according to size or timeout
                now_ns = time.perf_counter_ns()
                oldest_wait_ms = (now_ns - self._queue[0].submission_time_ns) / 1_000_000.0
                timeout_threshold_ms = self._config.batch_timeout_ms

                # Form batch if:
                # - Queue size reached max_batch_size
                # - Batch timeout expired on oldest request
                # - Scheduler is DRAINING
                # - Batch capacity filled by accumulated samples
                total_samples = 0
                for r in self._queue:
                    if not r.is_cancelled:
                        total_samples += r.batch_size

                should_dispatch = (
                    total_samples >= self._config.max_batch_size
                    or oldest_wait_ms >= timeout_threshold_ms
                    or self._state in (SchedulerLifecycleState.DRAINING, SchedulerLifecycleState.CLOSED)
                )

                if not should_dispatch:
                    # Sleep for remaining timeout duration
                    time_to_wait_sec = max(0.0001, (timeout_threshold_ms - oldest_wait_ms) / 1000.0)
                    self._cv.wait(timeout=time_to_wait_sec)
                    continue

                # 5. Extract compatible batch
                first_req = None
                for _ in range(len(self._queue)):
                    if not self._queue:
                        break
                    candidate = self._queue[0]
                    if candidate.is_cancelled:
                        self._queue.popleft()
                        with self._stats_lock:
                            self._cancelled_requests += 1
                        continue

                    if first_req is None:
                        first_req = candidate
                        if candidate.mark_batched():
                            batch_to_run.append(self._queue.popleft())
                    else:
                        # Check compatibility
                        if candidate.feature_shape == first_req.feature_shape and candidate.dtype == first_req.dtype:
                            curr_batch_samples = sum(r.batch_size for r in batch_to_run)
                            if curr_batch_samples + candidate.batch_size <= self._config.max_batch_size:
                                if candidate.mark_batched():
                                    batch_to_run.append(self._queue.popleft())
                            else:
                                break
                        else:
                            # Incompatible shape/dtype: leave for next batch
                            break

            # If no requests were extracted (e.g. all cancelled), continue
            if not batch_to_run:
                continue

            # 6. Execute batch through runtime outside of scheduler locks
            self._execute_batch(batch_to_run)

    def _execute_batch(self, batch_requests: List[InferenceRequest]) -> None:
        """Concatenate, execute, and demultiplex a batch of inference requests."""
        now_ns = time.perf_counter_ns()
        total_batch_samples = sum(req.batch_size for req in batch_requests)

        # Track queue wait duration
        for req in batch_requests:
            wait_ns = now_ns - req.submission_time_ns
            with self._stats_lock:
                self._total_queue_wait_ns += wait_ns

        # Record batch formation
        self._metrics.record_batch(
            batch_size=total_batch_samples,
            configured_max_batch=self._config.max_batch_size,
        )

        try:
            # 1. Assemble combined batch tensor
            if len(batch_requests) == 1:
                batch_tensor = batch_requests[0].input_tensor
            else:
                # Concatenate along batch dimension
                arrays = [req.input_tensor.numpy() for req in batch_requests]
                combined_arr = np.concatenate(arrays, axis=0)
                batch_tensor = Tensor(combined_arr, dtype=batch_requests[0].dtype, copy=False)

            # 2. Execute forward prediction through underlying InferenceRuntime
            t_exec_0 = time.perf_counter_ns()
            batch_output = self._runtime.predict(batch_tensor)
            t_exec_1 = time.perf_counter_ns()

            exec_duration_ns = t_exec_1 - t_exec_0
            exec_ms = exec_duration_ns / 1_000_000.0

            with self._stats_lock:
                self._total_execution_ns += exec_duration_ns

            # 3. Demultiplex output tensor to individual requests
            out_arr = batch_output.numpy()
            curr_row = 0
            for req in batch_requests:
                req_rows = req.batch_size
                sliced_arr = out_arr[curr_row : curr_row + req_rows]
                curr_row += req_rows

                # Restore original 1D shape if request was 1D
                if req.is_1d and len(sliced_arr.shape) == 2 and sliced_arr.shape[0] == 1:
                    req_out = Tensor(sliced_arr.reshape(-1), dtype=batch_output.dtype, copy=True)
                else:
                    req_out = Tensor(sliced_arr, dtype=batch_output.dtype, copy=True)

                req.set_result(req_out)

                q_wait_ms = (t_exec_0 - req.submission_time_ns) / 1_000_000.0
                e2e_ms = (t_exec_1 - req.submission_time_ns) / 1_000_000.0
                self._metrics.record_request_completed(
                    queue_wait_ms=q_wait_ms,
                    exec_ms=exec_ms,
                    e2e_ms=e2e_ms,
                    samples=req.batch_size,
                )

            self._metrics.record_backend(
                backend=self._runtime.backend,
                is_fused=self._runtime.is_optimized or self._runtime.is_compiled,
                is_compiled=self._runtime.is_compiled,
            )

            with self._stats_lock:
                self._batches_formed += 1
                self._completed_requests += len(batch_requests)
                self._total_samples_processed += total_batch_samples
                self._max_batch_size_observed = max(self._max_batch_size_observed, total_batch_samples)

        except Exception as e:
            t_err = time.perf_counter_ns()
            err_ms = (t_err - now_ns) / 1_000_000.0
            # Broadcast exception to all requests in the batch
            for req in batch_requests:
                req.set_exception(e)
                self._metrics.record_request_failed(exec_ms=err_ms)
            with self._stats_lock:
                self._failed_requests += len(batch_requests)

    def health(self) -> Dict[str, Any]:
        """Perform a lightweight operational health check on the scheduler.

        Returns:
            Dictionary with lifecycle state, queue depths, and batch statistics.
        """
        with self._lock:
            q_depth = len(self._queue)
            state_val = self._state.value

        with self._stats_lock:
            batches = self._batches_formed
            samples = self._total_samples_processed
            avg_batch_sz = (samples / batches) if batches > 0 else 0.0

            return {
                "status": "closed" if state_val == "CLOSED" else "healthy",
                "lifecycle_state": state_val,
                "accepting_requests": state_val == "RUNNING",
                "queue_depth": q_depth,
                "max_queue_size": self._config.max_queue_size,
                "max_batch_size": self._config.max_batch_size,
                "batches_formed": batches,
                "requests_submitted": self._submitted_requests,
                "requests_completed": self._completed_requests,
                "requests_failed": self._failed_requests,
                "requests_rejected": self._rejected_requests,
                "requests_cancelled": self._cancelled_requests,
                "average_batch_size": avg_batch_sz,
                "peak_queue_depth": self._peak_queue_depth,
                "runtime_health": self._runtime.health(),
            }

    def stats(self) -> Dict[str, Any]:
        """Generate an extended diagnostic and statistical report of scheduler telemetry."""
        h = self.health()
        with self._stats_lock:
            batches = self._batches_formed
            samples = self._total_samples_processed
            completed = self._completed_requests
            avg_batch_sz = (samples / batches) if batches > 0 else 0.0
            avg_wait_ms = (self._total_queue_wait_ns / completed / 1e6) if completed > 0 else 0.0
            avg_exec_ms = (self._total_execution_ns / batches / 1e6) if batches > 0 else 0.0

            return {
                "health": h,
                "lifecycle_state": h["lifecycle_state"],
                "config": self._config.to_dict(),
                "submitted_requests": self._submitted_requests,
                "completed_requests": self._completed_requests,
                "failed_requests": self._failed_requests,
                "rejected_requests": self._rejected_requests,
                "cancelled_requests": self._cancelled_requests,
                "batches_formed": self._batches_formed,
                "total_samples_processed": self._total_samples_processed,
                "average_batch_size": avg_batch_sz,
                "max_batch_size_observed": self._max_batch_size_observed,
                "peak_queue_depth": self._peak_queue_depth,
                "avg_queue_wait_ms": avg_wait_ms,
                "avg_batch_execution_ms": avg_exec_ms,
                "runtime_stats": self._runtime.stats(),
                "tensorforge_version": "1.7.0",
            }

    def performance_snapshot(self) -> PerformanceSnapshot:
        """Generate an immutable, comprehensive PerformanceSnapshot for the scheduler.

        Returns:
            PerformanceSnapshot containing requests, batches, latency, throughput,
            backend, compiler, memory, and scheduler analytics.
        """
        # Sync scheduler configuration and queue state
        self._metrics.set_scheduler_metrics(
            SchedulerMetrics(
                queue_depth=self.queue_depth,
                max_queue_size=self._config.max_queue_size,
                max_batch_size=self._config.max_batch_size,
                batch_timeout_ms=self._config.batch_timeout_ms,
                policy=self._config.policy.value,
                lifecycle_state=self.lifecycle_state,
            )
        )

        # Sync runtime memory metrics
        param_bytes = 0
        try:
            from tensorforge.serialization.checkpoint import compute_model_size
            param_bytes = compute_model_size(self._runtime.model).get("parameter_bytes", 0)
        except Exception:
            pass

        self._metrics.record_memory(
            workspace_bytes=self._runtime.workspace_size,
            planned_bytes=self._runtime.workspace_size,
            param_bytes=param_bytes,
            model_size_bytes=param_bytes,
        )

        # Sync compiler analytics if active
        if self._runtime.is_compiled:
            from tensorforge.inference.compiler import InferenceCompiler
            c_stats = InferenceCompiler.cache_stats()
            self._metrics._cache_hits = c_stats.get("hits", 0)
            self._metrics._cache_misses = c_stats.get("misses", 0)
            self._metrics._compile_requests = c_stats.get("total_lookups", 0)

        return self._metrics.snapshot()

    def metrics(self) -> PerformanceSnapshot:
        """Alias for performance_snapshot()."""
        return self.performance_snapshot()

    def export_metrics(self, filepath: str, indent: int = 2) -> None:
        """Export current scheduler performance snapshot to a JSON file.

        Args:
            filepath: Destination file path.
            indent: JSON indentation spaces (default: 2).
        """
        self.performance_snapshot().save_json(filepath, indent=indent)

    def reset_metrics(self) -> InferenceScheduler:
        """Reset all metrics collector counters, latency reservoirs, and timers.

        Returns:
            Self (enables method chaining).
        """
        self._metrics.reset()
        return self

    @property
    def metrics_collector(self) -> MetricsCollector:
        """Access the underlying MetricsCollector instance."""
        return self._metrics

    def __repr__(self) -> str:
        return (
            f"InferenceScheduler(\n"
            f"  state='{self.lifecycle_state}',\n"
            f"  queue_depth={self.queue_depth}/{self._config.max_queue_size},\n"
            f"  max_batch_size={self._config.max_batch_size},\n"
            f"  batch_timeout_ms={self._config.batch_timeout_ms},\n"
            f"  runtime={repr(self._runtime)}\n"
            f")"
        )
