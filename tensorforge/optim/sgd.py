"""Stochastic Gradient Descent (SGD) optimizer for TensorForge."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional
import numpy as np

from tensorforge.optim.optimizer import Optimizer
from tensorforge.tensor.tensor import Tensor


class SGD(Optimizer):
    """Implements Stochastic Gradient Descent optimizer with optional momentum and weight decay.

    Update equations:
        If weight_decay != 0:
            grad = grad + weight_decay * parameter
        If momentum != 0:
            v = momentum * v + grad
            parameter = parameter - lr * v
        Else:
            parameter = parameter - lr * grad

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (must be > 0.0).
        momentum: Momentum factor (default: 0.0).
        weight_decay: Weight decay (L2 penalty) factor (default: 0.0).
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr} (must be > 0.0)")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum} (must be >= 0.0)")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay} (must be >= 0.0)")

        defaults = dict(
            lr=float(lr),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        )
        super().__init__(params, defaults)

    def step(self) -> None:
        """Perform a single SGD optimization step."""
        for group in self.param_groups:
            lr: float = group["lr"]
            momentum: float = group["momentum"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad_np = p.grad.numpy()
                p_np = p.numpy()

                if weight_decay != 0.0:
                    d_p = grad_np + weight_decay * p_np
                else:
                    d_p = grad_np.copy()

                if momentum != 0.0:
                    param_state = self.state.setdefault(p, {})
                    if "momentum_buffer" not in param_state:
                        buf = d_p.copy()
                        param_state["momentum_buffer"] = buf
                    else:
                        buf = param_state["momentum_buffer"]
                        buf *= momentum
                        buf += d_p
                    update_step = buf
                else:
                    update_step = d_p

                new_p = p_np - lr * update_step
                np.copyto(p.storage.to_numpy(), new_p.reshape(-1).astype(p.dtype.numpy_dtype, copy=False))
