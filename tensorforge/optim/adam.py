"""Adam optimizer for TensorForge."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple
import numpy as np

from tensorforge.optim.optimizer import Optimizer
from tensorforge.tensor.tensor import Tensor


class Adam(Optimizer):
    """Implements Adam (Adaptive Moment Estimation) optimizer.

    Update equations:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999)).
        eps: Term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay: Weight decay (L2 penalty) factor (default: 0.0).
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr} (must be > 0.0)")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps} (must be > 0.0)")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]} (must be in [0, 1))")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]} (must be in [0, 1))")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay} (must be >= 0.0)")

        defaults = dict(
            lr=float(lr),
            betas=betas,
            eps=float(eps),
            weight_decay=float(weight_decay),
        )
        super().__init__(params, defaults)

    def step(self) -> None:
        """Perform a single Adam optimization step."""
        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad_np = p.grad.numpy().astype(np.float64, copy=False)
                p_np = p.numpy().astype(np.float64, copy=False)

                if weight_decay != 0.0:
                    grad_np = grad_np + weight_decay * p_np

                param_state = self.state.setdefault(p, {})
                if len(param_state) == 0:
                    param_state["step"] = 0
                    param_state["exp_avg"] = np.zeros_like(p_np, dtype=np.float64)
                    param_state["exp_avg_sq"] = np.zeros_like(p_np, dtype=np.float64)

                param_state["step"] += 1
                step_t = param_state["step"]

                exp_avg = param_state["exp_avg"]
                exp_avg_sq = param_state["exp_avg_sq"]

                # Update biased first moment estimate
                exp_avg *= beta1
                exp_avg += (1.0 - beta1) * grad_np

                # Update biased second raw moment estimate
                exp_avg_sq *= beta2
                exp_avg_sq += (1.0 - beta2) * (grad_np * grad_np)

                # Compute bias-corrected first and second moment estimates
                bias_correction1 = 1.0 - (beta1 ** step_t)
                bias_correction2 = 1.0 - (beta2 ** step_t)

                step_size = lr / bias_correction1
                denom = np.sqrt(exp_avg_sq / bias_correction2) + eps

                update_step = step_size * (exp_avg / denom)
                new_p = p_np - update_step
                np.copyto(p.storage.to_numpy(), new_p.astype(p.dtype.numpy_dtype, copy=False))
