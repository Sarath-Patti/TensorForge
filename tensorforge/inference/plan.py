"""Execution Plan Intermediate Representation for compiled TensorForge inference."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from tensorforge.tensor.dtype import DType, float32


class ExecutionStep:
    """A pre-resolved, deterministic instruction inside a compiled ExecutionPlan."""

    def __init__(
        self,
        step_index: int,
        op_type: str,
        input_slot: int,
        output_slot: int,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        backend_dispatch: str,
        params: Optional[Dict[str, Any]] = None,
        attrs: Optional[Dict[str, Any]] = None,
        dtype: DType = float32,
        is_quantized: bool = False,
    ) -> None:
        self.step_index: int = step_index
        self.op_type: str = op_type
        self.input_slot: int = input_slot
        self.output_slot: int = output_slot
        self.input_shape: Tuple[int, ...] = tuple(input_shape)
        self.output_shape: Tuple[int, ...] = tuple(output_shape)
        self.backend_dispatch: str = backend_dispatch
        self.params: Dict[str, Any] = params or {}
        self.attrs: Dict[str, Any] = attrs or {}
        self.dtype: DType = dtype
        self.is_quantized: bool = is_quantized

    def __repr__(self) -> str:
        in_desc = f"slot_{self.input_slot}" if self.input_slot >= 0 else "user_input"
        out_desc = f"slot_{self.output_slot}"
        quant_str = ", quantized=True" if self.is_quantized else ""
        return (
            f"ExecutionStep[{self.step_index}]({self.op_type}, {in_desc} -> {out_desc}, "
            f"shape={self.input_shape}->{self.output_shape}, backend='{self.backend_dispatch}'{quant_str})"
        )


class ExecutionPlan:
    """Compiled, reusable inference execution plan holding ordered steps and memory workspace layouts."""

    def __init__(
        self,
        steps: List[ExecutionStep],
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        workspace_plan: Dict[str, Any],
        target_backend: str,
        dtype: DType = float32,
        is_quantized: bool = False,
    ) -> None:
        self.steps: List[ExecutionStep] = list(steps)
        self.input_shape: Tuple[int, ...] = tuple(input_shape)
        self.output_shape: Tuple[int, ...] = tuple(output_shape)
        self.workspace_plan: Dict[str, Any] = workspace_plan
        self.target_backend: str = target_backend
        self.dtype: DType = dtype
        self.is_quantized: bool = is_quantized

    @property
    def total_workspace_bytes(self) -> int:
        """Total memory required in the workspace arena in bytes."""
        return int(self.workspace_plan.get("total_workspace_bytes", 0))

    @property
    def num_steps(self) -> int:
        """Total number of execution steps in the plan."""
        return len(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int) -> ExecutionStep:
        return self.steps[idx]

    def __iter__(self):
        return iter(self.steps)

    def summary(self) -> str:
        """Format a human-readable summary of the compiled plan."""
        lines = [
            f"ExecutionPlan (Steps: {len(self.steps)}, Backend: '{self.target_backend}', "
            f"Workspace: {self.total_workspace_bytes} bytes, Quantized: {self.is_quantized}):",
            f"  Input Shape:  {self.input_shape}",
            f"  Output Shape: {self.output_shape}",
            "  Execution Steps:",
        ]
        for step in self.steps:
            lines.append(f"    - {step}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
