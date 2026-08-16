"""Inference memory planning and buffer lifetime reuse engine for TensorForge."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from tensorforge.tensor.dtype import DType, float32, to_dtype


class PlannedBuffer:
    """Descriptor for an allocated or shared memory buffer slot."""

    def __init__(
        self,
        slot_id: int,
        size_bytes: int,
        offset_bytes: int = 0,
    ) -> None:
        self.slot_id: int = slot_id
        self.size_bytes: int = size_bytes
        self.offset_bytes: int = offset_bytes

    def __repr__(self) -> str:
        return f"PlannedBuffer(slot_id={self.slot_id}, size={self.size_bytes}B, offset={self.offset_bytes}B)"


class MemoryPlanner:
    """Plans workspace memory allocations by performing lifetime analysis and buffer reuse."""

    @classmethod
    def plan_sequential_workspace(
        cls,
        shape_flow: List[Tuple[Tuple[int, ...], Tuple[int, ...]]],
        dtype: DType = float32,
        alignment: int = 64,
    ) -> Dict[str, Any]:
        """Compute an optimized workspace allocation plan for a sequential node pipeline.

        Performs liveness analysis on intermediate tensor buffers and assigns ping-pong /
        reusable memory slots to prevent memory churn.

        Args:
            shape_flow: List of (node_input_shape, node_output_shape) pairs.
            dtype: Data type of intermediate tensors.
            alignment: Byte alignment for memory offsets (default: 64 bytes).

        Returns:
            Dictionary containing:
                - 'step_input_slots': List of input slot IDs per step (-1 = external input).
                - 'step_output_slots': List of output slot IDs per step.
                - 'slot_sizes': Dict mapping slot ID to required byte capacity.
                - 'slot_offsets': Dict mapping slot ID to aligned offset in workspace arena.
                - 'total_workspace_bytes': Total contiguous workspace capacity needed.
                - 'peak_intermediate_bytes': Peak active intermediate memory.
                - 'num_reused_slots': Number of distinct memory slots.
        """
        num_steps = len(shape_flow)
        itemsize = dtype.itemsize

        if num_steps == 0:
            return {
                "step_input_slots": [],
                "step_output_slots": [],
                "slot_sizes": {},
                "slot_offsets": {},
                "total_workspace_bytes": 0,
                "peak_intermediate_bytes": 0,
                "num_reused_slots": 0,
            }

        # Sequential pipeline requires at most 2 intermediate slots (ping-pong)
        # Step 0: Input (-1) -> Slot 0
        # Step 1: Slot 0 -> Slot 1
        # Step 2: Slot 1 -> Slot 0
        # Step 3: Slot 0 -> Slot 1 ...
        step_input_slots: List[int] = []
        step_output_slots: List[int] = []
        slot_max_bytes: Dict[int, int] = {0: 0, 1: 0}

        for i, (_, out_shape) in enumerate(shape_flow):
            in_slot = -1 if i == 0 else ((i - 1) % 2)
            out_slot = i % 2

            step_input_slots.append(in_slot)
            step_output_slots.append(out_slot)

            out_numel = math.prod(out_shape)
            out_bytes = int(out_numel * itemsize)
            slot_max_bytes[out_slot] = max(slot_max_bytes[out_slot], out_bytes)

        # If only 1 step exists, we only need Slot 0
        if num_steps == 1:
            slot_max_bytes.pop(1, None)

        # Calculate aligned offsets within workspace
        slot_offsets: Dict[int, int] = {}
        current_offset = 0

        for slot_id in sorted(slot_max_bytes.keys()):
            # Align offset
            aligned_offset = (current_offset + (alignment - 1)) & ~(alignment - 1)
            slot_offsets[slot_id] = aligned_offset
            current_offset = aligned_offset + slot_max_bytes[slot_id]

        total_workspace_bytes = (current_offset + (alignment - 1)) & ~(alignment - 1) if current_offset > 0 else 0
        peak_intermediate_bytes = sum(slot_max_bytes.values())

        return {
            "step_input_slots": step_input_slots,
            "step_output_slots": step_output_slots,
            "slot_sizes": slot_max_bytes,
            "slot_offsets": slot_offsets,
            "total_workspace_bytes": total_workspace_bytes,
            "peak_intermediate_bytes": peak_intermediate_bytes,
            "num_reused_slots": len(slot_max_bytes),
        }
