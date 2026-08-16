"""Inference memory planning and interval lifetime reuse engine for TensorForge."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from tensorforge.tensor.dtype import DType, float32, to_dtype


class BufferLifetime:
    """Represents the liveness interval and memory requirements of an intermediate tensor buffer."""

    def __init__(
        self,
        buffer_id: int,
        size_bytes: int,
        first_use: int,
        last_use: int,
        shape: Tuple[int, ...] = (),
        dtype: DType = float32,
    ) -> None:
        self.buffer_id: int = buffer_id
        self.size_bytes: int = size_bytes
        self.first_use: int = first_use
        self.last_use: int = last_use
        self.shape: Tuple[int, ...] = tuple(shape)
        self.dtype: DType = dtype
        self.region_id: Optional[int] = None
        self.offset_bytes: int = 0

    def overlaps_with(self, other: BufferLifetime) -> bool:
        """Check if this buffer's active lifetime interval overlaps with another buffer."""
        return not (self.last_use <= other.first_use or other.last_use <= self.first_use)

    def __repr__(self) -> str:
        return (
            f"BufferLifetime(id={self.buffer_id}, interval=[{self.first_use}, {self.last_use}], "
            f"size={self.size_bytes}B, region={self.region_id}, offset={self.offset_bytes}B)"
        )


class MemoryRegion:
    """A physical or virtual memory region allocated in the workspace arena shared among non-overlapping buffers."""

    def __init__(self, region_id: int, alignment: int = 64) -> None:
        self.region_id: int = region_id
        self.alignment: int = alignment
        self.size_bytes: int = 0
        self.offset_bytes: int = 0
        self.assigned_buffers: List[int] = []
        self.intervals: List[Tuple[int, int]] = []

    def can_accommodate(self, first_use: int, last_use: int) -> bool:
        """Check if an interval [first_use, last_use] does not overlap with any active interval in this region."""
        for start, end in self.intervals:
            # Overlap occurs if start < last_use and first_use < end
            if not (end <= first_use or last_use <= start):
                return False
        return True

    def assign_buffer(self, buffer: BufferLifetime) -> None:
        """Assign a buffer to this region and expand the region's size if necessary."""
        self.assigned_buffers.append(buffer.buffer_id)
        self.intervals.append((buffer.first_use, buffer.last_use))
        self.size_bytes = max(self.size_bytes, buffer.size_bytes)
        buffer.region_id = self.region_id

    def __repr__(self) -> str:
        return (
            f"MemoryRegion(id={self.region_id}, size={self.size_bytes}B, "
            f"offset={self.offset_bytes}B, buffers={self.assigned_buffers})"
        )


class PlannedBuffer:
    """Descriptor for an allocated or shared memory buffer slot (backward compatible with v1.1)."""

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


class MemoryPlan:
    """Complete, detailed memory allocation plan with lifetime analysis and region allocation."""

    def __init__(
        self,
        regions: Dict[int, MemoryRegion],
        buffers: Dict[int, BufferLifetime],
        step_input_slots: List[int],
        step_output_slots: List[int],
        total_workspace_bytes: int,
        peak_intermediate_bytes: int,
        alignment_padding_bytes: int = 0,
    ) -> None:
        self.regions: Dict[int, MemoryRegion] = regions
        self.buffers: Dict[int, BufferLifetime] = buffers
        self.step_input_slots: List[int] = step_input_slots
        self.step_output_slots: List[int] = step_output_slots
        self.total_workspace_bytes: int = total_workspace_bytes
        self.peak_intermediate_bytes: int = peak_intermediate_bytes
        self.alignment_padding_bytes: int = alignment_padding_bytes

    @property
    def num_regions(self) -> int:
        """Total number of physical reusable memory regions allocated in workspace."""
        return len(self.regions)

    @property
    def num_reused_buffers(self) -> int:
        """Number of intermediate buffers that share memory regions."""
        return len(self.buffers)

    @property
    def buffer_region_map(self) -> Dict[int, int]:
        """Mapping from buffer ID to assigned region ID."""
        return {buf_id: buf.region_id for buf_id, buf in self.buffers.items() if buf.region_id is not None}

    def summary(self) -> Dict[str, Any]:
        """Dictionary summary of the memory plan."""
        return {
            "peak_workspace_bytes": self.total_workspace_bytes,
            "peak_workspace_kb": self.total_workspace_bytes / 1024.0,
            "num_regions": self.num_regions,
            "num_reused_buffers": self.num_reused_buffers,
            "alignment_padding_bytes": self.alignment_padding_bytes,
            "step_input_slots": self.step_input_slots,
            "step_output_slots": self.step_output_slots,
            "region_sizes": {r_id: r.size_bytes for r_id, r in self.regions.items()},
            "region_offsets": {r_id: r.offset_bytes for r_id, r in self.regions.items()},
        }

    def __repr__(self) -> str:
        return (
            f"MemoryPlan(total={self.total_workspace_bytes}B, regions={self.num_regions}, "
            f"reused_buffers={self.num_reused_buffers}, padding={self.alignment_padding_bytes}B)"
        )


class MemoryPlanner:
    """Plans workspace memory allocations by performing interval lifetime analysis and region reuse."""

    @classmethod
    def plan_workspace(
        cls,
        shape_flow: List[Tuple[Tuple[int, ...], Tuple[int, ...]]],
        dtype: DType = float32,
        alignment: int = 64,
    ) -> MemoryPlan:
        """Compute an optimized workspace allocation plan using interval lifetime analysis.

        Performs graph coloring / interval scheduling over intermediate tensor lifespans
        to minimize total physical workspace memory while supporting general execution topologies.

        Args:
            shape_flow: List of (node_input_shape, node_output_shape) pairs.
            dtype: Data type of intermediate tensors.
            alignment: Byte alignment for memory offsets (default: 64 bytes).

        Returns:
            Configured MemoryPlan instance.
        """
        num_steps = len(shape_flow)
        itemsize = dtype.itemsize

        if num_steps == 0:
            return MemoryPlan(
                regions={},
                buffers={},
                step_input_slots=[],
                step_output_slots=[],
                total_workspace_bytes=0,
                peak_intermediate_bytes=0,
                alignment_padding_bytes=0,
            )

        # 1. Build Buffer Lifetimes
        # In a sequential pipeline:
        # Step i produces Buffer i.
        # Buffer i is created at step i and consumed at step i + 1.
        buffers: Dict[int, BufferLifetime] = {}
        for i, (_, out_shape) in enumerate(shape_flow):
            out_numel = math.prod(out_shape)
            out_bytes = int(out_numel * itemsize)
            first_use = i
            last_use = i + 1
            buffers[i] = BufferLifetime(
                buffer_id=i,
                size_bytes=out_bytes,
                first_use=first_use,
                last_use=last_use,
                shape=out_shape,
                dtype=dtype,
            )

        # 2. Assign Buffers to Memory Regions using Interval Scheduling
        regions: Dict[int, MemoryRegion] = {}
        region_counter = 0

        for buf_id in sorted(buffers.keys(), key=lambda b: buffers[b].first_use):
            buf = buffers[buf_id]
            assigned = False

            # Try to assign to an existing non-overlapping region
            for r_id in sorted(regions.keys()):
                region = regions[r_id]
                if region.can_accommodate(buf.first_use, buf.last_use):
                    region.assign_buffer(buf)
                    assigned = True
                    break

            # If all existing regions overlap, allocate a new memory region
            if not assigned:
                new_region = MemoryRegion(region_id=region_counter, alignment=alignment)
                new_region.assign_buffer(buf)
                regions[region_counter] = new_region
                region_counter += 1

        # 3. Calculate 64-Byte Aligned Memory Offsets
        current_offset = 0
        total_padding = 0

        for r_id in sorted(regions.keys()):
            region = regions[r_id]
            aligned_offset = (current_offset + (alignment - 1)) & ~(alignment - 1)
            padding = aligned_offset - current_offset
            total_padding += padding

            region.offset_bytes = aligned_offset
            current_offset = aligned_offset + region.size_bytes

            # Update offset for all buffers assigned to this region
            for buf_id in region.assigned_buffers:
                buffers[buf_id].offset_bytes = aligned_offset

        total_workspace_bytes = (current_offset + (alignment - 1)) & ~(alignment - 1) if current_offset > 0 else 0
        peak_intermediate_bytes = sum(r.size_bytes for r in regions.values())

        # Map step input/output slot IDs to assigned region IDs
        step_input_slots: List[int] = []
        step_output_slots: List[int] = []

        for i in range(num_steps):
            in_slot = -1 if i == 0 else buffers[i - 1].region_id
            out_slot = buffers[i].region_id
            step_input_slots.append(in_slot if in_slot is not None else -1)
            step_output_slots.append(out_slot if out_slot is not None else 0)

        return MemoryPlan(
            regions=regions,
            buffers=buffers,
            step_input_slots=step_input_slots,
            step_output_slots=step_output_slots,
            total_workspace_bytes=total_workspace_bytes,
            peak_intermediate_bytes=peak_intermediate_bytes,
            alignment_padding_bytes=total_padding,
        )

    @classmethod
    def plan_sequential_workspace(
        cls,
        shape_flow: List[Tuple[Tuple[int, ...], Tuple[int, ...]]],
        dtype: DType = float32,
        alignment: int = 64,
    ) -> Dict[str, Any]:
        """Compute an optimized workspace allocation plan for a sequential node pipeline.

        Backward-compatible dictionary format wrapper around MemoryPlan.
        """
        plan = cls.plan_workspace(shape_flow=shape_flow, dtype=dtype, alignment=alignment)
        summary = plan.summary()

        return {
            "step_input_slots": plan.step_input_slots,
            "step_output_slots": plan.step_output_slots,
            "slot_sizes": summary["region_sizes"],
            "slot_offsets": summary["region_offsets"],
            "total_workspace_bytes": plan.total_workspace_bytes,
            "peak_intermediate_bytes": plan.peak_intermediate_bytes,
            "num_reused_slots": plan.num_regions,
            "num_regions": plan.num_regions,
            "num_reused_buffers": plan.num_reused_buffers,
            "alignment_padding_bytes": plan.alignment_padding_bytes,
            "memory_plan": plan,
        }
