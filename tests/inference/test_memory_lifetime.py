"""Tests for interval-based memory lifetime analysis and memory region reuse."""

import pytest
from tensorforge.inference.memory import BufferLifetime, MemoryPlan, MemoryPlanner, MemoryRegion
from tensorforge.tensor.dtype import float32


def test_buffer_lifetime_overlap():
    b1 = BufferLifetime(buffer_id=0, size_bytes=1024, first_use=0, last_use=2)
    b2 = BufferLifetime(buffer_id=1, size_bytes=2048, first_use=1, last_use=3)
    b3 = BufferLifetime(buffer_id=2, size_bytes=512, first_use=2, last_use=4)

    assert b1.overlaps_with(b2) is True
    assert b2.overlaps_with(b3) is True
    assert b1.overlaps_with(b3) is False


def test_memory_region_assignment():
    region = MemoryRegion(region_id=0, alignment=64)
    b1 = BufferLifetime(buffer_id=0, size_bytes=1000, first_use=0, last_use=1)
    b2 = BufferLifetime(buffer_id=1, size_bytes=2000, first_use=1, last_use=2)

    assert region.can_accommodate(b1.first_use, b1.last_use) is True
    region.assign_buffer(b1)
    assert region.size_bytes == 1000

    assert region.can_accommodate(b2.first_use, b2.last_use) is True
    region.assign_buffer(b2)
    assert region.size_bytes == 2000
    assert region.assigned_buffers == [0, 1]


def test_memory_planner_interval_coloring():
    # 5-step pipeline
    shape_flow = [
        ((8, 16), (8, 64)),   # Buf 0: [0, 1], size = 2048
        ((8, 64), (8, 128)),  # Buf 1: [1, 2], size = 4096
        ((8, 128), (8, 64)),  # Buf 2: [2, 3], size = 2048
        ((8, 64), (8, 32)),   # Buf 3: [3, 4], size = 1024
        ((8, 32), (8, 4)),    # Buf 4: [4, 5], size = 128
    ]

    plan = MemoryPlanner.plan_workspace(shape_flow, dtype=float32, alignment=64)

    assert isinstance(plan, MemoryPlan)
    assert plan.num_regions == 2
    assert plan.num_reused_buffers == 5

    # Buffers 0, 2, 4 share region 0
    # Buffers 1, 3 share region 1
    assert plan.buffers[0].region_id == 0
    assert plan.buffers[1].region_id == 1
    assert plan.buffers[2].region_id == 0
    assert plan.buffers[3].region_id == 1
    assert plan.buffers[4].region_id == 0

    # Offsets are 64-byte aligned
    for region in plan.regions.values():
        assert region.offset_bytes % 64 == 0

    summary = plan.summary()
    assert summary["peak_workspace_bytes"] == plan.total_workspace_bytes
    assert summary["num_regions"] == 2
