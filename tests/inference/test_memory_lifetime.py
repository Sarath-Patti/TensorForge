"""Tests for interval-based memory lifetime analysis and memory region reuse."""

import unittest
from tensorforge.inference.memory import BufferLifetime, MemoryPlan, MemoryPlanner, MemoryRegion
from tensorforge.tensor.dtype import float32


class TestMemoryLifetime(unittest.TestCase):

    def test_buffer_lifetime_overlap(self):
        b1 = BufferLifetime(buffer_id=0, size_bytes=1024, first_use=0, last_use=2)
        b2 = BufferLifetime(buffer_id=1, size_bytes=2048, first_use=1, last_use=3)
        b3 = BufferLifetime(buffer_id=2, size_bytes=512, first_use=2, last_use=4)

        self.assertTrue(b1.overlaps_with(b2))
        self.assertTrue(b2.overlaps_with(b3))
        self.assertFalse(b1.overlaps_with(b3))

    def test_memory_region_assignment(self):
        region = MemoryRegion(region_id=0, alignment=64)
        b1 = BufferLifetime(buffer_id=0, size_bytes=1000, first_use=0, last_use=1)
        b2 = BufferLifetime(buffer_id=1, size_bytes=2000, first_use=1, last_use=2)

        self.assertTrue(region.can_accommodate(b1.first_use, b1.last_use))
        region.assign_buffer(b1)
        self.assertEqual(region.size_bytes, 1000)

        self.assertTrue(region.can_accommodate(b2.first_use, b2.last_use))
        region.assign_buffer(b2)
        self.assertEqual(region.size_bytes, 2000)
        self.assertEqual(region.assigned_buffers, [0, 1])

    def test_memory_planner_interval_coloring(self):
        # 5-step pipeline
        shape_flow = [
            ((8, 16), (8, 64)),   # Buf 0: [0, 1], size = 2048
            ((8, 64), (8, 128)),  # Buf 1: [1, 2], size = 4096
            ((8, 128), (8, 64)),  # Buf 2: [2, 3], size = 2048
            ((8, 64), (8, 32)),   # Buf 3: [3, 4], size = 1024
            ((8, 32), (8, 4)),    # Buf 4: [4, 5], size = 128
        ]

        plan = MemoryPlanner.plan_workspace(shape_flow, dtype=float32, alignment=64)

        self.assertIsInstance(plan, MemoryPlan)
        self.assertEqual(plan.num_regions, 2)
        self.assertEqual(plan.num_reused_buffers, 5)

        # Buffers 0, 2, 4 share region 0
        # Buffers 1, 3 share region 1
        self.assertEqual(plan.buffers[0].region_id, 0)
        self.assertEqual(plan.buffers[1].region_id, 1)
        self.assertEqual(plan.buffers[2].region_id, 0)
        self.assertEqual(plan.buffers[3].region_id, 1)
        self.assertEqual(plan.buffers[4].region_id, 0)

        # Offsets are 64-byte aligned
        for region in plan.regions.values():
            self.assertEqual(region.offset_bytes % 64, 0)

        summary = plan.summary()
        self.assertEqual(summary["peak_workspace_bytes"], plan.total_workspace_bytes)
        self.assertEqual(summary["num_regions"], 2)


if __name__ == "__main__":
    unittest.main()
