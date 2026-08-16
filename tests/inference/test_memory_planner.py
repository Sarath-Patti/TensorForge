"""Tests for inference memory planning and buffer lifetime reuse."""

import unittest
from tensorforge.inference.memory import MemoryPlanner
from tensorforge.tensor.dtype import float32


class TestMemoryPlanner(unittest.TestCase):

    def test_memory_planner_sequential_slots(self):
        # 4-step pipeline with changing shapes
        shape_flow = [
            ((8, 16), (8, 64)),   # Step 0: 8 * 64 * 4 = 2048 bytes (out -> slot 0)
            ((8, 64), (8, 128)),  # Step 1: 8 * 128 * 4 = 4096 bytes (out -> slot 1)
            ((8, 128), (8, 32)),  # Step 2: 8 * 32 * 4 = 1024 bytes (out -> slot 0)
            ((8, 32), (8, 4)),    # Step 3: 8 * 4 * 4 = 128 bytes (out -> slot 1)
        ]

        plan = MemoryPlanner.plan_sequential_workspace(shape_flow, dtype=float32, alignment=64)

        self.assertEqual(plan["num_reused_slots"], 2)
        self.assertEqual(plan["step_input_slots"], [-1, 0, 1, 0])
        self.assertEqual(plan["step_output_slots"], [0, 1, 0, 1])

        # Slot 0 max bytes: max(2048, 1024) = 2048 bytes
        # Slot 1 max bytes: max(4096, 128) = 4096 bytes
        self.assertEqual(plan["slot_sizes"][0], 2048)
        self.assertEqual(plan["slot_sizes"][1], 4096)

        # Offset for slot 0 should be 0
        # Offset for slot 1 should be aligned to 64 bytes
        self.assertEqual(plan["slot_offsets"][0], 0)
        self.assertEqual(plan["slot_offsets"][1], 2048)
        self.assertEqual(plan["total_workspace_bytes"], 2048 + 4096)

    def test_memory_planner_single_step(self):
        shape_flow = [((1, 8), (1, 16))]  # 1 * 16 * 4 = 64 bytes
        plan = MemoryPlanner.plan_sequential_workspace(shape_flow, dtype=float32, alignment=64)

        self.assertEqual(plan["num_reused_slots"], 1)
        self.assertEqual(plan["step_input_slots"], [-1])
        self.assertEqual(plan["step_output_slots"], [0])
        self.assertEqual(plan["slot_sizes"][0], 64)
        self.assertEqual(plan["total_workspace_bytes"], 64)


if __name__ == "__main__":
    unittest.main()
