"""Tests for INT8 quantized model loading and inference execution."""

import os
import tempfile
import unittest
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.quantization import dequantize, qmatmul, quantize
from tensorforge.serialization import save_model


class TestQuantizedInference(unittest.TestCase):

    def test_quantized_inference_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "quantized.tfmodel")

            # 1. Create FP32 Model and Quantize
            model = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 2),
            )

            q_state_dict = {}
            for name, param in model.named_parameters():
                q_state_dict[name] = quantize(param, scheme="symmetric")

            # Save with metadata architecture
            save_model(model, model_path)  # saves architecture
            # Overwrite with quantized state dict preserving architecture
            from tensorforge.serialization.format import write_tfmodel_container, extract_module_architecture
            write_tfmodel_container(
                model_path,
                q_state_dict,
                metadata={"is_quantized": True, "scheme": "symmetric"},
                architecture=extract_module_architecture(model),
            )

            # 2. Load into InferenceRuntime
            runtime = InferenceRuntime.load(model_path)
            self.assertTrue(runtime.is_quantized)
            self.assertEqual(runtime.input_shape, (8,))
            self.assertEqual(runtime.output_shape, (2,))

            # 3. Execute quantized prediction
            x = tf.randn((4, 8))
            out = runtime.predict(x)

            self.assertIsInstance(out, tf.Tensor)
            self.assertEqual(out.shape, (4, 2))
            self.assertFalse(np.any(np.isnan(out.numpy())))
            self.assertFalse(np.any(np.isinf(out.numpy())))


if __name__ == "__main__":
    unittest.main()
