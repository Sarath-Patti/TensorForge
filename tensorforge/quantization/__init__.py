"""TensorForge Quantization Subsystem for Low-Precision INT8 Inference."""

from tensorforge.quantization.calibration import (
    Calibrator,
    MinMaxCalibrator,
    MovingAverageCalibrator,
    PercentileCalibrator,
    calibrate_tensor,
    compute_quantization_params,
)
from tensorforge.quantization.metrics import (
    compare_tensors,
    max_absolute_error,
    mean_absolute_error,
    mean_squared_error,
    quantization_snr,
    relative_error,
)
from tensorforge.quantization.quantize import (
    dequantize,
    dequantize_tensor,
    qmatmul,
    quantize,
    quantize_tensor,
)
from tensorforge.quantization.quantized_tensor import QuantizedTensor

__all__ = [
    # Quantized Data Structure
    "QuantizedTensor",
    # Core Quantization Operations
    "quantize",
    "dequantize",
    "quantize_tensor",
    "dequantize_tensor",
    "qmatmul",
    # Calibration
    "Calibrator",
    "MinMaxCalibrator",
    "MovingAverageCalibrator",
    "PercentileCalibrator",
    "calibrate_tensor",
    "compute_quantization_params",
    # Metrics
    "max_absolute_error",
    "mean_absolute_error",
    "mean_squared_error",
    "relative_error",
    "quantization_snr",
    "compare_tensors",
]
