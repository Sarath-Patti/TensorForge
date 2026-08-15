"""High-level model and checkpoint serialization APIs for TensorForge."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from tensorforge.nn.module import Module
from tensorforge.optim.optimizer import Optimizer
from tensorforge.quantization.quantized_tensor import QuantizedTensor
from tensorforge.serialization.format import (
    extract_module_architecture,
    read_tfckpt_container,
    read_tfmodel_container,
    write_tfckpt_container,
    write_tfmodel_container,
)
from tensorforge.tensor.tensor import Tensor
from tensorforge.utils.validation import SerializationError


def save_model(
    model: Union[Module, Dict[str, Any]],
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model parameters and metadata to a .tfmodel container.

    Args:
        model: A TensorForge :class:`~tensorforge.nn.Module` instance or a state dictionary.
        filepath: Destination file path (e.g., 'model.tfmodel').
        metadata: Optional dictionary of additional model metadata or configuration.
    """
    architecture = None
    if isinstance(model, Module):
        state_dict = model.state_dict()
        architecture = extract_module_architecture(model)
    elif isinstance(model, dict):
        state_dict = model
    else:
        raise SerializationError(f"Expected Module or state_dict dictionary, got {type(model).__name__}")

    write_tfmodel_container(filepath, state_dict, metadata=metadata, architecture=architecture)


def load_model(
    model: Module,
    filepath: str,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model parameters from a .tfmodel container into an existing Module instance.

    Args:
        model: Target :class:`~tensorforge.nn.Module` instance.
        filepath: Path to the .tfmodel file.
        strict: Whether to strictly enforce parameter key matching.

    Returns:
        Metadata dictionary stored inside the model file.
    """
    if not isinstance(model, Module):
        raise SerializationError(f"load_model requires a Module instance, got {type(model).__name__}")

    state_dict, metadata = read_tfmodel_container(filepath)
    model.load_state_dict(state_dict, strict=strict)
    return metadata


def load_state_dict_from_file(
    filepath: str,
) -> Tuple[Dict[str, Union[Tensor, QuantizedTensor]], Dict[str, Any]]:
    """Load and return the raw state dictionary and metadata from a .tfmodel file.

    Args:
        filepath: Path to the .tfmodel file.

    Returns:
        Tuple of (state_dict, metadata).
    """
    return read_tfmodel_container(filepath)


def save_checkpoint(
    checkpoint: Dict[str, Any],
    filepath: str,
) -> None:
    """Save a training checkpoint dictionary into a .tfckpt container.

    Accepts model and optimizer instances directly (extracting their state_dicts automatically)
    or pre-extracted state dictionaries.

    Args:
        checkpoint: Checkpoint dictionary containing model/optimizer states and metadata.
            Standard keys include:
                - 'model' or 'model_state_dict' (Module or state_dict)
                - 'optimizer' or 'optimizer_state_dict' (Optimizer or state_dict)
                - 'epoch' (int)
                - 'step' (int)
                - 'loss' (float)
                - 'metrics' (dict)
                - 'history' (dict)
                - 'user_metadata' (dict)
        filepath: Target destination path (e.g., 'checkpoint.tfckpt').
    """
    if not isinstance(checkpoint, dict):
        raise SerializationError(f"save_checkpoint requires a dict, got {type(checkpoint).__name__}")

    prepared_ckpt: Dict[str, Any] = dict(checkpoint)

    # Automatically extract model state_dict if Module object passed
    if "model" in prepared_ckpt and isinstance(prepared_ckpt["model"], Module):
        prepared_ckpt["model_state_dict"] = prepared_ckpt.pop("model").state_dict()

    # Automatically extract optimizer state_dict if Optimizer object passed
    if "optimizer" in prepared_ckpt and isinstance(prepared_ckpt["optimizer"], Optimizer):
        prepared_ckpt["optimizer_state_dict"] = prepared_ckpt.pop("optimizer").state_dict()

    write_tfckpt_container(filepath, prepared_ckpt)


def load_checkpoint(
    filepath: str,
) -> Dict[str, Any]:
    """Load and return a training checkpoint dictionary from a .tfckpt container.

    Args:
        filepath: Source checkpoint path.

    Returns:
        Reconstructed checkpoint dictionary containing 'model_state_dict',
        'optimizer_state_dict', 'epoch', 'step', 'loss', etc.
    """
    return read_tfckpt_container(filepath)


def compute_model_size(
    model_or_state_dict: Union[Module, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute parameter count and memory size statistics for a model or state dictionary.

    Args:
        model_or_state_dict: A Module instance or state dictionary.

    Returns:
        Dictionary with keys:
            - 'num_parameters': Total number of parameter elements.
            - 'total_bytes': Total physical storage memory in bytes.
            - 'size_kb': Total memory in Kilobytes.
            - 'size_mb': Total memory in Megabytes.
    """
    if isinstance(model_or_state_dict, Module):
        state_dict = model_or_state_dict.state_dict()
    elif isinstance(model_or_state_dict, dict):
        state_dict = model_or_state_dict
    else:
        raise SerializationError(f"Expected Module or dict, got {type(model_or_state_dict).__name__}")

    total_elements = 0
    total_bytes = 0

    for val in state_dict.values():
        if isinstance(val, (Tensor, QuantizedTensor)):
            total_elements += val.numel
            total_bytes += val.nbytes
        elif isinstance(val, np.ndarray):
            total_elements += val.size
            total_bytes += val.nbytes

    return {
        "num_parameters": total_elements,
        "total_bytes": total_bytes,
        "size_kb": total_bytes / 1024.0,
        "size_mb": total_bytes / (1024.0 * 1024.0),
    }
