"""Structured ZIP-based container format for TensorForge models and checkpoints."""

from __future__ import annotations

import io
import json
import time
from typing import Any, Dict, Optional, Tuple, Union
import zipfile
import numpy as np

from tensorforge.quantization.quantized_tensor import QuantizedTensor
from tensorforge.tensor.dtype import to_dtype
from tensorforge.tensor.tensor import Tensor
from tensorforge.utils.validation import SerializationError

FORMAT_VERSION = "1.0"
LIBRARY_VERSION = "1.0.0"


def extract_module_architecture(module: Any) -> Optional[Dict[str, Any]]:
    """Extract serializable architecture specification from a Module instance.

    Supports standard TensorForge layers:
        - Sequential
        - Linear
        - ReLU
        - Sigmoid
        - Tanh
        - Softmax

    Args:
        module: A TensorForge Module instance.

    Returns:
        Dictionary representation of the module architecture, or None if not a Module.
    """
    from tensorforge.nn.activations import ReLU, Sigmoid, Softmax, Tanh
    from tensorforge.nn.linear import Linear
    from tensorforge.nn.module import Module
    from tensorforge.nn.sequential import Sequential

    if not isinstance(module, Module):
        return None

    if isinstance(module, Linear):
        return {
            "type": "Linear",
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias is not None,
            "dtype": str(module.dtype),
        }
    elif isinstance(module, ReLU):
        return {"type": "ReLU"}
    elif isinstance(module, Sigmoid):
        return {"type": "Sigmoid"}
    elif isinstance(module, Tanh):
        return {"type": "Tanh"}
    elif isinstance(module, Softmax):
        return {"type": "Softmax", "dim": module.dim}
    elif isinstance(module, Sequential):
        layers = []
        for name, submod in module._modules.items():
            if submod is not None:
                sub_config = extract_module_architecture(submod)
                if sub_config is not None:
                    layers.append({"name": name, "module": sub_config})
        return {
            "type": "Sequential",
            "layers": layers,
        }
    return {"type": module.__class__.__name__}


def serialize_state_dict_to_zip(
    state_dict: Dict[str, Any],
    zip_file: zipfile.ZipFile,
    prefix: str = "tensors",
) -> Dict[str, Any]:
    """Serialize a dictionary of Tensors, QuantizedTensors, or arrays into a zip container.

    Args:
        state_dict: Dictionary of named tensors.
        zip_file: Open ZipFile instance with write access.
        prefix: Directory prefix inside the zip container.

    Returns:
        Metadata dictionary describing serialized tensors.
    """
    tensors_meta: Dict[str, Any] = {}

    for name, val in state_dict.items():
        # Clean file path safe key
        safe_name = name.replace("/", "_")
        target_path = f"{prefix}/{safe_name}.npy"

        if isinstance(val, QuantizedTensor):
            arr = val.numpy()
            tensors_meta[name] = {
                "path": target_path,
                "is_quantized": True,
                "shape": list(val.shape),
                "dtype": str(val.dtype),
                "scale": float(val.scale),
                "zero_point": int(val.zero_point),
                "scheme": str(val.scheme),
                "orig_shape": list(val.shape),
                "orig_dtype": str(val.orig_dtype),
                "nbytes": int(val.nbytes),
            }
        elif isinstance(val, Tensor):
            arr = val.numpy()
            tensors_meta[name] = {
                "path": target_path,
                "is_quantized": False,
                "shape": list(val.shape),
                "dtype": str(val.dtype),
                "nbytes": int(val.nbytes),
            }
        elif isinstance(val, np.ndarray):
            arr = val
            tensors_meta[name] = {
                "path": target_path,
                "is_quantized": False,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "nbytes": int(arr.nbytes),
            }
        else:
            arr = np.asarray(val)
            tensors_meta[name] = {
                "path": target_path,
                "is_quantized": False,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "nbytes": int(arr.nbytes),
            }

        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        zip_file.writestr(target_path, buf.getvalue())

    return tensors_meta


def deserialize_state_dict_from_zip(
    zip_file: zipfile.ZipFile,
    tensors_meta: Dict[str, Any],
) -> Dict[str, Union[Tensor, QuantizedTensor]]:
    """Deserialize named tensors from a zip container using its tensor metadata descriptor.

    Args:
        zip_file: Open ZipFile instance with read access.
        tensors_meta: Metadata dictionary describing serialized tensors.

    Returns:
        Dictionary mapping tensor names to reconstructed Tensor or QuantizedTensor objects.
    """
    state_dict: Dict[str, Union[Tensor, QuantizedTensor]] = {}

    for name, meta in tensors_meta.items():
        path = meta.get("path")
        if not path or path not in zip_file.namelist():
            raise SerializationError(f"Corrupted model file: tensor '{name}' data missing at '{path}'")

        try:
            with zip_file.open(path) as f:
                buf = io.BytesIO(f.read())
                arr = np.load(buf, allow_pickle=False)
        except Exception as e:
            raise SerializationError(f"Failed to read tensor '{name}' data from '{path}': {e}") from e

        is_quantized = meta.get("is_quantized", False)
        if is_quantized:
            scale = meta.get("scale", 1.0)
            zero_point = meta.get("zero_point", 0)
            scheme = meta.get("scheme", "symmetric")
            orig_dtype = meta.get("orig_dtype", "float32")
            orig_shape = tuple(meta.get("orig_shape", arr.shape))

            state_dict[name] = QuantizedTensor(
                qdata=arr,
                scale=scale,
                zero_point=zero_point,
                scheme=scheme,
                orig_dtype=orig_dtype,
                orig_shape=orig_shape,
            )
        else:
            dtype_str = meta.get("dtype")
            target_dtype = to_dtype(dtype_str) if dtype_str else None
            state_dict[name] = Tensor(arr, dtype=target_dtype, copy=False)

    return state_dict


def write_tfmodel_container(
    filepath: str,
    model_state_dict: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    architecture: Optional[Dict[str, Any]] = None,
) -> None:
    """Write model weights, architecture, and metadata into a .tfmodel zip archive.

    Args:
        filepath: Destination file path.
        model_state_dict: Dictionary of parameter tensors.
        metadata: Optional user/model metadata dictionary.
        architecture: Optional model architecture configuration dictionary.
    """
    try:
        with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            tensors_meta = serialize_state_dict_to_zip(model_state_dict, zf, prefix="tensors")

            # Check if architecture is in metadata or passed explicitly
            arch_meta = architecture
            if arch_meta is None and metadata and "architecture" in metadata:
                arch_meta = metadata["architecture"]

            header_meta = {
                "format_version": FORMAT_VERSION,
                "library": "TensorForge",
                "tensorforge_version": LIBRARY_VERSION,
                "timestamp": time.time(),
                "num_parameters": len(model_state_dict),
                "total_nbytes": sum(m.get("nbytes", 0) for m in tensors_meta.values()),
                "architecture": arch_meta,
                "user_metadata": metadata or {},
                "tensors": tensors_meta,
            }

            zf.writestr("metadata.json", json.dumps(header_meta, indent=2))
    except Exception as e:
        if not isinstance(e, SerializationError):
            raise SerializationError(f"Failed to write model to '{filepath}': {e}") from e
        raise


def read_tfmodel_container(
    filepath: str,
) -> Tuple[Dict[str, Union[Tensor, QuantizedTensor]], Dict[str, Any]]:
    """Read model weights and metadata from a .tfmodel zip archive.

    Args:
        filepath: Source file path.

    Returns:
        Tuple of (state_dict, metadata_dictionary).
    """
    if not zipfile.is_zipfile(filepath):
        raise SerializationError(f"Invalid model file: '{filepath}' is not a valid TensorForge .tfmodel container")

    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            if "metadata.json" not in zf.namelist():
                raise SerializationError(f"Corrupted model file: missing 'metadata.json' in '{filepath}'")

            try:
                with zf.open("metadata.json") as f:
                    meta = json.loads(f.read().decode("utf-8"))
            except Exception as e:
                raise SerializationError(f"Failed to parse 'metadata.json' in '{filepath}': {e}") from e

            format_ver = meta.get("format_version")
            if not format_ver:
                raise SerializationError("Unsupported model file: missing 'format_version' metadata")

            # Check format major version compatibility
            major_ver = format_ver.split(".")[0]
            if major_ver != FORMAT_VERSION.split(".")[0]:
                raise SerializationError(
                    f"Incompatible format version: file is version {format_ver}, but runtime expects {FORMAT_VERSION}"
                )

            tensors_meta = meta.get("tensors", {})
            state_dict = deserialize_state_dict_from_zip(zf, tensors_meta)
            return state_dict, meta
    except Exception as e:
        if not isinstance(e, SerializationError):
            raise SerializationError(f"Failed to load model from '{filepath}': {e}") from e
        raise


def write_tfckpt_container(
    filepath: str,
    checkpoint: Dict[str, Any],
) -> None:
    """Write training checkpoint (model, optimizer, epoch, step, metrics) into a .tfckpt container.

    Args:
        filepath: Target checkpoint path.
        checkpoint: Checkpoint dictionary.
    """
    try:
        with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # 1. Model State Dict
            model_state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
            model_tensors_meta: Dict[str, Any] = {}
            if model_state:
                model_tensors_meta = serialize_state_dict_to_zip(model_state, zf, prefix="model_tensors")

            # 2. Optimizer State Dict
            optim_state = checkpoint.get("optimizer_state_dict")
            optim_meta: Dict[str, Any] = {}
            if optim_state and isinstance(optim_state, dict):
                raw_state = optim_state.get("state", {})
                param_groups = optim_state.get("param_groups", [])

                serialized_optim_state: Dict[str, Any] = {}
                for idx_str, p_state in raw_state.items():
                    entry_meta: Dict[str, Any] = {}
                    for k, v in p_state.items():
                        if isinstance(v, (np.ndarray, Tensor)):
                            arr = v.numpy() if isinstance(v, Tensor) else v
                            target_path = f"optim_tensors/param_{idx_str}_{k}.npy"
                            buf = io.BytesIO()
                            np.save(buf, arr, allow_pickle=False)
                            zf.writestr(target_path, buf.getvalue())
                            entry_meta[k] = {"path": target_path, "shape": list(arr.shape), "dtype": str(arr.dtype)}
                        else:
                            entry_meta[k] = v
                    serialized_optim_state[str(idx_str)] = entry_meta

                optim_meta = {
                    "param_groups": param_groups,
                    "state": serialized_optim_state,
                }

            # 3. Checkpoint Manifest & Metadata
            manifest = {
                "format_version": FORMAT_VERSION,
                "library": "TensorForge",
                "tensorforge_version": LIBRARY_VERSION,
                "timestamp": time.time(),
                "epoch": checkpoint.get("epoch"),
                "step": checkpoint.get("step"),
                "loss": float(checkpoint.get("loss")) if checkpoint.get("loss") is not None else None,
                "metrics": checkpoint.get("metrics"),
                "history": checkpoint.get("history"),
                "user_metadata": checkpoint.get("user_metadata", {}),
                "model_tensors": model_tensors_meta,
                "optimizer": optim_meta,
            }

            zf.writestr("checkpoint.json", json.dumps(manifest, indent=2))
    except Exception as e:
        if not isinstance(e, SerializationError):
            raise SerializationError(f"Failed to write checkpoint to '{filepath}': {e}") from e
        raise


def read_tfckpt_container(
    filepath: str,
) -> Dict[str, Any]:
    """Read training checkpoint from a .tfckpt container.

    Args:
        filepath: Source checkpoint path.

    Returns:
        Reconstructed checkpoint dictionary.
    """
    if not zipfile.is_zipfile(filepath):
        raise SerializationError(f"Invalid checkpoint file: '{filepath}' is not a valid TensorForge container")

    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            if "checkpoint.json" not in zf.namelist():
                raise SerializationError(f"Corrupted checkpoint file: missing 'checkpoint.json' in '{filepath}'")

            try:
                with zf.open("checkpoint.json") as f:
                    manifest = json.loads(f.read().decode("utf-8"))
            except Exception as e:
                raise SerializationError(f"Failed to parse 'checkpoint.json' in '{filepath}': {e}") from e

            format_ver = manifest.get("format_version")
            if not format_ver:
                raise SerializationError("Unsupported checkpoint file: missing 'format_version'")

            # 1. Deserialize Model State
            model_tensors_meta = manifest.get("model_tensors", {})
            model_state_dict = deserialize_state_dict_from_zip(zf, model_tensors_meta) if model_tensors_meta else {}

            # 2. Deserialize Optimizer State
            optim_meta = manifest.get("optimizer", {})
            optimizer_state_dict: Dict[str, Any] = {}
            if optim_meta:
                param_groups = optim_meta.get("param_groups", [])
                raw_state = optim_meta.get("state", {})
                restored_state: Dict[int, Dict[str, Any]] = {}

                for idx_str, p_state in raw_state.items():
                    idx = int(idx_str)
                    restored_entry: Dict[str, Any] = {}
                    for k, v in p_state.items():
                        if isinstance(v, dict) and "path" in v:
                            path = v["path"]
                            with zf.open(path) as f:
                                buf = io.BytesIO(f.read())
                                arr = np.load(buf, allow_pickle=False)
                            restored_entry[k] = arr
                        else:
                            restored_entry[k] = v
                    restored_state[idx] = restored_entry

                optimizer_state_dict = {
                    "param_groups": param_groups,
                    "state": restored_state,
                }

            checkpoint: Dict[str, Any] = {
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "epoch": manifest.get("epoch"),
                "step": manifest.get("step"),
                "loss": manifest.get("loss"),
                "metrics": manifest.get("metrics"),
                "history": manifest.get("history"),
                "user_metadata": manifest.get("user_metadata", {}),
                "format_version": format_ver,
                "timestamp": manifest.get("timestamp"),
            }
            return checkpoint
    except Exception as e:
        if not isinstance(e, SerializationError):
            raise SerializationError(f"Failed to load checkpoint from '{filepath}': {e}") from e
        raise
