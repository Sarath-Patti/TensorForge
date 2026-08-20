"""Declarative Model Deployment Manifest & Server Bootstrapping for TensorForge v2.0."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Dict, List, Optional, Union

from tensorforge.inference.profile import RuntimeProfileType, get_runtime_profile
from tensorforge.inference.server import InferenceServer, ServerConfig
from tensorforge.utils.validation import TensorForgeInputError


@dataclass
class ModelDeploymentSpec:
    """Declarative deployment specification for a single model version."""

    name: str
    path: str
    version: str = "1"
    profile_type: str = "BALANCED"
    active: bool = True
    backend: str = "numpy"
    num_threads: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert specification to dictionary."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelDeploymentSpec:
        """Construct specification from dictionary."""
        return cls(
            name=data["name"],
            path=data["path"],
            version=str(data.get("version", "1")),
            profile_type=data.get("profile_type", "BALANCED"),
            active=data.get("active", True),
            backend=data.get("backend", "numpy"),
            num_threads=data.get("num_threads"),
            metadata=data.get("metadata", {}),
        )


# Public alias for v2.0 specification consistency
DeploymentSpec = ModelDeploymentSpec


@dataclass
class DeploymentManifest:
    """Declarative multi-model deployment manifest for bootstrapping InferenceServer."""

    name: str = "tensorforge_deployment"
    server_config: ServerConfig = field(default_factory=ServerConfig)
    models: List[ModelDeploymentSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert deployment manifest to dictionary."""
        return {
            "name": self.name,
            "server_config": self.server_config.to_dict(),
            "models": [m.to_dict() for m in self.models],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, filepath: str, indent: int = 2) -> None:
        """Save manifest to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=indent))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeploymentManifest:
        """Construct DeploymentManifest from dictionary."""
        srv_cfg_data = data.get("server_config", {})
        srv_cfg = ServerConfig(
            max_loaded_models=srv_cfg_data.get("max_loaded_models"),
            max_total_pending_requests=srv_cfg_data.get("max_total_pending_requests"),
            auto_start=srv_cfg_data.get("auto_start", True),
        )
        models_list = [ModelDeploymentSpec.from_dict(m) for m in data.get("models", [])]
        return cls(
            name=data.get("name", "tensorforge_deployment"),
            server_config=srv_cfg,
            models=models_list,
        )

    @classmethod
    def load_json(cls, filepath: str) -> DeploymentManifest:
        """Load DeploymentManifest from JSON file."""
        if not os.path.exists(filepath):
            raise TensorForgeInputError(f"Deployment manifest file not found: '{filepath}'")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def deploy(self, server: Optional[InferenceServer] = None) -> InferenceServer:
        """Bootstrap and deploy models defined in manifest to an InferenceServer instance."""
        target_server = server if server is not None else InferenceServer(config=self.server_config)

        for spec in self.models:
            profile = get_runtime_profile(spec.profile_type)
            entry = target_server.load_model(
                name=spec.name,
                path=spec.path,
                version=spec.version,
                active=spec.active,
                scheduler_config=profile.scheduler_config,
                runtime_limits=profile.runtime_limits,
                num_threads=spec.num_threads,
                backend=spec.backend,
                metadata=spec.metadata,
                overwrite=True,
            )
            # Apply profile circuit breaker & retry policies
            entry.circuit_breaker._config = profile.circuit_breaker_config
            entry.retry_config = profile.retry_config

        return target_server
