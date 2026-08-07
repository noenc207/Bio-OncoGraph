"""
Checkpoint framework for Bio-OncoGraph.

Provides:
  - ``Checkpoint`` — abstract backend interface with save / load /
    latest / exists / list_checkpoints.
  - ``CheckpointManager`` — multi-backend orchestrator with default
    backend support.

All types are pure Python — **no** ``torch`` or ML-framework
imports.  Concrete backends (filesystem, cloud, database) are
implemented in future milestones within the Infrastructure layer.

Design references:
  - Software Architecture § 15 (Checkpoint System)
  - API Bible § 13 (Checkpoint API)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger("BioOncoGraph")


# ── Checkpoint Backend Interface ────────────────────────────────────


class Checkpoint(ABC):
    """
    Abstract checkpoint storage backend.

    Concrete implementations (local filesystem, cloud object storage,
    database, …) must implement all abstract methods.
    """

    @abstractmethod
    def save(self, state: Dict[str, Any], identifier: str) -> None:
        """
        Persist a state dictionary.

        Args:
            state: Serialisable state dictionary.
            identifier: Unique checkpoint identifier (e.g.
                ``"epoch_05"`` or a path string).
        """
        ...

    @abstractmethod
    def load(self, identifier: str) -> Dict[str, Any]:
        """
        Load a state dictionary.

        Args:
            identifier: Checkpoint identifier.

        Returns:
            The loaded state dictionary.

        Raises:
            KeyError: If the checkpoint does not exist.
        """
        ...

    @abstractmethod
    def latest(self) -> Optional[str]:
        """
        Return the identifier of the most recent checkpoint.

        Returns:
            Identifier string, or ``None`` if no checkpoints exist.
        """
        ...

    @abstractmethod
    def exists(self, identifier: str) -> bool:
        """
        Check whether a checkpoint exists.

        Args:
            identifier: Checkpoint identifier.

        Returns:
            ``True`` if the checkpoint exists.
        """
        ...

    @abstractmethod
    def list_checkpoints(self) -> List[str]:
        """
        List all checkpoint identifiers.

        Returns:
            List of identifier strings in creation order.
        """
        ...


# ── Checkpoint Manager ─────────────────────────────────────────────


class CheckpointManager:
    """
    Multi-backend checkpoint orchestrator.

    Allows registration of multiple ``Checkpoint`` backends
    (e.g. local, cloud) and routes operations to the appropriate
    backend.  One backend may be designated as the default.

    Example::

        mgr = CheckpointManager()
        mgr.register_backend("local", LocalCheckpoint(), default=True)
        mgr.save_checkpoint(state, "epoch_05")
        state = mgr.load_checkpoint("epoch_05")
    """

    def __init__(self) -> None:
        """Initialise with no backends."""
        self._backends: Dict[str, Checkpoint] = {}
        self._default_backend: Optional[str] = None

    @property
    def default_backend(self) -> Optional[str]:
        """Return the name of the default backend, if set."""
        return self._default_backend

    def register_backend(
        self,
        name: str,
        backend: Checkpoint,
        default: bool = False,
    ) -> None:
        """
        Register a checkpoint backend.

        Args:
            name: Unique backend name.
            backend: A ``Checkpoint`` implementation.
            default: If ``True``, this backend becomes the default.

        Raises:
            ValueError: If *name* is already registered.
            TypeError: If *backend* is not a ``Checkpoint``.
        """
        if not isinstance(backend, Checkpoint):
            raise TypeError(
                f"Expected Checkpoint, got "
                f"{type(backend).__name__}."
            )
        if name in self._backends:
            raise ValueError(
                f"Backend '{name}' is already registered."
            )
        self._backends[name] = backend
        if default or self._default_backend is None:
            self._default_backend = name
        logger.info(
            "Registered checkpoint backend: %s (default=%s)",
            name,
            name == self._default_backend,
        )

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        identifier: str,
        backend: Optional[str] = None,
    ) -> None:
        """
        Save a checkpoint.

        Args:
            state: State dictionary to persist.
            identifier: Checkpoint identifier.
            backend: Backend name. Uses default if ``None``.

        Raises:
            ValueError: If no backend is available.
            KeyError: If the named backend is not registered.
        """
        b = self._resolve_backend(backend)
        b.save(state, identifier)
        logger.info(
            "Checkpoint saved: %s (backend=%s)",
            identifier,
            backend or self._default_backend,
        )

    def load_checkpoint(
        self,
        identifier: str,
        backend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            identifier: Checkpoint identifier.
            backend: Backend name. Uses default if ``None``.

        Returns:
            The loaded state dictionary.

        Raises:
            ValueError: If no backend is available.
            KeyError: If the checkpoint or backend is not found.
        """
        b = self._resolve_backend(backend)
        if not b.exists(identifier):
            raise KeyError(
                f"Checkpoint '{identifier}' not found."
            )
        state = b.load(identifier)
        logger.info(
            "Checkpoint loaded: %s (backend=%s)",
            identifier,
            backend or self._default_backend,
        )
        return state

    def latest_checkpoint(
        self, backend: Optional[str] = None
    ) -> Optional[str]:
        """
        Return the latest checkpoint identifier.

        Args:
            backend: Backend name. Uses default if ``None``.

        Returns:
            Identifier string, or ``None``.
        """
        b = self._resolve_backend(backend)
        return b.latest()

    def list_backends(self) -> List[str]:
        """
        List registered backend names.

        Returns:
            List of name strings.
        """
        return list(self._backends.keys())

    def __repr__(self) -> str:
        return (
            f"CheckpointManager(backends="
            f"{self.list_backends()}, "
            f"default={self._default_backend!r})"
        )

    # ── Private helpers ─────────────────────────────────────────

    def _resolve_backend(
        self, name: Optional[str]
    ) -> Checkpoint:
        """Resolve a backend by name or use the default."""
        target = name or self._default_backend
        if target is None:
            raise ValueError(
                "No checkpoint backend registered."
            )
        if target not in self._backends:
            raise KeyError(
                f"Backend '{target}' not registered. "
                f"Available: {list(self._backends.keys())}"
            )
        return self._backends[target]

from dataclasses import dataclass

@dataclass
class ReproducibilityManifest:
    dataset_version: str = ""
    git_commit: str = ""
    seed: int = 42
    torch_version: str = ""
    cuda_version: str = ""

class ExtendedCheckpointManager(CheckpointManager):
    def save_full_checkpoint(
        self,
        state: Dict[str, Any],
        identifier: str,
        scaler_state_dict: Any = None,
        torch_rng_state: Any = None,
        cuda_rng_state: Any = None,
        numpy_state: Any = None,
        dataset_version: str = "",
        git_commit: str = "",
        backend: Optional[str] = None,
    ) -> None:
        state["scaler_state_dict"] = scaler_state_dict
        state["torch_rng_state"] = torch_rng_state
        state["cuda_rng_state"] = cuda_rng_state
        state["numpy_state"] = numpy_state
        state["dataset_version"] = dataset_version
        state["git_commit"] = git_commit
        self.save_checkpoint(state, identifier, backend)
