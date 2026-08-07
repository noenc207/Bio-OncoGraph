"""
Runtime Engine for Bio-OncoGraph.

Provides ``RuntimeEngine`` — a state-machine-governed runtime that
initialises the pipeline, orchestrates the ``Trainer``, manages
lifecycle transitions, and publishes events through the ``EventBus``.

Valid state transitions are enforced; invalid transitions raise
``InvalidStateTransitionError``.

Design references:
  - Software Architecture § 20 (Runtime Management)
  - Software Architecture § 18 (Lifecycle Management)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from enum import Enum, auto
from typing import Any, Dict, Optional, Set
import json
import hashlib
import time
import subprocess
from datetime import datetime

from bio_oncograph.core.events import EventBus
from bio_oncograph.core.exceptions import BioOncoGraphException

from bio_oncograph.application.trainer import (
    RuntimeErrorOccurred,
    Trainer,
    TrainingFinished,
    TrainingStarted,
)

logger = logging.getLogger("BioOncoGraph")


# ── Exception ───────────────────────────────────────────────────────


class InvalidStateTransitionError(BioOncoGraphException):
    """Raised when an invalid runtime state transition is attempted."""

    pass


# ── State Machine ───────────────────────────────────────────────────


class RuntimeState(Enum):
    """States of the RuntimeEngine finite state machine."""

    CREATED = auto()
    INITIALIZED = auto()
    TRAINING = auto()
    VALIDATING = auto()
    TESTING = auto()
    PREDICTING = auto()
    STOPPED = auto()
    FINISHED = auto()
    ERROR = auto()


_VALID_TRANSITIONS: Dict[RuntimeState, Set[RuntimeState]] = {
    RuntimeState.CREATED: {RuntimeState.INITIALIZED},
    RuntimeState.INITIALIZED: {
        RuntimeState.TRAINING,
        RuntimeState.VALIDATING,
        RuntimeState.TESTING,
        RuntimeState.PREDICTING,
        RuntimeState.STOPPED,
    },
    RuntimeState.TRAINING: {
        RuntimeState.VALIDATING,
        RuntimeState.STOPPED,
        RuntimeState.FINISHED,
        RuntimeState.ERROR,
    },
    RuntimeState.VALIDATING: {
        RuntimeState.TRAINING,
        RuntimeState.STOPPED,
        RuntimeState.FINISHED,
        RuntimeState.ERROR,
    },
    RuntimeState.TESTING: {
        RuntimeState.STOPPED,
        RuntimeState.FINISHED,
        RuntimeState.ERROR,
    },
    RuntimeState.PREDICTING: {
        RuntimeState.STOPPED,
        RuntimeState.FINISHED,
        RuntimeState.ERROR,
    },
    RuntimeState.STOPPED: {RuntimeState.INITIALIZED},
    RuntimeState.FINISHED: {RuntimeState.INITIALIZED},
    RuntimeState.ERROR: {
        RuntimeState.INITIALIZED,
        RuntimeState.STOPPED,
    },
}

class ExperimentRegistry:
    """Registry to track experiments across runs."""
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        if not self.registry_path.exists():
            self._save_registry([])

    def _load_registry(self) -> list:
        try:
            with open(self.registry_path, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_registry(self, data: list) -> None:
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=4)

    def _get_git_hash(self) -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception:
            return "unknown_git_hash"

    def register_run(self, experiment_id: str, config_path: Path, checkpoint_path: Optional[Path] = None) -> None:
        data = self._load_registry()
        
        config_hash = "unknown"
        if config_path.exists():
            with open(config_path, "rb") as f:
                config_hash = hashlib.md5(f.read()).hexdigest()
                
        checkpoint_hash = "unknown"
        if checkpoint_path and checkpoint_path.exists():
            with open(checkpoint_path, "rb") as f:
                checkpoint_hash = hashlib.md5(f.read()).hexdigest()

        entry = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "git_hash": self._get_git_hash(),
            "config_hash": config_hash,
            "checkpoint_hash": checkpoint_hash
        }
        data.append(entry)
        self._save_registry(data)

# ── Runtime Engine ──────────────────────────────────────────────────


class RuntimeEngine:
    """
    State-machine-governed runtime for Bio-OncoGraph.

    Lifecycle::

        CREATED → initialize() → INITIALIZED
        INITIALIZED → run(mode) → TRAINING/VALIDATING/… → FINISHED
        any → shutdown() → STOPPED
        STOPPED/FINISHED → reset() → CREATED

    Invalid transitions raise ``InvalidStateTransitionError``.
    """

    def __init__(self, event_bus: EventBus) -> None:
        """
        Initialise the runtime engine.

        Args:
            event_bus: The framework ``EventBus``.
        """
        self._event_bus = event_bus
        self._state = RuntimeState.CREATED
        self._trainer: Optional[Trainer] = None

    # ── Properties ──────────────────────────────────────────────

    @property
    def state(self) -> RuntimeState:
        """Return the current runtime state."""
        return self._state

    @property
    def event_bus(self) -> EventBus:
        """Return the event bus."""
        return self._event_bus

    @property
    def trainer(self) -> Optional[Trainer]:
        """Return the trainer, if initialized."""
        return self._trainer

    # ── Public API ──────────────────────────────────────────────

    def initialize(self, trainer: Trainer, experiment_name: str = "default_experiment", base_dir: str = "runs") -> None:
        """
        Initialise the runtime with a trainer and create experiment folder structure.

        Transitions: `CREATED → INITIALIZED`,
        `STOPPED → INITIALIZED`, `FINISHED → INITIALIZED`.

        Args:
            trainer: A configured `Trainer` instance.
            experiment_name: Name of the experiment.
            base_dir: Base directory for runs.

        Raises:
            InvalidStateTransitionError: If the current state does
                not allow transitioning to `INITIALIZED`.
        """
        self._transition(RuntimeState.INITIALIZED)
        self._trainer = trainer
        
        # Create standard experiment folder structure
        exp_dir = Path(base_dir) / experiment_name
        self._setup_experiment_folders(exp_dir)
        self._setup_logging(exp_dir)
        
        logger.info(f"Runtime initialized for experiment: {experiment_name}")

    def _setup_experiment_folders(self, exp_dir: Path) -> None:
        (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (exp_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
        (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
        
        config_path = exp_dir / "config.yaml"
        if not config_path.exists():
            config_path.write_text("# Auto-generated config\n")
            
        registry = ExperimentRegistry(exp_dir.parent / "registry.json")
        registry.register_run(exp_dir.name, config_path)

    def _setup_logging(self, exp_dir: Path) -> None:
        log_file = exp_dir / "logs" / "runtime.log"
        file_handler = logging.FileHandler(str(log_file))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def run(self, mode: str = "train") -> Dict[str, Any]:
        """
        Run the trainer in the specified mode.

        Supported modes:

        - ``"train"``: calls ``Trainer.fit()``.
        - ``"validate"``: calls ``Trainer.validate()``.
        - ``"test"``: calls ``Trainer.evaluate()``.
        - ``"predict"``: calls ``Trainer.predict()``.

        Args:
            mode: Execution mode string.

        Returns:
            Result dictionary from the trainer.

        Raises:
            InvalidStateTransitionError: If transition is invalid.
            ValueError: If *mode* is not recognised or trainer is
                not initialized.
        """
        if self._trainer is None:
            raise ValueError(
                "Trainer not initialized. Call initialize() first."
            )

        mode_lower = mode.lower()
        try:
            if mode_lower == "train":
                self._transition(RuntimeState.TRAINING)
                result = self._trainer.fit()
                self._transition(RuntimeState.FINISHED)
                return result

            elif mode_lower == "validate":
                self._transition(RuntimeState.VALIDATING)
                result = self._trainer.validate()
                self._transition(RuntimeState.FINISHED)
                return result

            elif mode_lower == "test":
                self._transition(RuntimeState.TESTING)
                result = self._trainer.evaluate()
                self._transition(RuntimeState.FINISHED)
                return result

            elif mode_lower == "predict":
                self._transition(RuntimeState.PREDICTING)
                predictions = self._trainer.predict()
                self._transition(RuntimeState.FINISHED)
                return {"predictions": predictions}

            else:
                raise ValueError(
                    f"Unknown mode '{mode}'. "
                    f"Supported: train, validate, test, predict."
                )

        except (InvalidStateTransitionError, ValueError):
            raise
        except Exception as exc:
            self._state = RuntimeState.ERROR
            self._event_bus.publish(
                RuntimeErrorOccurred(
                    source="RuntimeEngine",
                    payload={"error": str(exc), "mode": mode},
                )
            )
            logger.exception("Runtime error in mode '%s'", mode)
            raise

    def shutdown(self) -> None:
        """
        Shut down the runtime.

        Transitions to ``STOPPED`` from any active state.

        Raises:
            InvalidStateTransitionError: If already ``CREATED``.
        """
        self._transition(RuntimeState.STOPPED)
        self._trainer = None
        logger.info("Runtime shutdown")

    def reset(self) -> None:
        """
        Reset the runtime to ``CREATED`` state.

        Must be in ``STOPPED`` or ``FINISHED`` state first.

        Raises:
            InvalidStateTransitionError: If current state does not
                allow transitioning to ``CREATED``.
        """
        if self._state not in (
            RuntimeState.STOPPED,
            RuntimeState.FINISHED,
        ):
            raise InvalidStateTransitionError(
                f"Cannot reset from {self._state.name}. "
                f"Must be STOPPED or FINISHED."
            )
        self._state = RuntimeState.CREATED
        self._trainer = None
        logger.info("Runtime reset to CREATED")

    # ── Private helpers ─────────────────────────────────────────

    def _transition(self, new_state: RuntimeState) -> None:
        """
        Attempt a state transition.

        Args:
            new_state: The target state.

        Raises:
            InvalidStateTransitionError: If the transition is
                not in the valid-transitions table.
        """
        valid = _VALID_TRANSITIONS.get(self._state, set())
        if new_state not in valid:
            raise InvalidStateTransitionError(
                f"Cannot transition from {self._state.name} "
                f"to {new_state.name}. "
                f"Valid targets: "
                f"{sorted(s.name for s in valid)}"
            )
        old = self._state
        self._state = new_state
        logger.info(
            "Runtime state: %s → %s", old.name, new_state.name
        )
