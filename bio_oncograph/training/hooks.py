"""Training hooks module for injecting custom logic into the training loop."""

from typing import Any, Dict


class TrainingHook:
    """Abstract base class for training hooks."""

    def pre_epoch(self, epoch: int, state: Dict[str, Any]) -> None:
        """Hook called before the start of an epoch.

        Args:
            epoch: The current epoch number.
            state: A dictionary containing training state.
        """
        pass

    def post_epoch(self, epoch: int, state: Dict[str, Any]) -> None:
        """Hook called after the end of an epoch.

        Args:
            epoch: The current epoch number.
            state: A dictionary containing training state.
        """
        pass

    def pre_step(self, step: int, state: Dict[str, Any]) -> None:
        """Hook called before a training step.

        Args:
            step: The current step number.
            state: A dictionary containing training state.
        """
        pass

    def post_step(self, step: int, state: Dict[str, Any]) -> None:
        """Hook called after a training step.

        Args:
            step: The current step number.
            state: A dictionary containing training state.
        """
        pass
