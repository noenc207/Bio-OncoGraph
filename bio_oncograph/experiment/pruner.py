"""Pruners for early stopping of poorly performing trials."""

from typing import Any
import optuna
from optuna.pruners import (
    MedianPruner,
    SuccessiveHalvingPruner,
    HyperbandPruner,
    NopPruner,
    BasePruner,
)


def get_pruner(pruner_name: str, **kwargs: Any) -> BasePruner:
    """Creates an Optuna pruner based on the provided name.

    Args:
        pruner_name: The name of the pruner ('median', 'successive_halving',
            'hyperband', 'nop').
        **kwargs: Additional keyword arguments to pass to the pruner constructor.

    Returns:
        An instantiated Optuna pruner.

    Raises:
        ValueError: If the pruner name is unknown.
    """
    pruner_name = pruner_name.lower()

    if pruner_name == "median":
        return MedianPruner(**kwargs)
    elif pruner_name == "successive_halving":
        return SuccessiveHalvingPruner(**kwargs)
    elif pruner_name == "hyperband":
        return HyperbandPruner(**kwargs)
    elif pruner_name in ("nop", "none"):
        return NopPruner()
    else:
        raise ValueError(
            f"Unknown pruner name: '{pruner_name}'. Supported pruners are: "
            "'median', 'successive_halving', 'hyperband', 'nop'."
        )
