"""Search space definitions mapping to Optuna distributions."""

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import optuna


class BaseHyperparameter(abc.ABC):
    """Base class for hyperparameter definitions."""

    def __init__(self, name: str):
        """Initializes the hyperparameter.

        Args:
            name: The name of the hyperparameter.
        """
        self.name = name

    @abc.abstractmethod
    def suggest(self, trial: optuna.Trial) -> Any:
        """Suggests a value for the hyperparameter using an Optuna trial.

        Args:
            trial: The Optuna trial object.

        Returns:
            The suggested value.
        """
        pass


@dataclass
class CategoricalHyperparameter(BaseHyperparameter):
    """Categorical hyperparameter definition."""

    name: str
    choices: List[Any]

    def suggest(self, trial: optuna.Trial) -> Any:
        """Suggests a categorical value."""
        return trial.suggest_categorical(self.name, self.choices)


@dataclass
class FloatHyperparameter(BaseHyperparameter):
    """Float hyperparameter definition."""

    name: str
    low: float
    high: float
    log: bool = False
    step: Optional[float] = None

    def suggest(self, trial: optuna.Trial) -> float:
        """Suggests a float value."""
        return trial.suggest_float(self.name, self.low, self.high, log=self.log, step=self.step)


@dataclass
class IntHyperparameter(BaseHyperparameter):
    """Integer hyperparameter definition."""

    name: str
    low: int
    high: int
    log: bool = False
    step: int = 1

    def suggest(self, trial: optuna.Trial) -> int:
        """Suggests an integer value."""
        return trial.suggest_int(self.name, self.low, self.high, log=self.log, step=self.step)


class SearchSpace:
    """A collection of hyperparameters defining the search space."""

    def __init__(self):
        """Initializes an empty search space."""
        self._hyperparameters: Dict[str, BaseHyperparameter] = {}

    def add(self, hyperparameter: BaseHyperparameter) -> None:
        """Adds a hyperparameter to the search space.

        Args:
            hyperparameter: The hyperparameter to add.
        """
        self._hyperparameters[hyperparameter.name] = hyperparameter

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggests values for all hyperparameters in the search space.

        Args:
            trial: The Optuna trial object.

        Returns:
            A dictionary mapping hyperparameter names to their suggested values.
        """
        return {
            name: hp.suggest(trial) for name, hp in self._hyperparameters.items()
        }

    @classmethod
    def get_default_space(cls) -> "SearchSpace":
        """Returns a default search space for Bio-OncoGraph models.

        Returns:
            A SearchSpace object populated with default hyperparameters.
        """
        space = cls()
        space.add(FloatHyperparameter("learning_rate", low=1e-5, high=1e-2, log=True))
        space.add(FloatHyperparameter("weight_decay", low=1e-6, high=1e-3, log=True))
        space.add(FloatHyperparameter("dropout", low=0.0, high=0.5, step=0.1))
        space.add(IntHyperparameter("hidden_dim", low=32, high=512, log=True))
        space.add(CategoricalHyperparameter("optimizer", choices=["adam", "adamw", "sgd"]))
        space.add(FloatHyperparameter("loss_weight_clinical", low=0.1, high=1.0))
        space.add(FloatHyperparameter("loss_weight_genomic", low=0.1, high=1.0))
        return space
