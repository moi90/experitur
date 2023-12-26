from typing import TYPE_CHECKING

from unavailable_object import UnavailableObject

from experitur.core.configurators import (
    AdditiveConfiguratorChain,
    Const,
    FilterConfig,
    Grid,
    MultiplicativeConfiguratorChain,
    RandomGrid,
    Clear,
)

from .conditions import Conditions
from .pruning import Prune

__all__ = [
    "AdditiveConfiguratorChain",
    "Conditions",
    "Const",
    "FilterConfig",
    "Grid",
    "MultiplicativeConfiguratorChain",
    "Prune",
    "RandomGrid",
    "Clear",
]

try:
    from .random import Random

    __all__.append("Random")
except ImportError:
    if not TYPE_CHECKING:
        Random = UnavailableObject("Random")

try:
    from .skopt import SKOpt

    __all__.append("SKOpt")
except ImportError:
    if not TYPE_CHECKING:
        SKOpt = UnavailableObject("SKOpt")
