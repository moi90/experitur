from typing import TYPE_CHECKING
from unavailable_object import UnavailableObject
from experitur.core.configurators import (
    Const,
    Grid,
    MultiplicativeConfiguratorChain,
    AdditiveConfiguratorChain,
)

from .conditions import Conditions
from .pruning import Prune

try:
    from .random import Random
except ImportError:
    if not TYPE_CHECKING:
        Random = UnavailableObject("Random")

try:
    from .skopt import SKOpt
except ImportError:
    if not TYPE_CHECKING:
        SKOpt = UnavailableObject("SKOpt")
