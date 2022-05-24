from collections import OrderedDict
from typing import Iterable, List, Mapping, Optional, Union, TYPE_CHECKING

import numpy as np

from experitur.util import ensure_list

Result = Optional[Mapping[str, float]]
Objective = Union[str, Iterable[str], None]

if TYPE_CHECKING:
    from experitur.core.trial import Trial


class Optimization:
    def __init__(
        self, minimize: Objective = None, maximize: Objective = None,
    ):
        minimize, maximize = ensure_list(minimize), ensure_list(maximize)  # type: ignore

        common = set(minimize) & set(maximize)

        if common:
            common = ", ".join(sorted(common))
            raise ValueError(f"minimize and maximize share common metrics: {common}")

        _minimize = OrderedDict()
        for m in maximize:
            _minimize[m] = False
        for m in minimize:
            _minimize[m] = True

        if not _minimize:
            raise ValueError("At least one of minimize and maximize has to be supplied")

        self._minimize = _minimize

    def invert_signs(self, values: Result):
        """Invert signs so that only minimization has to be considered."""
        if values is None:
            return None

        try:
            return {
                k: values[k] if m else -values[k] for k, m in self._minimize.items()
            }
        except KeyError as exc:
            print(f"{exc} for {values}")
            raise

    def to_minimization(
        self, results: List[Result], quantile=1.0
    ) -> List[Optional[float]]:
        """
        Turn a list of of (possibly multi-value) results into a list of numbers that can be minimized to optimize the provided objective(s).
        """

        results = [self.invert_signs(r) for r in results]

        # If only a single objective is requested, return this one
        if len(self._minimize) == 1:
            objective = next(iter(self._minimize.keys()))
            return [r[objective] if r is not None else None for r in results]

        if len(results) < 2:
            return [0.0] * len(results)

        # Y.shape(n_trials, n_objectives)
        Y = np.array(
            [
                [
                    r[objective] if r is not None else np.nan
                    for objective in self._minimize.keys()
                ]
                for r in results
            ]
        )

        lower = np.nanmin(Y, axis=0)
        if np.isnan(lower).any():
            return [0] * len(results)

        # Translate Y into positive quadrant
        Y -= lower

        # Sample the objective space to estimate the dominated hypervolume using Monte-Carlo-sampling
        N = 10000

        # TODO: Replace uniform sampling by sampling 1-dimensional KDEs

        # Calculate reference point using quantiles for robustness
        reference_point = np.nanquantile(Y, quantile, axis=0) / quantile

        samples = np.random.uniform(0, reference_point, (N, Y.shape[1]))

        # Pareto dominance of the solutions over the samples
        n_dominates = _dominance(Y, samples).sum(axis=1)
        loss = 1 - n_dominates / N

        return loss.tolist()

    def n_dominated(self, trials: List["Trial"]):
        results = [self._trial_to_losses(t) for t in trials]

        # Y.shape(n_trials, n_objectives)
        Y = np.array(
            [
                [
                    r[objective] if r is not None else np.inf
                    for objective in self._minimize.keys()
                ]
                for r in results
            ]
        )

        return _dominance(Y, Y).sum(axis=0).tolist()

    def _trial_to_losses(self, trial: "Trial"):
        try:
            return self.invert_signs(trial.result)
        except:
            print(f"Error converting {trial.id} to losses")
            raise


def _dominance(X, Y):
    """Strict dominance of of X over Y."""
    return (X[:, None, :] <= Y).all(axis=-1) & (X[:, None, :] < Y).any(axis=-1)

