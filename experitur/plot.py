import itertools
from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Callable, TYPE_CHECKING, List, Mapping, Optional, Tuple, Union

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skopt.optimizer
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats.distributions import rv_discrete, uniform
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelBinarizer
from matplotlib.ticker import Formatter, StrMethodFormatter

from experitur.util import freeze

if TYPE_CHECKING:
    from experitur.core.trial import TrialCollection
    from experitur.optimization import Objective


class Transformer(ABC):
    @abstractmethod
    def transform(self, X):
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, X):
        raise NotImplementedError


class Identity(Transformer):
    def transform(self, X):
        return X

    def inverse_transform(self, Xt):
        return Xt


def not_na(x):
    if x is None:
        return False

    if np.isnan(x):
        return False

    try:
        return ~np.isnat(x)
    except TypeError:
        pass

    return True


class Normalize(Transformer):
    def __init__(self, low, high, is_int=False):
        self.low = float(low)
        self.high = float(high)
        self.is_int = is_int

    def transform(self, X):
        X: np.ndarray = np.asarray(X)

        if np.any(np.isnan(X)):
            raise ValueError("X contains NaNs")

        if np.any(X > self.high + 1e-8):
            raise ValueError(f"All values should be less than {self.high}")
        if np.any(X < self.low - 1e-8):
            raise ValueError(f"All values should be greater than {self.low}")

        return (X - self.low) / (self.high - self.low)

    def inverse_transform(self, Xt):
        Xt: np.ndarray = np.asarray(Xt)

        X_orig = Xt * (self.high - self.low) + self.low
        if self.is_int:
            return np.round(X_orig).astype(np.int)
        return X_orig


class CategoricalEncoder(Transformer):
    """OneHotEncoder that can handle categorical variables."""

    def __init__(self):
        """Convert labeled categories into one-hot encoded features."""
        self._lb = LabelBinarizer()

    def fit(self, X):
        """Fit a list or array of categories.

        Parameters
        ----------
        X : array-like, shape=(n_categories,)
            List of categories.
        """
        self.mapping_ = {v: i for i, v in enumerate(X)}
        self.inverse_mapping_ = {i: v for v, i in self.mapping_.items()}
        self._lb.fit([self.mapping_[v] for v in X])
        self.n_classes = len(self._lb.classes_)

        return self

    def transform(self, X):
        """Transform an array of categories to a one-hot encoded representation.

        Parameters
        ----------
        X : array-like, shape=(n_samples,)
            List of categories.

        Returns
        -------
        Xt : array-like, shape=(n_samples, n_categories)
            The one-hot encoded categories.
        """
        return self._lb.transform([self.mapping_[v] for v in X])

    def inverse_transform(self, Xt):
        """Inverse transform one-hot encoded categories back to their original
           representation.

        Parameters
        ----------
        Xt : array-like, shape=(n_samples, n_categories)
            One-hot encoded categories.

        Returns
        -------
        X : array-like, shape=(n_samples,)
            The original categories.
        """
        Xt: np.ndarray = np.asarray(Xt)
        return [self.inverse_mapping_[i] for i in self._lb.inverse_transform(Xt)]


class Dimension(ABC):
    name: Optional[str]
    transformer: Transformer
    formatter: Optional[Formatter]

    @staticmethod
    def from_values(name, values, dimension=None):
        if dimension is None:
            dimension = _KIND_TO_DIMENSION[values.dtype.kind]()

        return dimension.init(name, values)

    @abstractmethod
    def init(self, name, values):
        return self

    @abstractmethod
    def rvs_transformed(self, n_samples):
        """Draw samples in the transformed space."""
        pass

    @abstractmethod
    def linspace(self, n_samples):
        """Evenly sample the original space."""
        pass

    def prepare(self, X) -> pd.Series:
        """
        Prepare the values in X.

        E.g. fill NaNs, replace values, ...
        """

        return pd.Series(X)

    def transform(self, X):
        """Transform samples form the original space to a warped space."""

        try:
            return self.transformer.transform(X)
        except:
            print(f"self: {self!r}")
            print(f"X: {X!r}")
            raise

    @property
    def transformed_size(self):
        return 1

    def __repr__(self):
        parameters = ", ".join(
            f"{p}={getattr(self, p)!r}" for p in signature(self.__init__).parameters
        )
        return f"{self.__class__.__name__}({parameters})"

    def __str__(self):
        return self.name


def _uniform_inclusive(loc=0.0, scale=1.0):
    # like scipy.stats.distributions but inclusive of `high`
    return uniform(loc=loc, scale=np.nextafter(scale, scale + 1.0))


class Numeric(Dimension):
    def __init__(
        self,
        low=None,
        high=None,
        *,
        scale="linear",
        formatter=None,
        name=None,
        replace_na=None,
    ):
        self.low = low
        self.high = high
        self.name = name
        self.replace_na = replace_na
        self.formatter = formatter

        if scale in ("linear", None):
            self.scale = None
        elif scale == "log10":
            self.scale = np.log10
            self.formatter = formatter or StrMethodFormatter("1e{x}")
        elif callable(scale):
            self.scale = scale
        else:
            raise ValueError(f"Unknown scale parameter: {scale!r}")

    @property
    def transformer(self):
        try:
            return self._transformer  # pylint: disable=access-member-before-definition
        except AttributeError:
            pass

        self._transformer = Normalize(
            self.low,
            self.high,
            is_int=isinstance(self, Integer),
        )

        return self._transformer

    def init(self, name, values):
        # Replace NaNs
        values = self.prepare(values)

        if np.isnan(values).any():
            print(
                f"Warning: {name} contains NaN values. Supply dimensions={{'{name}': Real/Integer(..., fillna=...)}}."
            )

        values = np.asarray(values)

        if self.name is None:
            self.name = name

        if self.low is None:
            self.low = values.min()

        if self.high is None:
            self.high = values.max()

        return self

    def rvs_transformed(self, n_samples):
        return _uniform_inclusive(0, 1).rvs(size=n_samples)

    def linspace(self, n_samples):
        return np.linspace(self.low, self.high, n_samples)

    def transform(self, X):
        """
        Transform samples form the original space to a warped space.

        This does not include preparation.
        """

        try:
            return self.transformer.transform(X)
        except:
            print(f"self: {self!r}")
            print(f"X: {X!r}")
            raise

    def inverse_transform(self, Xt):
        return self.transformer.inverse_transform(Xt)

    def prepare(self, X):
        X = super().prepare(X)

        if self.replace_na is not None:
            X = X.fillna(self.replace_na)

        if self.scale is not None:
            X = self.scale(X)

        return X


class Real(Numeric):
    pass


class Integer(Numeric):
    pass


class _PassThroughMissingDict(dict):
    def __missing__(self, key):
        return key


class Categorical(Dimension):
    """
    Args:
        replace (dict, optional): Map string representations to other strings.
    """

    def __init__(
        self,
        categories=None,
        *,
        name=None,
        replace: Optional[Mapping[str, str]] = None,
        formatter=None,
    ):
        self.name = name
        self.categories = categories
        self.replace = replace
        self.formatter = formatter

    def init(self, name, values):
        if self.name is None:
            self.name = name

        values = self.prepare(values)

        if self.categories is None:
            self.categories = sorted(set(values))

        return self

    def prepare(self, X):
        X = super().prepare(X)

        X = X.astype(str)

        if self.replace is not None:
            replace = _PassThroughMissingDict(self.replace)
            X = X.map(replace)

        return X

    @property
    def transformer(self):
        try:
            return self._transformer  # pylint: disable=access-member-before-definition
        except AttributeError:
            pass

        self._transformer = CategoricalEncoder()
        self._transformer.fit(self.categories)

        return self._transformer

    def transform(self, X):
        """
        Transform samples from the original space to a warped space.

        Does not include data preparation.
        """

        try:
            return self.transformer.transform(X)
        except:
            print()
            print(f"self: {self!r}")
            print(f"X: {X!r}")
            raise

    @property
    def transformed_size(self):
        return len(self.categories)

    def rvs_transformed(self, n_samples):
        """Draw samples in the transformed space."""

        prior = np.tile(1.0 / len(self.categories), len(self.categories))

        numerical = rv_discrete(values=(range(len(self.categories)), prior)).rvs(
            size=n_samples
        )

        return self.transform(np.array(self.categories)[numerical])

    def linspace(self, n_samples):
        """Evenly sample the original space."""

        return np.array(
            list(itertools.islice(itertools.cycle(self.categories), n_samples))
        )

    def to_numeric(self, X, jitter=0):
        X = self.prepare(X)

        mapping = {x: i for i, x in enumerate(self.categories)}
        X = X.map(mapping)

        assert not X.isna().any()

        if jitter:
            X = X + np.random.randn(len(X)) * jitter

        return X


_KIND_TO_DIMENSION = {
    "i": Integer,
    "u": Integer,
    "f": Real,
    "b": Integer,
    "O": Categorical,
}


class Space:
    def __init__(self, dimensions: List[Dimension]):
        self.dimensions = dimensions

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dimensions})"

    def transform(self, *vecs):
        return np.column_stack(
            [dim.transform(vec) for dim, vec in zip(self.dimensions, vecs)]
        )

    def rvs_transformed(self, n_samples):
        """Draw samples in the transformed space."""
        return np.column_stack(
            [dim.rvs_transformed(n_samples) for dim in self.dimensions]
        )


def partial_dependence(
    space,
    model,
    i,
    j=None,
    sample_points=None,
    n_points=40,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    dim_locs = np.cumsum([0] + [d.transformed_size for d in space.dimensions])

    # One-dimensional case
    if j is None:
        dim_i = space.dimensions[i]
        xi = dim_i.linspace(n_points)
        xi_t = dim_i.transform(xi)
        yi = []
        errors = []
        for x_ in xi_t:
            # Copy
            sample_points_ = np.array(sample_points)

            # Partial dependence according to Friedman (2001)
            sample_points_[:, dim_locs[i] : dim_locs[i + 1]] = x_
            predictions = model.predict(sample_points_)
            yi.append(np.mean(predictions))
            errors.append(np.std(predictions))

        if isinstance(dim_i, Categorical):
            dim_i: Categorical
            xi = xi[: len(dim_i.categories)]
            yi = yi[: len(dim_i.categories)]
            errors = errors[: len(dim_i.categories)]

        return xi, np.array(yi), np.array(errors)

    # Two-dimensional case
    xi = space.dimensions[j].linspace(n_points)
    xi_t = space.dimensions[j].transform(xi)
    yi = space.dimensions[i].linspace(n_points)
    yi_t = space.dimensions[i].transform(yi)

    zi = []
    for x_ in xi_t:
        row = []
        for y_ in yi_t:
            sample_points_ = np.array(sample_points, copy=True)  # copy
            sample_points_[:, dim_locs[j] : dim_locs[j + 1]] = x_
            sample_points_[:, dim_locs[i] : dim_locs[i + 1]] = y_
            row.append(np.mean(model.predict(sample_points_)))
        zi.append(row)

    return xi, yi, np.array(zi).T


def _rand_jitter(arr):
    stdev = 0.01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def _plot_partial_dependence_nd(
    objective_dim,
    results: pd.DataFrame,
    objective,
    space,
    model,
    varying_parameters,
    n_points,
    samples,
    cmap,
    idx_opt,
    jitter,
):
    n_parameters = len(varying_parameters)

    fig = plt.figure(
        constrained_layout=True,
        figsize=(12, 12),
    )

    ratios = [4] * (n_parameters - 1)
    gs = GridSpec(
        n_parameters,
        n_parameters + 1,
        figure=fig,
        width_ratios=[3] + ratios + [0.5],
        height_ratios=ratios + [3],
    )

    # Create axes
    axes_i = np.array([fig.add_subplot(gs[i, 0]) for i in range(n_parameters - 1)])
    axes_j = np.array([fig.add_subplot(gs[-1, j + 1]) for j in range(n_parameters - 1)])
    axes_ij = np.array(
        [
            [
                fig.add_subplot(gs[i, j + 1], sharex=axes_j[j], sharey=axes_i[i])
                if j <= i
                else None
                for j in range(n_parameters - 1)
            ]
            for i in range(n_parameters - 1)
        ]
    )

    # Legend
    if False:
        # TODO: Use blend patch for 2d
        # https://stackoverflow.com/a/55501861/1116842
        lax = fig.add_subplot(gs[-1, 0])
        lax.axis("off")
        legend_elements = []
        legend_elements.append(
            Patch(color=(0, 0, 0, 0), label="Cluster selection method:")
        )
        lax.legend(handles=legend_elements, loc="center")

    for ax in list(axes_ij.flat) + list(axes_i) + list(axes_j):
        if ax is None:
            continue
        ax.use_sticky_edges = False
        ax.margins(0.01)

    fig.suptitle(objective_dim.name)

    color_norm = matplotlib.colors.Normalize(
        results[objective].min(), results[objective].max()
    )

    for i, i_dim in enumerate(reversed(range(n_parameters - 1))):
        dim_row = space.dimensions[i_dim]
        axes_i[i].set_ylabel(dim_row)

        if isinstance(dim_row, Integer):
            axes_i[i].get_yaxis().set_major_locator(MaxNLocator(integer=True))

        results_i = results[varying_parameters[i_dim]]

        # Show partial dependence of dim_row on objective
        xi, yit, errors = partial_dependence(
            space, model, i_dim, n_points=n_points, sample_points=samples
        )

        yi = objective_dim.inverse_transform(yit)
        upper = objective_dim.inverse_transform(yit + errors)
        lower = objective_dim.inverse_transform(yit - errors)

        if isinstance(dim_row, Categorical):
            xi = dim_row.to_numeric(xi)
            results_i = dim_row.to_numeric(results_i, jitter)
            axes_i[i].set_yticks(list(range(len(dim_row.categories))))
            axes_i[i].set_yticklabels(dim_row.categories)

        if dim_row.formatter is not None:
            axes_i[i].get_yaxis().set_major_formatter(dim_row.formatter)

        axes_i[i].fill_betweenx(xi, upper, lower, alpha=0.5)
        axes_i[i].plot(yi, xi)
        axes_i[i].scatter(
            results[objective],
            results_i,
            c=results[objective],
            cmap=cmap,
            ec="w",
            norm=color_norm,
        )

        # Show optimum
        axes_i[i].axhline(
            results_i.loc[idx_opt],
            c="r",
            ls="--",
        )

        for j, j_dim in enumerate(reversed(range(i_dim + 1, n_parameters))):
            dim_col = space.dimensions[j_dim]

            results_j = results[varying_parameters[j_dim]]

            if isinstance(dim_col, Categorical):
                results_j = dim_col.to_numeric(results_j, jitter)
                axes_j[j].set_xticks(list(range(len(dim_col.categories))))
                axes_j[j].set_xticklabels(dim_col.categories)

            if dim_col.formatter is not None:
                axes_j[j].get_xaxis().set_major_formatter(dim_col.formatter)

            if i == n_parameters - 2:
                # Show partial dependence of dim_col on objective
                axes_j[j].set_xlabel(dim_col)

                if isinstance(dim_col, Integer):
                    axes_j[j].get_xaxis().set_major_locator(MaxNLocator(integer=True))

                xi, yit, errors = partial_dependence(
                    space, model, j_dim, n_points=n_points, sample_points=samples
                )

                yi = objective_dim.inverse_transform(yit)
                upper = objective_dim.inverse_transform(yit + errors)
                lower = objective_dim.inverse_transform(yit - errors)

                if isinstance(dim_col, Categorical):
                    xi = dim_col.to_numeric(xi)

                axes_j[j].fill_between(xi, upper, lower, alpha=0.5)

                axes_j[j].plot(xi, yi)

                # Plot true observations
                axes_j[j].scatter(
                    results_j,
                    results[objective],
                    c=results[objective],
                    cmap=cmap,
                    ec="w",
                    norm=color_norm,
                )

                # Show optimum
                axes_j[j].axvline(
                    results_j.loc[idx_opt],
                    c="r",
                    ls="--",
                )

            # Show partial dependence of dim_col/dim_col on objective
            # axes_ij[i, j].set_xlabel(dim_col)
            # axes_ij[i, j].set_ylabel(dim_row)

            # Hide tick labels for inner subplots
            plt.setp(axes_ij[i, j].get_xticklabels(), visible=False)
            plt.setp(axes_ij[i, j].get_yticklabels(), visible=False)

            xi, yi, zit = partial_dependence(
                space, model, i_dim, j_dim, sample_points=samples, n_points=n_points
            )

            zi = objective_dim.inverse_transform(zit)

            if isinstance(dim_row, Categorical):
                yi = dim_row.to_numeric(yi)

            if isinstance(dim_col, Categorical):
                xi = dim_col.to_numeric(xi)

            levels = 50

            cnt = axes_ij[i, j].contourf(xi, yi, zi, levels, norm=color_norm, cmap=cmap)

            # Fix for countour lines showing in PDF autput:
            # https://stackoverflow.com/a/32911283/1116842
            for c in cnt.collections:
                c.set_edgecolor("face")

            # Plot true observations
            axes_ij[i, j].scatter(
                results_j,
                results_i,
                c=results[objective],
                cmap=cmap,
                ec="w",
                norm=color_norm,
                alpha=0.75,
            )

            # Plot optimum
            # TODO:
            axes_ij[i, j].scatter(
                results_j.loc[idx_opt],
                results_i.loc[idx_opt],
                fc="none",
                ec="r",
            )

    # ax[-2, 0].set_xlabel(objective_dim)
    # ax[-2, 0].xaxis.set_tick_params(labelbottom=True)

    # ax[-1, 1].set_ylabel(objective_dim)
    # ax[-1, 1].yaxis.set_tick_params(labelleft=True)

    cax = fig.add_subplot(gs[:-1, -1])
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=color_norm, cmap=cmap),
        cax=cax,
    )


def _plot_partial_dependence_1d(
    objective_dim,
    results: pd.DataFrame,
    objective,
    space: Space,
    model,
    parameter: str,
    n_points,
    samples,
    cmap,
    idx_opt,
    jitter,
):
    fig = plt.figure(
        constrained_layout=True,
        figsize=(12, 12),
    )
    ax = fig.add_subplot(111)

    fig.suptitle(objective_dim.name)

    color_norm = matplotlib.colors.Normalize(
        results[objective].min(), results[objective].max()
    )

    ax.set_xlabel(space.dimensions[0])
    ax.set_ylabel(objective_dim.name)

    # Show partial dependence of dimension on objective
    xi, yit, errors = partial_dependence(
        space, model, 0, n_points=n_points, sample_points=samples
    )

    yi = objective_dim.inverse_transform(yit)
    upper = objective_dim.inverse_transform(yit + errors)
    lower = objective_dim.inverse_transform(yit - errors)

    if isinstance(space.dimensions[0], Categorical):
        xi = space.dimensions[0].to_numeric(xi)
        results_i = space.dimensions[0].to_numeric(results_i, jitter)
        ax.set_yticks(list(range(len(space.dimensions[0].categories))))
        ax.set_yticklabels(space.dimensions[0].categories)

    if space.dimensions[0].formatter is not None:
        ax.get_xaxis().set_major_formatter(space.dimensions[0].formatter)

    ax.plot(xi, yi)
    ax.scatter(
        results[parameter],
        results[objective],
        c=results[objective],
        cmap=cmap,
        ec="w",
        norm=color_norm,
    )

    # Show optimum
    ax.axvline(
        results.loc[idx_opt, parameter],
        c="r",
        ls="--",
    )


_RUNTIME_DIVISORS = {"s": 1, "min": 60, "h": 60 * 60, "d": 24 * 60 * 60}


def plot_partial_dependence(
    trials: "TrialCollection",
    objective,
    dimensions=None,
    model=None,
    objective_dim=None,
    cmap=None,
    maximize=False,
    runtime_unit="s",
    jitter=0.025,
    ignore=None,
):
    if dimensions is None:
        dimensions = {}

    if ignore is None:
        ignore = []

    ignore = set(ignore)

    if cmap is None:
        cmap = "viridis" if maximize else "viridis_r"

    runtime_divisor = _RUNTIME_DIVISORS[runtime_unit]

    varying_parameters = sorted(
        p for p in trials.varying_parameters.keys() if p not in ignore
    )

    results = pd.DataFrame(
        (
            {
                objective: t.result.get(objective),
                "_runtime": (t.time_end - t.time_start).total_seconds()
                / runtime_divisor,
                **{p: t.get(p) for p in varying_parameters},
            }
            for t in trials
            if t.result is not None
        ),
        dtype=object,
    )

    results[objective] = pd.to_numeric(results[objective])
    results = results.dropna(subset=[objective])

    varying_parameters = [
        c
        for c in results.columns
        if c not in (objective, "_runtime")
        and len(set(freeze(x) for x in results[c])) > 1
    ]
    n_parameters = len(varying_parameters)

    for c in results.columns:
        if not isinstance(dimensions.get(c), Categorical):
            results[c] = results[c].infer_objects()

    objective_dim = Dimension.from_values(
        f"Runtime ({runtime_unit})" if objective == "_runtime" else objective,
        results[objective],
        objective_dim,
    )

    # Calculate optimum
    idx_opt = results[objective].idxmax() if maximize else results[objective].idxmin()

    print("Optimum:")
    print(results.loc[idx_opt])

    if not len(results):
        raise ValueError("No results!")

    space = Space(
        [
            Dimension.from_values(p, results[p], dimensions.get(p))
            for p in varying_parameters
        ]
    )

    # Prepare results
    for p, dim in zip(varying_parameters, space.dimensions):
        results[p] = dim.prepare(results[p])

    print(space)

    n_samples = 1000
    n_points = 50
    samples = space.rvs_transformed(n_samples=n_samples)

    y = objective_dim.transform(results[objective])

    Xt = space.transform(*(results[p] for p in varying_parameters))

    if model is None:
        n_neighbors = min(5, len(Xt))
        model = KNeighborsRegressor(weights="distance", n_neighbors=n_neighbors)

    model.fit(Xt, y)

    if n_parameters > 1:
        _plot_partial_dependence_nd(
            objective_dim,
            results,
            objective,
            space,
            model,
            varying_parameters,
            n_points,
            samples,
            cmap,
            idx_opt,
            jitter,
        )
    else:
        _plot_partial_dependence_1d(
            objective_dim,
            results,
            objective,
            space,
            model,
            varying_parameters[0],
            n_points,
            samples,
            cmap,
            idx_opt,
            jitter,
        )
