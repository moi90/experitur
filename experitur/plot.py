import difflib
import warnings
from abc import ABC, abstractmethod
from inspect import signature
from typing import TYPE_CHECKING, Any, Iterable, List, Mapping, Optional, Tuple, Union

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import (
    Formatter,
    Locator,
    MaxNLocator,
    StrMethodFormatter,
    FuncFormatter,
)
from scipy.stats.distributions import rv_discrete, uniform
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelBinarizer

from experitur.util import freeze

if TYPE_CHECKING:
    from experitur.core.trial import TrialCollection
    from experitur.optimization import Objective

try:
    from natsort import natsorted
except ImportError:
    warnings.warn("Falling back to sorted for natsorted")

    def natsorted(x):
        return sorted(x)


class Transformer(ABC):
    @abstractmethod
    def transform(self, X):
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, X):
        raise NotImplementedError

    @property
    def transformed_size(self) -> int:
        return 1


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

    @property
    def transformed_size(self) -> int:
        if self._lb.y_type_ == "binary":
            return 1
        return len(self._lb.classes_)


class Dimension(ABC):
    transformer: Transformer
    formatter: Optional[Formatter]
    locator: Optional[Locator]

    def __init__(self, label=None):
        self.name = None  # Set by initialize
        self.label = label

    @staticmethod
    def get_instance(name, values, dimension_or_label: Optional["Dimension"] = None):

        if isinstance(dimension_or_label, Dimension):
            dimension = dimension_or_label
        else:
            dimension = _KIND_TO_DIMENSION[values.dtype.kind]()
            dimension.label = dimension_or_label

        try:
            return dimension.initialize(name, values)
        except:
            print(
                f"Error initializing {dimension.__class__.__name__} {name} from {values}"
            )
            raise

    def initialize(self, name, values):
        self.name = name

        if self.label is None:
            self.label = name

        return self

    @abstractmethod
    def rvs_transformed(self, n_samples):
        """Draw samples in the transformed space."""
        pass

    @abstractmethod
    def linspace(self, n_samples) -> np.ndarray:
        """
        Evenly samples up to n_samples from the original space.

        Used in partial_dependence.
        """
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
        return self.transformer.transformed_size

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
    """
    Arguments:
        low (optional): Lower bound of the interval.
        high (optional): Upper bound of the interval.
        scale (optional): Scaling applied to the values.
        formatter (:class:`~matplotlib.ticker.Formatter`, optional): Formatter for axis ticks.
        label (str, optional): Axis label.
        replace_na (optional): Replace NaNs with this value.

    If low, high or name are not set during construction, they will be guessed from the provided values.
    """

    def __init__(
        self,
        low=None,
        high=None,
        *,
        scale="linear",
        formatter=None,
        locator=None,
        label=None,
        replace_na=None,
        marks: Optional[List] = None,
        ticks: Optional[List] = None,
    ):
        super().__init__()

        self.low = low
        self.high = high
        self.label = label
        self.replace_na = replace_na
        self.formatter = formatter
        self.marks = marks
        self.ticks = ticks

        if locator is None and isinstance(self, Integer):
            locator = MaxNLocator(integer=True)

        self.locator = locator

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
            self.low, self.high, is_int=isinstance(self, Integer),
        )

        return self._transformer

    def initialize(self, name, values):
        super().initialize(name, values)

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
        label=None,
        replace: Optional[Mapping[str, str]] = None,
        formatter=None,
        locator=None,
    ):
        super().__init__()

        self.label = label
        self.categories = categories
        self.replace = replace
        self.formatter = formatter
        self.locator = locator

    def initialize(self, name, values):
        super().initialize(name, values)

        values = self.prepare(values)

        if self.categories is None:
            self.categories = natsorted(set(values))

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

    def transform(self, X) -> np.ndarray:
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

    def rvs_transformed(self, n_samples):
        """Draw samples in the transformed space."""

        prior = np.tile(1.0 / len(self.categories), len(self.categories))

        numerical = rv_discrete(values=(range(len(self.categories)), prior)).rvs(
            size=n_samples
        )

        return self.transform(np.array(self.categories)[numerical])

    def linspace(self, n_samples) -> np.ndarray:
        """
        Evenly sample the original space.

        Ignores n_samples and returns the list of categories instead to avoid duplicate samples.
        """

        del n_samples

        return np.array(self.categories)

    def to_numeric(self, X, jitter=0):
        X = self.prepare(X)

        mapping = {x: i for i, x in enumerate(self.categories)}
        X = X.map(mapping)

        assert not X.isna().any()

        if jitter:
            X = X + np.random.randn(len(X)) * jitter

        return X


# After all Dimension types are defined, we can define the mapping of dtype kinds
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
    space: Space, model, i, j=None, sample_points=None, n_points=40,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Calculates the partial dependence of one parameter or a pair of parameters.

    Returns:
        (xi, zi, errors) if  j is None.
        (xi, yi, zi) if j is not None such that len(xi) == M is the number of columns in Z and len(yi) == N is the number of rows in Z.
    """

    dim_locs = np.cumsum([0] + [d.transformed_size for d in space.dimensions])

    assert sample_points.shape[-1] == dim_locs[-1]

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

            try:
                sample_points_[:, dim_locs[j] : dim_locs[j + 1]] = x_
            except:
                print(f"Error for dimension {space.dimensions[j]}")
                print("Transformed linspace:", x_)
                print("transformed_size:", space.dimensions[j].transformed_size)
                print("dim_locs:", dim_locs)
                print("sample_points_.shape:", sample_points_.shape)
                raise

            try:
                sample_points_[:, dim_locs[i] : dim_locs[i + 1]] = y_
            except:
                print(f"Error for dimension {space.dimensions[i]}")
                print("Transformed linspace:", y_)
                print("transformed_size", space.dimensions[i].transformed_size)
                raise

            row.append(np.mean(model.predict(sample_points_)))
        zi.append(row)

    return xi, yi, np.array(zi).T


def _rand_jitter(arr):
    stdev = 0.01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def _link_matching_points(
    ax: Axes,
    results,
    varying_parameters,
    parameter,
    dimension,
    objective,
    *,
    swapaxes=False,
    link_opt=None,
):
    """Show fine lines that link points that vary only in parameter."""

    if len(varying_parameters) > 1:
        groups = [
            group
            for _, group in results.groupby(
                [p for p in varying_parameters if p != parameter]
            )
        ]
    else:
        groups = [results]

    for group in groups:
        if len(group) < 2:
            continue

        # for v, configurations in group.groupby(parameter):
        #     if len(configurations) > 1:
        #         print(f"Warning: Multiple configurations for {parameter}={v}:")
        #         print(configurations)

        if isinstance(dimension, Categorical):
            group[parameter] = dimension.to_numeric(group[parameter])

        group = group.sort_values(parameter)

        if link_opt == "max":
            xy = zip(*[(v, c[objective].max()) for v, c in group.groupby(parameter)])
        elif link_opt == "min":
            xy = zip(*[(v, c[objective].min()) for v, c in group.groupby(parameter)])
        elif link_opt is None:
            xy = (group[parameter], group[objective])
        else:
            raise ValueError(f"Unknown link_opt value: {link_opt!r}")

        if swapaxes:
            xy = reversed(xy)

        ax.plot(*xy, lw=0.5, c="k", alpha=0.5)


def _plot_partial_dependence_nd(
    *,
    objective_dim: Numeric,
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
    show_optima,
    title,
    gs_kwargs,
    xticklabels_kwargs,
    yticklabels_kwargs,
    highlight_levels: Iterable,
    link_matching_points: bool,
    error_bands: bool,
):
    n_parameters = len(varying_parameters)

    fig = plt.figure(constrained_layout=True, figsize=(12, 12),)

    ratios = [4.0] * (n_parameters - 1)
    gs = GridSpec(
        n_parameters,
        n_parameters + 1,
        figure=fig,
        width_ratios=[3.0] + ratios + [0.5],
        height_ratios=ratios + [3.0],
        **gs_kwargs,
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

    for ax in list(axes_ij.flat) + list(axes_i) + list(axes_j):
        if ax is None:
            continue
        ax.use_sticky_edges = False
        ax.margins(0.01)

    fig.suptitle(title)

    color_norm = matplotlib.colors.Normalize(objective_dim.low, objective_dim.high)

    for i, i_dim in enumerate(reversed(range(n_parameters - 1))):
        dim_row = space.dimensions[i_dim]
        axes_i[i].set_ylabel(dim_row.label)
        axes_i[i].set_xlim(objective_dim.low, objective_dim.high)

        results_i = results[varying_parameters[i_dim]]

        # Show partial dependence of dim_row on one objective
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
            axes_i[i].set_yticklabels(dim_row.categories, **yticklabels_kwargs)

        if dim_row.formatter is not None:
            axes_i[i].get_yaxis().set_major_formatter(dim_row.formatter)

        if dim_row.locator is not None:
            axes_i[i].get_yaxis().set_major_locator(dim_row.locator)

        if error_bands:
            axes_i[i].fill_betweenx(xi, upper, lower, alpha=0.5)

        axes_i[i].plot(yi, xi)

        # Show fine lines that link points that vary only in this current dimension (results_i)
        if link_matching_points:
            _link_matching_points(
                axes_i[i],
                results,
                varying_parameters,
                varying_parameters[i_dim],
                dim_row,
                objective,
                swapaxes=True,
            )

        # Show points
        axes_i[i].scatter(
            results[objective],
            results_i,
            c=results[objective],
            cmap=cmap,
            ec="w",
            norm=color_norm,
        )

        if show_optima:
            # Show optimum
            axes_i[i].axhline(
                results_i.loc[idx_opt], c="r", ls="--",
            )

        for l in highlight_levels:
            axes_i[i].axvline(l, c="red", ls="dashed")

        for j, j_dim in enumerate(reversed(range(i_dim + 1, n_parameters))):
            dim_col = space.dimensions[j_dim]

            results_j = results[varying_parameters[j_dim]]

            if isinstance(dim_col, Categorical):
                results_j = dim_col.to_numeric(results_j, jitter)
                axes_j[j].set_xticks(list(range(len(dim_col.categories))))
                axes_j[j].set_xticklabels(dim_col.categories)

            if xticklabels_kwargs:
                # Only if xticklabels_kwargs is non-empty, otherwise setp thinks it should print the current values.
                plt.setp(axes_j[j].get_xticklabels(), **xticklabels_kwargs)

            if i == n_parameters - 2:
                # Show partial dependence of dim_col on one objective
                axes_j[j].set_xlabel(dim_col.label)
                axes_j[j].set_ylim(objective_dim.low, objective_dim.high)

                if dim_col.formatter is not None:
                    axes_j[j].get_xaxis().set_major_formatter(dim_col.formatter)

                if dim_col.locator is not None:
                    axes_j[j].get_xaxis().set_major_locator(dim_col.locator)

                xi, yit, errors = partial_dependence(
                    space, model, j_dim, n_points=n_points, sample_points=samples
                )

                yi = objective_dim.inverse_transform(yit)
                upper = objective_dim.inverse_transform(yit + errors)
                lower = objective_dim.inverse_transform(yit - errors)

                if isinstance(dim_col, Categorical):
                    xi = dim_col.to_numeric(xi)

                if error_bands:
                    axes_j[j].fill_between(xi, upper, lower, alpha=0.5)

                axes_j[j].plot(xi, yi)

                # Show fine lines that link points that vary only in this current dimension (results_i)
                if link_matching_points:
                    _link_matching_points(
                        axes_j[j],
                        results,
                        varying_parameters,
                        varying_parameters[j_dim],
                        dim_col,
                        objective,
                        swapaxes=False,
                    )

                # Plot true observations
                axes_j[j].scatter(
                    results_j,
                    results[objective],
                    c=results[objective],
                    cmap=cmap,
                    ec="w",
                    norm=color_norm,
                )

                if show_optima:
                    # Show optimum
                    axes_j[j].axvline(
                        results_j.loc[idx_opt], c="r", ls="--",
                    )

                for l in highlight_levels:
                    axes_j[j].axhline(l, c="red", ls="dashed")

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

            # xi and yi must be 1-D such that len(xi) == M is the number of columns in Z and len(yi) == N is the number of rows in Z.
            cnt = axes_ij[i, j].contourf(xi, yi, zi, levels, norm=color_norm, cmap=cmap)

            # Fix for countour lines showing in PDF autput:
            # https://stackoverflow.com/a/32911283/1116842
            for c in cnt.collections:
                c.set_edgecolor("face")

            if highlight_levels:
                axes_ij[i, j].contour(
                    xi, yi, zi, highlight_levels, colors="red", linestyles="dashed"
                )

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

            if show_optima:
                # Plot optimum
                # TODO:
                axes_ij[i, j].scatter(
                    results_j.loc[idx_opt], results_i.loc[idx_opt], fc="none", ec="r",
                )

    for ax in axes_i[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    for ax in axes_j[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    # ax[-2, 0].set_xlabel(objective_dim)
    # ax[-2, 0].xaxis.set_tick_params(labelbottom=True)

    # ax[-1, 1].set_ylabel(objective_dim)
    # ax[-1, 1].yaxis.set_tick_params(labelleft=True)

    axes_i[-1].set_xlabel(objective_dim.label)
    axes_j[0].set_ylabel(objective_dim.label)

    cax = fig.add_subplot(gs[:-1, -1])
    cb = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=color_norm, cmap=cmap),
        cax=cax,
        label=objective_dim.label,
    )

    if highlight_levels:
        for l in highlight_levels:
            cb.ax.axhline(l, c="red", ls="dashed")


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
    show_optima,
    title,
):
    fig = plt.figure(constrained_layout=True, figsize=(12, 12),)
    ax = fig.add_subplot(111)

    fig.suptitle(title)

    color_norm = matplotlib.colors.Normalize(
        results[objective].min(), results[objective].max()
    )

    ax.set_xlabel(space.dimensions[0].label)
    ax.set_ylabel(objective_dim.label)

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

    if show_optima:
        # Show optimum
        ax.axvline(
            results.loc[idx_opt, parameter], c="r", ls="--",
        )


_RUNTIME_DIVISORS = {"s": 1, "min": 60, "h": 60 * 60, "d": 24 * 60 * 60}


def _try_get(mapping: Mapping[str, Any], key: str):
    try:
        return mapping[key]
    except KeyError:
        close_matches = ", ".join(
            repr(m) for m in difflib.get_close_matches(key, mapping.keys(), cutoff=0.1)
        )
        if close_matches:
            print(f"{key!r} not found. Did you mean one of the following?")
            print(close_matches)
        raise


def plot_partial_dependence(
    trials: "TrialCollection",
    objective,
    *,
    dimensions=None,
    model=None,
    cmap=None,
    maximize=False,
    runtime_unit="s",
    jitter=0.025,
    ignore=None,
    show_optima=True,
    title=None,
    gs_kwargs=None,
    xticklabels_kwargs=None,
    yticklabels_kwargs=None,
    highlight_levels=None,
    link_matching_points=False,
    error_bands=True,
):
    if dimensions is None:
        dimensions = {}

    if gs_kwargs is None:
        gs_kwargs = {}

    if xticklabels_kwargs is None:
        xticklabels_kwargs = {}
    if yticklabels_kwargs is None:
        yticklabels_kwargs = {}

    if ignore is None:
        ignore = []

    ignore = set(ignore)

    if highlight_levels is None:
        highlight_levels = []

    if not isinstance(highlight_levels, Iterable):
        highlight_levels = [highlight_levels]

    highlight_levels = sorted(highlight_levels)

    if cmap is None:
        cmap = "viridis" if maximize else "viridis_r"

    runtime_divisor = _RUNTIME_DIVISORS[runtime_unit]

    varying_parameters = sorted(
        p for p in trials.varying_parameters.keys() if p not in ignore
    )

    results = pd.DataFrame(
        (
            {
                objective: _try_get(t.result, objective),
                "_runtime": (t.time_end - t.time_start).total_seconds()
                / runtime_divisor,
                **{p: t.get(p) for p in varying_parameters},
            }
            for t in trials
            if t.is_successful and t.result is not None
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

    objective_dim = Dimension.get_instance(
        f"Runtime ({runtime_unit})" if objective == "_runtime" else objective,
        results[objective],
        dimensions.get(objective),
    )

    if title is None:
        title = objective_dim.label

    # Calculate optimum
    idx_opt = results[objective].idxmax() if maximize else results[objective].idxmin()

    print("Optimum:")
    print(results.loc[idx_opt])

    if not len(results):
        raise ValueError("No results!")

    space = Space(
        [
            Dimension.get_instance(p, results[p], dimensions.get(p))
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
            objective_dim=objective_dim,
            results=results,
            objective=objective,
            space=space,
            model=model,
            varying_parameters=varying_parameters,
            n_points=n_points,
            samples=samples,
            cmap=cmap,
            idx_opt=idx_opt,
            jitter=jitter,
            show_optima=show_optima,
            title=title,
            gs_kwargs=gs_kwargs,
            xticklabels_kwargs=xticklabels_kwargs,
            yticklabels_kwargs=yticklabels_kwargs,
            highlight_levels=highlight_levels,
            link_matching_points=link_matching_points,
            error_bands=error_bands,
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
            show_optima,
            title,
        )


def joinex(parts):
    parts = list(parts)

    return ", ".join(parts[:-1]) + " and " + parts[-1]


def textplot(xx, yy, ss, *, ax, **kwargs):
    for x, y, s in zip(xx, yy, ss):
        ax.text(x, y, str(s), **kwargs)


def _get_formatter(formatter: Union[str, Formatter]) -> Formatter:
    if isinstance(formatter, Formatter):
        return formatter

    if isinstance(formatter, str):
        return StrMethodFormatter(formatter)

    if callable(formatter):
        return FuncFormatter(formatter)

    raise ValueError(f"Unknown formatter: {formatter!r}")


def plot_parameters_objectives(
    trials,
    objectives,
    *,
    dimensions=None,
    show_partial_dependence=True,
    model=None,
    ignore=None,
    title=None,
    highlight_levels=None,
    link_matching_points=False,
    xticklabels_kwargs=None,
    number_points=False,
    mark_kwds=None,
):

    if dimensions is None:
        dimensions = {}

    if ignore is None:
        ignore = []

    if xticklabels_kwargs is None:
        xticklabels_kwargs = {}

    if mark_kwds is None:
        mark_kwds = {}

    ignore = set(ignore)

    if highlight_levels is None:
        highlight_levels = []

    if not isinstance(highlight_levels, Iterable):
        highlight_levels = [highlight_levels]

    highlight_levels = sorted(highlight_levels)

    varying_parameters = sorted(
        p for p in trials.varying_parameters.keys() if p not in ignore
    )

    data = pd.DataFrame(
        (
            {
                **{p: t.get(p) for p in varying_parameters},
                **{target: t.result.get(target) for target in objectives},
            }
            for t in trials
            if t.is_successful and t.result is not None
        ),
        dtype=object,
    )

    for t in objectives:
        data[t] = data[t].infer_objects()

    for p in varying_parameters:
        if not isinstance(dimensions.get(p), Categorical):
            data[p] = data[p].infer_objects()

    data = data.dropna(subset=objectives)

    # Sort varying_parameters by position in dimensions
    parameter_position = {k: i for i, k in enumerate(dimensions.keys())}
    varying_parameters = sorted(
        varying_parameters, key=lambda p: parameter_position.get(p, 1000)
    )

    parameter_space = Space(
        [
            Dimension.get_instance(p, data[p], dimensions.get(p))
            for p in varying_parameters
        ]
    )

    objective_space = Space(
        [Dimension.get_instance(t, data[t], dimensions.get(t)) for t in objectives]
    )

    print(parameter_space)

    # Prepare results
    for p, dim in zip(varying_parameters, parameter_space.dimensions):
        data[p] = dim.prepare(data[p])

    n_samples = 1000
    samples = parameter_space.rvs_transformed(n_samples=n_samples)

    Xt = parameter_space.transform(*(data[p] for p in varying_parameters))
    Yt = objective_space.transform(*(data[t] for t in objectives))

    if model is None:
        n_neighbors = min(5, len(Xt))
        model = KNeighborsRegressor(weights="distance", n_neighbors=n_neighbors)

    regressor = MultiOutputRegressor(model)
    regressor.fit(Xt, Yt)

    fig, axes = plt.subplots(
        nrows=len(objective_space.dimensions),
        ncols=len(parameter_space.dimensions),
        sharey="row",
        sharex="col",
        squeeze=False,
    )

    for i, parameter in enumerate(parameter_space.dimensions):
        for j, target in enumerate(objective_space.dimensions):
            xi, y, e = partial_dependence(
                parameter_space, regressor.estimators_[j], i, sample_points=samples
            )
            yi = target.inverse_transform(y)

            data_p = data[parameter.name]

            if isinstance(parameter, Categorical):
                xi = parameter.to_numeric(xi)
                data_p = parameter.to_numeric(data_p, False)
                axes[j, i].set_xticks(list(range(len(parameter.categories))))
                if parameter.formatter:
                    formatter = _get_formatter(parameter.formatter)
                    categories = [formatter(x=x) for x in parameter.categories]
                else:
                    categories = parameter.categories
                axes[j, i].set_xticklabels(categories)

            if isinstance(target, Numeric) and target.marks is not None:
                for m in target.marks:
                    axes[j, i].axhline(m, **mark_kwds)

            if isinstance(parameter, Numeric) and parameter.marks is not None:
                for m in parameter.marks:
                    axes[j, i].avhline(m, **mark_kwds)

            # Format all xticklabels
            if xticklabels_kwargs:
                # Only if xticklabels_kwargs is non-empty, otherwise setp thinks it should print the current values.
                plt.setp(axes[j, i].get_xticklabels(), **xticklabels_kwargs)

            if show_partial_dependence:
                axes[j, i].plot(xi, yi)

            # Show points
            if number_points:
                textplot(
                    data_p,
                    data[target.name],
                    data.index,
                    ax=axes[j, i],
                    va="center",
                    ha="center",
                )
            else:
                axes[j, i].scatter(
                    data_p,
                    data[target.name],
                    # c=results[target],
                    # cmap=cmap,
                    ec="w",
                    # norm=color_norm,
                )

            if link_matching_points:
                link_opt = (
                    None if link_matching_points is True else link_matching_points
                )
                _link_matching_points(
                    axes[j, i],
                    data,
                    varying_parameters,
                    parameter.name,
                    parameter,
                    target.name,
                    link_opt=link_opt,
                )

            if j == len(objective_space.dimensions) - 1:
                axes[j, i].set_xlabel(parameter.label)
                if parameter.formatter is not None and not isinstance(
                    parameter, Categorical
                ):
                    axes[j, i].get_xaxis().set_major_formatter(parameter.formatter)

                if isinstance(parameter, Numeric) and parameter.ticks is not None:
                    axes[j, i].set_xticks(parameter.prepare(parameter.ticks))
                    ticks = parameter.ticks
                    if parameter.formatter is not None:
                        ticks = [parameter.formatter(t, None) for t in ticks]
                    axes[j, i].set_xticklabels(ticks)

            if i == 0:
                axes[j, i].set_ylabel(target.label)
                if target.formatter is not None:
                    axes[j, i].get_yaxis().set_major_formatter(target.formatter)

    if title is None:
        title = "Partial dependence of {} on {}".format(
            joinex(d.label for d in objective_space.dimensions),
            joinex(d.label for d in parameter_space.dimensions),
        )

    if title:
        fig.suptitle(title)

    fig.align_ylabels(axes)
    fig.align_xlabels(axes)
