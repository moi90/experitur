from abc import ABC, abstractmethod
from inspect import signature
from typing import TYPE_CHECKING, Tuple, Union

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.stats.distributions import uniform
from sklearn.neighbors import KNeighborsRegressor

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
        X = np.asarray(X)

        if self.is_int:
            if np.any(np.round(X) > self.high):
                raise ValueError(f"All values should be less than {self.high}")
            if np.any(np.round(X) < self.low):
                raise ValueError(
                    f"All integer values should be greater than {self.low}"
                )
        else:
            if np.any(X > self.high + 1e-8):
                raise ValueError(f"All values should be less than {self.high}")
            if np.any(X < self.low - 1e-8):
                raise ValueError(f"All values should be greater than {self.low}")

        return (X - self.low) / (self.high - self.low)

    def inverse_transform(self, Xt):
        Xt = np.asarray(Xt)
        if np.any(Xt > 1.0):
            raise ValueError(
                f"All values should be less than 1.0. Maximum was {Xt.max()}"
            )
        if np.any(Xt < 0.0):
            raise ValueError(
                f"All values should be greater than 0.0. Minimum was {Xt.min()}"
            )
        X_orig = Xt * (self.high - self.low) + self.low
        if self.is_int:
            return np.round(X_orig).astype(np.int)
        return X_orig


class Dimension(ABC):
    name: str
    transformer: "Any"

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

    def transform(self, X):
        """Transform samples form the original space to a warped space."""

        # Replace NaNs

        try:
            return self.transformer.transform(X)
        except:
            print(f"{self!r}")
            print(f"{X!r}")
            raise

    @abstractmethod
    def fillna(self, X):
        pass

    @property
    def transformed_size(self):
        return 1

    def __repr__(self):
        parameters = ", ".join(
            f"{p}={getattr(self, p)}" for p in signature(self.__init__).parameters
        )
        return f"{self.__class__.__name__}({parameters})"

    def __str__(self):
        return self.name


def _uniform_inclusive(loc=0.0, scale=1.0):
    # like scipy.stats.distributions but inclusive of `high`
    return uniform(loc=loc, scale=np.nextafter(scale, scale + 1.0))


class Numeric(Dimension):
    def __init__(
        self, low=None, high=None, prior="uniform", base=10, name=None, replace_na=None
    ):
        self.low = low
        self.high = high
        self.prior = prior
        self.base = base
        self.name = name
        self.replace_na = replace_na

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

    def init(self, name, values):
        # Replace NaNs
        values = self.fillna(values)

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
        """Transform samples form the original space to a warped space."""

        # Replace NaNs
        X = self.fillna(X)

        try:
            return self.transformer.transform(X)
        except:
            print(f"{self!r}")
            print(f"{X!r}")
            raise

    def inverse_transform(self, Xt):
        return self.transformer.inverse_transform(Xt)

    def fillna(self, X):
        if self.replace_na is not None:
            if np.isscalar(X):
                return X if not_na(X) else self.replace_na

            return np.asarray([x if not_na(x) else self.replace_na for x in X])
        return X


class Real(Numeric):
    pass


class Integer(Numeric):
    pass


class Categorical(Dimension):
    def init(self, name, values):
        self.name = name
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"


_KIND_TO_DIMENSION = {
    "i": Integer,
    "u": Integer,
    "f": Real,
    "b": Integer,
    "O": Categorical,
}


class Space:
    def __init__(self, dimensions):
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
    space, model, i, j=None, sample_points=None, n_points=40,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    dim_locs = np.cumsum([0] + [d.transformed_size for d in space.dimensions])

    # One-dimensional case
    if j is None:
        xi = space.dimensions[i].linspace(n_points)
        xi_t = space.dimensions[i].transform(xi)
        yi = []
        for x_ in xi_t:
            sample_points_ = np.array(sample_points)

            # Partial dependence according to Friedman (2001)
            sample_points_[:, dim_locs[i] : dim_locs[i + 1]] = x_
            yi.append(np.mean(model.predict(sample_points_)))

        return xi, yi

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
):
    n_parameters = len(varying_parameters)

    fig = plt.figure(constrained_layout=True, figsize=(12, 12),)

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

        # Show partial dependence of dim_row on objective
        xi, yit = partial_dependence(
            space, model, i_dim, n_points=n_points, sample_points=samples
        )

        yi = objective_dim.inverse_transform(yit)

        axes_i[i].plot(yi, xi)
        axes_i[i].scatter(
            results[objective],
            dim_row.fillna(results[varying_parameters[i_dim]]),
            c=results[objective],
            cmap=cmap,
            ec="w",
            norm=color_norm,
        )

        # Show optimum
        axes_i[i].axhline(
            dim_row.fillna(results.loc[idx_opt, varying_parameters[i_dim]]),
            c="r",
            ls="--",
        )

        for j, j_dim in enumerate(reversed(range(i_dim + 1, n_parameters))):
            dim_col = space.dimensions[j_dim]

            if i == n_parameters - 2:
                # Show partial dependence of dim_col on objective
                axes_j[j].set_xlabel(dim_col)
                xi, yit = partial_dependence(
                    space, model, j_dim, n_points=n_points, sample_points=samples
                )

                yi = objective_dim.inverse_transform(yit)

                axes_j[j].plot(xi, yi)

                # Plot true observations
                axes_j[j].scatter(
                    dim_col.fillna(results[varying_parameters[j_dim]]),
                    results[objective],
                    c=results[objective],
                    cmap=cmap,
                    ec="w",
                    norm=color_norm,
                )

                # Show optimum
                axes_j[j].axvline(
                    dim_col.fillna(results.loc[idx_opt, varying_parameters[j_dim]]),
                    c="r",
                    ls="--",
                )

            # Show partial dependence of dim_col/dim_col on objective
            axes_ij[i, j].set_xlabel(dim_col)
            axes_ij[i, j].set_ylabel(dim_row)

            xi, yi, zit = partial_dependence(
                space, model, i_dim, j_dim, sample_points=samples, n_points=n_points
            )

            zi = objective_dim.inverse_transform(zit)

            levels = 50
            cnt = axes_ij[i, j].contourf(xi, yi, zi, levels, norm=color_norm, cmap=cmap)

            # Fix for countour lines showing in PDF autput:
            # https://stackoverflow.com/a/32911283/1116842
            for c in cnt.collections:
                c.set_edgecolor("face")

            # Plot true observations
            axes_ij[i, j].scatter(
                dim_col.fillna(results[varying_parameters[j_dim]]),
                dim_row.fillna(results[varying_parameters[i_dim]]),
                c=results[objective],
                cmap=cmap,
                ec="w",
                norm=color_norm,
                alpha=0.75,
            )

            # Plot optimum
            # TODO:
            axes_ij[i, j].scatter(
                dim_col.fillna(results.loc[idx_opt, varying_parameters[j_dim]]),
                dim_row.fillna(results.loc[idx_opt, varying_parameters[i_dim]]),
                fc="none",
                ec="r",
            )

    # ax[-2, 0].set_xlabel(objective_dim)
    # ax[-2, 0].xaxis.set_tick_params(labelbottom=True)

    # ax[-1, 1].set_ylabel(objective_dim)
    # ax[-1, 1].yaxis.set_tick_params(labelleft=True)

    cax = fig.add_subplot(gs[:-1, -1])
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=color_norm, cmap=cmap), cax=cax,
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
):
    fig = plt.figure(constrained_layout=True, figsize=(12, 12),)
    ax = fig.add_subplot(111)

    fig.suptitle(objective_dim.name)

    color_norm = matplotlib.colors.Normalize(
        results[objective].min(), results[objective].max()
    )

    ax.set_xlabel(space.dimensions[0])
    ax.set_ylabel(objective_dim.name)

    # Show partial dependence of dimension on objective
    xi, yit = partial_dependence(
        space, model, 0, n_points=n_points, sample_points=samples
    )

    yi = objective_dim.inverse_transform(yit)

    ax.plot(xi, yi)
    ax.scatter(
        space.dimensions[0].fillna(results[parameter]),
        results[objective],
        c=results[objective],
        cmap=cmap,
        ec="w",
        norm=color_norm,
    )

    # Show optimum
    ax.axvline(
        space.dimensions[0].fillna(results.loc[idx_opt, parameter]), c="r", ls="--",
    )


_RUNTIME_DIVISORS = {"s": 1, "min": 60, "h": 60 * 60, "d": 24 * 60 * 60}


def plot_partial_dependence(
    trials: "TrialCollection",
    objective,
    dimensions=None,
    model=None,
    objective_dim=None,
    cmap="viridis_r",
    maximize=False,
    runtime_unit="s",
):
    if dimensions is None:
        dimensions = {}

    runtime_divisor = _RUNTIME_DIVISORS[runtime_unit]

    varying_parameters = sorted(trials.varying_parameters.keys())
    n_parameters = len(varying_parameters)

    results = pd.DataFrame(
        {
            objective: t.result.get(objective),
            "_runtime": (t.time_end - t.time_start).total_seconds() / runtime_divisor,
            **{p: t.get(p) for p in varying_parameters},
        }
        for t in trials
        if t.result is not None
    )

    results[objective] = pd.to_numeric(results[objective])
    results = results.dropna(subset=[objective]).infer_objects()

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

    n_samples = 1000
    n_points = 50
    samples = space.rvs_transformed(n_samples=n_samples)

    y = objective_dim.transform(results[objective])

    Xt = space.transform(*(results[p].to_numpy() for p in varying_parameters))

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
        )

