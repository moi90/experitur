import math

import numpy as np
import pytest

from experitur import Experiment, Trial
from experitur.configurators import Grid, Prune
from experitur.core.context import Context


def B(S, B0, k, t):
    return S - (S - B0) * math.exp(-k * t)


# Skip with -k not slow
@pytest.mark.slow
@pytest.mark.parametrize("min_steps", [1, 5])
@pytest.mark.parametrize("min_count", [1, 5])
@pytest.mark.parametrize("maximize,minimize", [("gain", None), (None, "loss")])
def test_pruning(tmp_path, min_steps, min_count, maximize, minimize):
    with Context(str(tmp_path), writable=True) as ctx:

        @Prune(
            "i",
            quantile=0.5,
            maximize=maximize,
            minimize=minimize,
            min_steps=min_steps,
            min_count=min_count,
        )
        @Grid({"k": np.linspace(1.0, 0.0, 21)})
        @Experiment()
        def experiment(trial: Trial):

            for i in range(20):
                t = i / 4.0
                gain = B(1, 0, trial["k"], t)
                loss = B(-1, 1, trial["k"], t)

                trial.log(i=i, gain=gain, t=t, loss=loss)

                if trial.should_prune():
                    print(f"Trial {trial.id} pruned in iteration {i}")
                    break

            return {"last_i": i, "last_t": t, "last_gain": gain, "last_loss": loss}

    ctx.run()

    print(tmp_path)

    trials = ctx.trials.match()

    assert min(t.result["last_i"] for t in trials) >= min_steps
    assert sum(1 for t in trials if t.result["last_i"] == 19) == min_count
    assert max(t.result["last_gain"] for t in trials) > 0.9
    assert min(t.result["last_loss"] for t in trials) < 0.1
