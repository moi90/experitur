from experitur import Experiment, Trial
from experitur.core.context import Context


def mean(values):
    return sum(values) / len(values)


def test_logger(tmp_path):
    with Context(str(tmp_path), writable=True) as ctx:

        @Experiment()
        def a(trial: Trial):
            for i in range(10):
                trial.log(i=i)

            return trial.aggregate_log(["i"])

    ctx.run()

    trial = a.trials.one()

    assert list(trial.get_log()) == [{"i": i} for i in range(10)]

    assert trial.result == {"max_i": 9, "min_i": 0, "final_i": 9, "mean_i": 4.5}
