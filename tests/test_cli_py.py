import inspect
import os.path

from click.testing import CliRunner

from experitur.cli import clean, collect, do, run

example_py = inspect.cleandoc(
    """
    from experitur import Experiment
    import numpy as np

    @Experiment(
        parameters={
            "a1": [1],
            "a2": [2],
            "b": [1, 2],
            "a": ["{a{b}}"],
        })
    def baseline(trial):
        pass

    @Experiment(parent=baseline)
    def e1(trial):
        pass

    @Experiment()
    def e2(trial):
        raise NotImplementedError()

    @Experiment()
    def e3(trial):
        pass

    @e3.command("cmd")
    def e3_cmd(trial):
        pass

    @e3.command("experiment_cmd", target="experiment")
    def e3_exp_cmd(experiment):
        pass
    """
)


def test_run():
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("example.py", "w") as f:
            f.write(example_py)

        result = runner.invoke(run, ["example.py"], catch_exceptions=True)
        print(result.output)
        assert result.exit_code == 0

        result = runner.invoke(run, ["example.py", "--clean-failed"], input="y\n")
        assert "The following 1 trials will be deleted:" in result.output
        assert "Continue? [y/N]: y\n" in result.output
        assert result.exit_code == 0

        result = runner.invoke(collect, ["example.py"], catch_exceptions=False)
        assert result.exit_code == 0

        assert os.path.isfile("example.csv")

        with open("example.csv") as f:
            print(f.read())

        result = runner.invoke(clean, ["example.py"], catch_exceptions=False)
        assert result.exit_code == 0


def test_do():
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("example.py", "w") as f:
            f.write(example_py)

        result = runner.invoke(run, ["example.py"], catch_exceptions=False)
        assert result.exit_code == 0

        # Assert that trial command succeedes
        result = runner.invoke(do, ["example.py", "cmd", "e3/_"])
        print(result.output)
        assert result.exit_code == 0

        # Assert that experiment command succeedes
        result = runner.invoke(do, ["example.py", "experiment_cmd", "e3"])
        print(result.output)
        assert result.exit_code == 0

        # Assert that inexistent command fails
        result = runner.invoke(
            do, ["example.py", "inexistent_cmd", "e3/_"], catch_exceptions=False
        )
        print(result.output)

        # Fail with click.UsageError
        assert result.exit_code == 2

        # Assert that inexistent trial fails
        result = runner.invoke(
            do, ["example.py", "cmd", "e3/inexistent_trial"], catch_exceptions=False
        )
        print(result.output)

        # Fail with click.UsageError
        assert result.exit_code == 2
