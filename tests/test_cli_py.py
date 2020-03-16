import inspect
import os.path

from click.testing import CliRunner

from experitur.cli import collect, run, do, clean

example_py = inspect.cleandoc(
    """
    from experitur import experiment, run

    @experiment(
        parameter_grid={
            "a1": [1],
            "a2": [2],
            "b": [1, 2],
            "a": ["{a_{b}}"],
        })
    def baseline(trial):
        pass

    @experiment(parent=baseline)
    def e1(trial):
        pass

    @experiment()
    def e2(trial):
        raise NotImplementedError()

    @experiment()
    def e3(trial):
        pass

    @e3.command("cmd")
    def e3_cmd(trial):
        pass
    """
)


def test_run():
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("example.py", "w") as f:
            f.write(example_py)

        result = runner.invoke(run, ["example.py"])
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

        result = runner.invoke(do, ["example.py", "cmd", "e3/_"])
        print(result.output)
        assert result.exit_code == 0
