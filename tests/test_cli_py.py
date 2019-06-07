import inspect
import os.path

from click.testing import CliRunner

from experitur.cli import collect, run

example_py = inspect.cleandoc("""
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
    """)


def test_run():
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open('example.py', 'w') as f:
            f.write(example_py)

        result = runner.invoke(run, ['example.py'])
        assert result.exit_code == 0

        result = runner.invoke(collect, ['example.py'], catch_exceptions=False)
        assert result.exit_code == 0

        assert os.path.isfile("example.csv")

        with open('example.csv') as f:
            print(f.read())
