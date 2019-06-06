from experitur.cli import run
from click.testing import CliRunner
import inspect

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
