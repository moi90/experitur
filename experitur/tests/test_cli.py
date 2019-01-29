from experitur.cli import run
from click.testing import CliRunner
import inspect

example = inspect.cleandoc("""
    ---
    -
        id: example
        run: experitur.tests.test_cli:noop
    ---
    # Example experiment
    """)


def noop(wdir, parameters):
    return parameters


def test_run():
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open('example.md', 'w') as f:
            f.write(example)

        result = runner.invoke(run, ['example.md'])
        assert result.exit_code == 0
