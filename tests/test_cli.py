from experitur.cli import run
from click.testing import CliRunner
import inspect

example_md = inspect.cleandoc("""
    ---
    -
        id: example
        run: example:noop
    ---
    # Example experiment
    """)

example_py = inspect.cleandoc("""
    def noop(wdir, parameters):
        return parameters
    """)


def test_run():
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open('example.md', 'w') as f:
            f.write(example_md)

        with open('example.py', 'w') as f:
            f.write(example_py)

        result = runner.invoke(run, ['example.md'])
        assert result.exit_code == 0
