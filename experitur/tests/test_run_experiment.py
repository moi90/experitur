from experitur.cli import run
from click.testing import CliRunner

example = \
    """---
- id: example
---
# Example experiment
"""


def test_run():
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open('example.md', 'w') as f:
            f.write(example)

        result = runner.invoke(run, ['example.md'])
        assert result.exit_code == 0
