from click.testing import CliRunner
import pytest

from sever.cli import cli


@pytest.fixture()
def runner():
    return CliRunner()

def test_cli_template(runner):
    result = runner.invoke(cli)
    assert result.exit_code == 0