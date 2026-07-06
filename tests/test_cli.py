from typer.testing import CliRunner
from rabbit_hunter.cli import app

runner = CliRunner()


def test_cli_help():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ("fetch", "features", "backtest", "data"):
        assert cmd in r.stdout


def test_backtest_dry_run(tmp_path):
    # dry-run 不落文件、不拉数据，只走 config 加载
    r = runner.invoke(app, ["backtest", "--config", "configs/default.yaml", "--dry-run"])
    assert r.exit_code == 0
    assert "dry-run" in r.stdout.lower()
