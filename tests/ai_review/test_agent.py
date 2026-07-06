import json
import pandas as pd
import pytest
from rabbit_hunter.ai_review.agent import (
    ReportBundle,
    load_report_bundle,
    build_review_prompt,
)


def _mk_trades(n=10, winrate=0.4):
    """Synthesize a trades dataframe with the columns the agent reads."""
    return pd.DataFrame([
        {
            "symbol": "BTC-USDT-SWAP" if i % 2 == 0 else "ETH-USDT-SWAP",
            "side": "long" if i % 3 == 0 else "short",
            "pnl_after_fees": 10.0 if (i * 7 % 10) < (winrate * 10) else -8.0,
            "exit_reason": "take_profit" if i % 3 == 0 else "stop_loss",
            "regime_t0": "trending" if i % 2 == 0 else "ranging",
            "strategy_scores": json.dumps(
                {"trend_following": 0.6, "price_action": 0.3} if i % 2 == 0
                else {"trend_following": 0.2, "price_action": 0.7}
            ),
        }
        for i in range(n)
    ])


def test_load_report_bundle_from_dir(tmp_path):
    """Full-shape bundle load with all files present."""
    d = tmp_path / "2026-07-06-1000"
    d.mkdir()
    (d / "report.md").write_text("# Report", encoding="utf-8")
    (d / "ai_context.md").write_text("# AI Context\n\n## Baselines\n...", encoding="utf-8")
    (d / "config_snapshot.yaml").write_text("data: {symbols: [BTC]}", encoding="utf-8")
    _mk_trades().to_parquet(d / "trades.parquet", index=False)
    pd.DataFrame({"action": ["wait"], "timestamp": [0]}).to_parquet(d / "snapshots.parquet", index=False)

    bundle = load_report_bundle(d)
    assert bundle.run_id == "2026-07-06-1000"
    assert "# Report" in bundle.report_md
    assert "AI Context" in bundle.ai_context_md
    assert "symbols" in bundle.config_yaml
    assert bundle.trade_count() == 10
    assert len(bundle.snapshots_df) == 1


def test_load_report_bundle_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_report_bundle(tmp_path / "does-not-exist")


def test_load_report_bundle_gracefully_handles_missing_files(tmp_path):
    """Not all reports have every file (empty backtest, etc)."""
    d = tmp_path / "2026-07-06-1000"
    d.mkdir()
    # Only ai_context.md exists
    (d / "ai_context.md").write_text("# empty", encoding="utf-8")

    bundle = load_report_bundle(d)
    assert bundle.report_md == ""
    assert "empty" in bundle.ai_context_md
    assert bundle.config_yaml == ""
    assert bundle.trade_count() == 0
    assert bundle.snapshots_df.empty


def test_build_review_prompt_contains_key_sections(tmp_path):
    d = tmp_path / "2026-07-06-1000"
    d.mkdir()
    (d / "report.md").write_text("# Report", encoding="utf-8")
    (d / "ai_context.md").write_text("# AI Context body", encoding="utf-8")
    (d / "config_snapshot.yaml").write_text("data:\n  symbols: [BTC]", encoding="utf-8")
    _mk_trades().to_parquet(d / "trades.parquet", index=False)
    pd.DataFrame().to_parquet(d / "snapshots.parquet", index=False)

    bundle = load_report_bundle(d)
    prompt = build_review_prompt(bundle)

    # Framing: LLM is analyst, not trader
    assert "analyst" in prompt.lower()
    assert "NOT trader" in prompt or "not trader" in prompt.lower()

    # Sections the AI needs
    assert "Per-strategy breakdown" in prompt
    assert "Per-symbol breakdown" in prompt
    assert "Per-regime breakdown" in prompt
    assert "Exit-reason distribution" in prompt

    # Body of ai_context + config
    assert "AI Context body" in prompt
    assert "symbols: [BTC]" in prompt

    # Response schema
    assert "top_findings" in prompt
    assert "parameter_change_suggestions" in prompt
    assert "confidence" in prompt


def test_prompt_truncates_huge_ai_context(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    huge = "x" * 20_000
    (d / "ai_context.md").write_text(huge, encoding="utf-8")
    (d / "config_snapshot.yaml").write_text("data: {}", encoding="utf-8")
    pd.DataFrame().to_parquet(d / "trades.parquet", index=False)
    pd.DataFrame().to_parquet(d / "snapshots.parquet", index=False)

    bundle = load_report_bundle(d)
    prompt = build_review_prompt(bundle, max_ai_context_chars=1000)
    # Should have truncation marker
    assert "truncated" in prompt
    # Overall prompt should be far less than 20_000 chars for the AI context section
    assert prompt.count("x") < 2000
