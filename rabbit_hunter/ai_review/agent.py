"""AI Review Agent — build LLM prompts from backtest reports.

Strictly consumes-only; does NOT touch strategy code or config. The
architecture spec (§ 3.5) requires that any AI suggestion goes through:
  1. Sample-out validation on a held-out window
  2. Shadow-mode parallel run
  3. Human approval

This module produces the INPUT to that process — a prompt an LLM can
respond to with structured suggestions. Consuming those suggestions is
a manual workflow.
"""
from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ReportBundle:
    """All the pieces the AI Review needs to reason about a backtest."""
    run_id: str
    report_dir: Path
    report_md: str
    ai_context_md: str
    config_yaml: str
    trades_df: pd.DataFrame
    snapshots_df: pd.DataFrame

    def trade_count(self) -> int:
        return len(self.trades_df)


def load_report_bundle(report_dir: Path | str) -> ReportBundle:
    """Load a full backtest report bundle from disk."""
    d = Path(report_dir)
    if not d.exists():
        raise FileNotFoundError(f"Report dir does not exist: {d}")

    report_md = (d / "report.md").read_text(encoding="utf-8") if (d / "report.md").exists() else ""
    ai_context = (d / "ai_context.md").read_text(encoding="utf-8") if (d / "ai_context.md").exists() else ""
    cfg_yaml = (d / "config_snapshot.yaml").read_text(encoding="utf-8") if (d / "config_snapshot.yaml").exists() else ""
    trades = pd.read_parquet(d / "trades.parquet") if (d / "trades.parquet").exists() else pd.DataFrame()
    snapshots = pd.read_parquet(d / "snapshots.parquet") if (d / "snapshots.parquet").exists() else pd.DataFrame()

    return ReportBundle(
        run_id=d.name,
        report_dir=d,
        report_md=report_md,
        ai_context_md=ai_context,
        config_yaml=cfg_yaml,
        trades_df=trades,
        snapshots_df=snapshots,
    )


def _summarize_by_strategy(trades: pd.DataFrame) -> str:
    """A short table of per-strategy performance for the prompt."""
    if trades.empty or "strategy_scores" not in trades.columns:
        return "(no trades to break down)"

    def _dominant(js):
        try:
            d = json.loads(js) if isinstance(js, str) else js
            return max(d, key=d.get) if d else "unknown"
        except Exception:
            return "unknown"

    trades = trades.copy()
    trades["dominant"] = trades["strategy_scores"].apply(_dominant)
    grouped = trades.groupby("dominant").agg(
        n=("pnl_after_fees", "count"),
        winrate=("pnl_after_fees", lambda x: (x > 0).mean()),
        total_pnl=("pnl_after_fees", "sum"),
        avg_pnl=("pnl_after_fees", "mean"),
    ).round(2)
    return grouped.to_string()


def _summarize_by_regime(trades: pd.DataFrame) -> str:
    """Per-regime performance from the _t0-suffixed regime column."""
    if trades.empty or "regime_t0" not in trades.columns:
        return "(no regime column in trades)"
    grouped = trades.groupby("regime_t0", dropna=False).agg(
        n=("pnl_after_fees", "count"),
        winrate=("pnl_after_fees", lambda x: (x > 0).mean()),
        total_pnl=("pnl_after_fees", "sum"),
    ).round(2)
    return grouped.to_string()


def _summarize_by_symbol(trades: pd.DataFrame) -> str:
    """Per-symbol breakdown (critical for multi-symbol backtests)."""
    if trades.empty or "symbol" not in trades.columns:
        return "(no symbol column)"
    grouped = trades.groupby("symbol").agg(
        n=("pnl_after_fees", "count"),
        winrate=("pnl_after_fees", lambda x: (x > 0).mean()),
        total_pnl=("pnl_after_fees", "sum"),
        avg_pnl=("pnl_after_fees", "mean"),
    ).round(2).sort_values("total_pnl", ascending=False)
    return grouped.to_string()


def _summarize_exit_reasons(trades: pd.DataFrame) -> str:
    if trades.empty or "exit_reason" not in trades.columns:
        return "(no exit_reason)"
    return trades["exit_reason"].value_counts().to_string()


def build_review_prompt(
    bundle: ReportBundle,
    max_ai_context_chars: int = 6000,
    max_config_chars: int = 2500,
) -> str:
    """Compose a single-shot LLM prompt that asks for structured suggestions.

    Prompt structure:
      1. Framing: who the LLM is (analyst, NOT trader), what it can/cannot suggest
      2. Numeric summary tables (compact)
      3. Full ai_context.md and config_snapshot.yaml
      4. Structured response schema request

    The LLM's job is to return concrete, testable hypotheses — NOT to
    directly change the strategy. That step is human-only.
    """
    strategy_summary = _summarize_by_strategy(bundle.trades_df)
    symbol_summary = _summarize_by_symbol(bundle.trades_df)
    regime_summary = _summarize_by_regime(bundle.trades_df)
    exit_summary = _summarize_exit_reasons(bundle.trades_df)

    # Truncate potentially huge inputs
    ai_ctx = bundle.ai_context_md
    if len(ai_ctx) > max_ai_context_chars:
        ai_ctx = ai_ctx[:max_ai_context_chars] + "\n… (truncated)"
    cfg = bundle.config_yaml
    if len(cfg) > max_config_chars:
        cfg = cfg[:max_config_chars] + "\n… (truncated)"

    prompt = f"""\
You are the AI Review Agent for a rule-based crypto perpetual quant engine
(Rabbit Hunter V5.1). Your role is analyst, NOT trader. You produce
suggestions — you never directly change strategy code or parameters.

Any suggestion you make will go through:
  1. Backtest validation on the reported window
  2. Out-of-sample validation on a held-out window
  3. Shadow-mode parallel run
  4. Human approval before touching production

# Backtest identity

Run: {bundle.run_id}
Total trades: {bundle.trade_count()}

# Per-strategy breakdown

{strategy_summary}

# Per-symbol breakdown

{symbol_summary}

# Per-regime breakdown

{regime_summary}

# Exit-reason distribution

{exit_summary}

# Full ai_context.md (source of truth on baselines / failure clusters / regime perf / feature correlations)

{ai_ctx}

# Config in effect for this run

```yaml
{cfg}
```

# Your task

Return a JSON object with EXACTLY these keys. Nothing else. No markdown.
Use plain quotes.

{{
  "top_findings": [
    // 3-5 concrete numeric observations. Each is a short sentence with numbers.
  ],
  "hypotheses": [
    // 2-4 falsifiable hypotheses about WHY the strategy behaved this way.
    // Each is a testable claim that a follow-up backtest could refute.
  ],
  "parameter_change_suggestions": [
    // Each suggestion is a dict with keys:
    // - "file": e.g. "configs/strategies/trend_following.yaml"
    // - "field": e.g. "adx_threshold"
    // - "current": current value
    // - "suggested": new value
    // - "rationale": short reason grounded in the data above
    // - "risk": what could go wrong if this is wrong
    // Only propose changes that are CONFIG-only. Never propose code changes.
  ],
  "regime_specific_insights": [
    // 1-3 observations tying performance to specific regimes/sessions/days.
  ],
  "failure_mode_analysis": {{
    // Which cluster of trades hurts the most? What's the pattern?
    "worst_cluster": "description",
    "root_cause_hypothesis": "your best guess",
    "would_disprove_it_by": "what data/test would refute this"
  }},
  "confidence": "low" | "medium" | "high",
  "confidence_reasoning": "short reason for confidence rating"
}}
"""
    return textwrap.dedent(prompt)
