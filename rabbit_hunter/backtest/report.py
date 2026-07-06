from __future__ import annotations
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape
import yaml
from scipy import stats as sci_stats

from rabbit_hunter.config.schema import AppConfig
from .engine import BacktestResult


_TEMPLATE_DIR = Path(__file__).parent / "templates"


def compute_stats(trades: list[dict], equity_curve: pd.DataFrame, initial_capital: float) -> dict:
    if not trades:
        return {"total_return_pct": 0.0, "annualized_return_pct": 0.0, "max_drawdown_pct": 0.0,
                "sharpe": 0.0, "trade_count": 0, "win_rate_pct": 0.0, "profit_factor": 0.0}
    df = pd.DataFrame(trades)
    total_pnl = df["pnl_after_fees"].sum()
    total_return = total_pnl / initial_capital
    wins = df[df["pnl_after_fees"] > 0]["pnl_after_fees"].sum()
    losses = df[df["pnl_after_fees"] < 0]["pnl_after_fees"].sum()
    win_rate = (df["pnl_after_fees"] > 0).mean() if len(df) else 0.0
    pf = (wins / -losses) if losses < 0 else float("inf")

    # 年化 & 夏普基于 equity_curve
    if not equity_curve.empty:
        eq = equity_curve["equity"].to_numpy()
        ret = np.diff(eq) / eq[:-1]
        ret = ret[np.isfinite(ret)]
        sharpe = float(np.sqrt(24 * 365) * ret.mean() / ret.std()) if len(ret) > 1 and ret.std() > 0 else 0.0
        peak = np.maximum.accumulate(eq)
        drawdowns = (eq - peak) / peak
        max_dd = float(-drawdowns.min())
        days = (equity_curve["timestamp"].iloc[-1] - equity_curve["timestamp"].iloc[0]) / 86_400_000 or 1
        annualized = (1 + total_return) ** (365.0 / days) - 1
    else:
        sharpe = 0.0; max_dd = 0.0; annualized = 0.0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(annualized * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 2),
        "trade_count": len(df),
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": round(pf, 2) if pf != float("inf") else 999.0,
    }


def find_loss_clusters(trades_df: pd.DataFrame, min_trades: int = 20, max_winrate: float = 0.4) -> list[dict]:
    if trades_df.empty:
        return []
    dims_all = ["regime", "session", "day_of_week"]
    dims = [d for d in dims_all if d in trades_df.columns]
    clusters: list[dict] = []
    for r in (1, 2):
        for combo in itertools.combinations(dims, r):
            grouped = trades_df.groupby(list(combo), dropna=False).agg(
                trades=("pnl_after_fees", "count"),
                winrate=("pnl_after_fees", lambda x: (x > 0).mean()),
                total_pnl=("pnl_after_fees", "sum"),
            ).reset_index()
            hits = grouped[(grouped["trades"] >= min_trades) & (grouped["winrate"] <= max_winrate)]
            for _, row in hits.iterrows():
                dim_str = " AND ".join(f"{c}={row[c]}" for c in combo)
                clusters.append({
                    "dim": dim_str,
                    "trades": int(row["trades"]),
                    "winrate_pct": round(row["winrate"] * 100, 1),
                    "total_pnl": round(row["total_pnl"], 2),
                })
    return sorted(clusters, key=lambda x: x["total_pnl"])[:10]


def _flatten_trade_row(t: dict) -> dict:
    row = {k: v for k, v in t.items() if k not in ("entry_snapshot", "exit_snapshot", "strategy_scores")}
    for k, v in (t.get("entry_snapshot") or {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            row[f"{k}_t0"] = v
    for k, v in (t.get("exit_snapshot") or {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            row[f"{k}_texit"] = v
    row["strategy_scores"] = json.dumps(t.get("strategy_scores", {}), default=str)
    row["entry_snapshot_json"] = json.dumps(t.get("entry_snapshot", {}), default=str)
    row["exit_snapshot_json"] = json.dumps(t.get("exit_snapshot", {}), default=str)
    return row


def _regime_conditional_performance(trades_df: pd.DataFrame) -> list[dict]:
    if trades_df.empty or "regime_t0" not in trades_df.columns:
        return []
    grouped = trades_df.groupby("regime_t0", dropna=False).agg(
        count=("pnl_after_fees", "count"),
        total_pnl=("pnl_after_fees", "sum"),
        winrate=("pnl_after_fees", lambda x: (x > 0).mean()),
        avg_pnl=("pnl_after_fees", "mean"),
    ).reset_index()
    return [
        {"regime": row["regime_t0"], "count": int(row["count"]),
         "return_pct": round(row["total_pnl"] / 10_000 * 100, 2),
         "win_rate_pct": round(row["winrate"] * 100, 1),
         "avg_pnl": round(row["avg_pnl"], 2)}
        for _, row in grouped.iterrows()
    ]


def _feature_correlation(trades_df: pd.DataFrame, top_n: int = 10) -> list[dict]:
    if trades_df.empty or "pnl_after_fees" not in trades_df.columns:
        return []
    numeric_cols = [c for c in trades_df.columns if c.endswith("_t0")
                    and pd.api.types.is_numeric_dtype(trades_df[c])]
    results: list[dict] = []
    for c in numeric_cols:
        try:
            rho, _ = sci_stats.spearmanr(trades_df[c], trades_df["pnl_after_fees"], nan_policy="omit")
            if pd.notna(rho):
                results.append({"feature": c, "rho": round(float(rho), 3)})
        except Exception:
            continue
    return sorted(results, key=lambda x: -abs(x["rho"]))[:top_n]


def _compute_baselines(features_by_symbol: dict[str, pd.DataFrame], initial_capital: float) -> list[dict]:
    baselines: list[dict] = []
    for symbol, feats in features_by_symbol.items():
        if len(feats) < 2:
            continue
        first = float(feats["close"].iloc[0])
        last = float(feats["close"].iloc[-1])
        ret = (last - first) / first
        # 简单夏普：小时收益
        rets = feats["close"].pct_change().dropna().to_numpy()
        sharpe = float(np.sqrt(24 * 365) * rets.mean() / rets.std()) if len(rets) > 1 and rets.std() > 0 else 0.0
        peak = np.maximum.accumulate(feats["close"].to_numpy())
        dd = (feats["close"].to_numpy() - peak) / peak
        max_dd = float(-dd.min())
        baselines.append({
            "name": f"Buy-and-Hold {symbol}",
            "total_return_pct": round(ret * 100, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
        })
    return baselines


class ReportBuilder:
    def __init__(self, cfg: AppConfig, features_by_symbol: dict[str, pd.DataFrame]):
        self.cfg = cfg
        self.features_by_symbol = features_by_symbol
        self.env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=select_autoescape())

    def build(self, result: BacktestResult, output_root: Path, git_commit: str = "unknown") -> Path:
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")
        out = Path(output_root) / run_id
        (out / "charts").mkdir(parents=True, exist_ok=True)

        trades = result.ledger.closed_trades
        stats = compute_stats(trades, result.equity_curve, self.cfg.backtest.initial_capital)

        # trades.parquet
        if trades:
            trades_df = pd.DataFrame([_flatten_trade_row(t) for t in trades])
        else:
            trades_df = pd.DataFrame(columns=["symbol", "side", "entry_time", "exit_time",
                                              "entry_price", "exit_price", "size",
                                              "pnl_raw", "pnl_after_fees", "fees", "funding",
                                              "slippage", "hold_bars", "exit_reason"])
        trades_df.to_parquet(out / "trades.parquet", index=False)

        # snapshots.parquet
        snap_df = result.snapshots.copy()
        if not snap_df.empty and "long_score" in snap_df.columns:
            snap_df["long_score"] = snap_df["long_score"].apply(lambda x: json.dumps(x, default=str))
        snap_df.to_parquet(out / "snapshots.parquet", index=False)

        # config_snapshot.yaml
        cfg_dict = self.cfg.model_dump()
        cfg_yaml = yaml.safe_dump(cfg_dict, allow_unicode=True, sort_keys=False)
        (out / "config_snapshot.yaml").write_text(cfg_yaml, encoding="utf-8")
        cfg_hash = hashlib.sha256(cfg_yaml.encode("utf-8")).hexdigest()[:12]

        # charts
        self._plot_equity_curve(result.equity_curve, out / "charts" / "equity_curve.png")
        self._plot_monthly_pnl(trades_df, out / "charts" / "monthly_pnl.png")

        # 报告数据
        per_symbol = self._per_symbol_stats(trades_df)
        per_strategy = self._per_strategy_stats(trades_df, snap_df)
        worst_trades = self._worst_trades(trades_df, n=10)
        baselines = _compute_baselines(self.features_by_symbol, self.cfg.backtest.initial_capital)
        regime_perf = _regime_conditional_performance(trades_df)
        clusters = find_loss_clusters(trades_df) if not trades_df.empty else []
        feature_corr = _feature_correlation(trades_df)

        report_md = self.env.get_template("report.md.j2").render(
            run_id=run_id,
            start=self.cfg.backtest.start, end=self.cfg.backtest.end,
            symbols=self.cfg.data.symbols,
            main_interval=self.cfg.data.main_interval,
            confirm_interval=self.cfg.data.confirm_interval,
            strategies=list(self.cfg.strategy_router.enabled_strategies.keys()),
            config_hash=cfg_hash,
            git_commit=git_commit,
            stats=stats,
            per_symbol=per_symbol,
            per_strategy=per_strategy,
            worst_trades=worst_trades,
            snapshot_count=len(snap_df),
            config_yaml=cfg_yaml,
        )
        (out / "report.md").write_text(report_md, encoding="utf-8")

        ai_ctx = self.env.get_template("ai_context.md.j2").render(
            run_id=run_id,
            start=self.cfg.backtest.start, end=self.cfg.backtest.end,
            feature_engine_version=self.cfg.feature_engine.version,
            fees=self.cfg.execution.fees.model_dump(),
            slippage_atr_multiplier=self.cfg.execution.slippage_atr_multiplier,
            baselines=baselines,
            stats=stats,
            clusters=clusters,
            regime_perf=regime_perf,
            feature_corr=feature_corr,
        )
        (out / "ai_context.md").write_text(ai_ctx, encoding="utf-8")

        return out

    def _per_symbol_stats(self, trades_df: pd.DataFrame) -> list[dict]:
        if trades_df.empty:
            return []
        rows = []
        for sym, grp in trades_df.groupby("symbol"):
            rows.append({
                "symbol": sym,
                "return_pct": round(grp["pnl_after_fees"].sum() / self.cfg.backtest.initial_capital * 100, 2),
                "win_rate_pct": round((grp["pnl_after_fees"] > 0).mean() * 100, 1),
                "count": len(grp),
            })
        return rows

    def _per_strategy_stats(self, trades_df: pd.DataFrame, snap_df: pd.DataFrame) -> list[dict]:
        strategies = list(self.cfg.strategy_router.enabled_strategies.keys())
        rows = []
        for s in strategies:
            trig = 0
            if not snap_df.empty and "long_score" in snap_df.columns:
                for js in snap_df["long_score"].dropna():
                    try:
                        d = json.loads(js) if isinstance(js, str) else js
                        if s in d:
                            trig += 1
                    except Exception:
                        continue
            rows.append({"strategy": s, "avg_long": "-", "trigger_count": trig})
        return rows

    def _worst_trades(self, trades_df: pd.DataFrame, n: int = 10) -> list[dict]:
        if trades_df.empty:
            return []
        worst = trades_df.nsmallest(n, "pnl_after_fees")
        rows = []
        for _, row in worst.iterrows():
            rows.append({
                "entry_time": datetime.fromtimestamp(row["entry_time"] / 1000, tz=timezone.utc).isoformat(),
                "symbol": row["symbol"], "side": row["side"],
                "entry_price": round(row["entry_price"], 2),
                "exit_price": round(row["exit_price"], 2),
                "pnl_after_fees": round(row["pnl_after_fees"], 2),
                "exit_reason": row["exit_reason"],
            })
        return rows

    def _plot_equity_curve(self, eq: pd.DataFrame, path: Path):
        fig, ax = plt.subplots(figsize=(10, 4))
        if not eq.empty:
            ax.plot(pd.to_datetime(eq["timestamp"], unit="ms", utc=True), eq["equity"])
        ax.set_title("Equity Curve")
        ax.set_xlabel("Time (UTC)"); ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)

    def _plot_monthly_pnl(self, trades_df: pd.DataFrame, path: Path):
        fig, ax = plt.subplots(figsize=(10, 4))
        if not trades_df.empty:
            trades_df = trades_df.copy()
            trades_df["month"] = pd.to_datetime(trades_df["exit_time"], unit="ms", utc=True).dt.to_period("M").astype(str)
            monthly = trades_df.groupby("month")["pnl_after_fees"].sum()
            ax.bar(monthly.index, monthly.values)
        ax.set_title("Monthly PnL")
        ax.set_xlabel("Month"); ax.set_ylabel("PnL (USDT)")
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
