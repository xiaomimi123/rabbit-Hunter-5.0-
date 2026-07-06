from __future__ import annotations


def daily_loss_tripped(
    daily_realized_pnl: float,
    initial_capital: float,
    daily_max_loss_pct: float,
) -> bool:
    max_loss = initial_capital * (daily_max_loss_pct / 100.0)
    return daily_realized_pnl <= -max_loss
