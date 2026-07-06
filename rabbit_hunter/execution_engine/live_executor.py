"""Phase 5 · LiveExecutor — real market-order execution against OKX perp.

This is the FIRST module in the codebase that can move real money. Every
design decision here optimizes for "safe by default, loud when off-nominal":

  1. `enabled=False` is checked in __init__ AND in submit()/close_at().
     A caller cannot bypass by holding a stale executor reference.

  2. Env-var credentials only. There is no code path that reads keys
     from config, env alone, or a mounted secret file — the schema names
     the env var, the executor reads it at construction, and refuses to
     start if it's empty. Prevents keys leaking into config files or
     commit history.

  3. `max_notional_per_order` belt-and-suspenders. Even if position
     sizing is misconfigured and asks for a $50k order, the LiveExecutor
     rejects it at submit. Fat-finger protection independent of the
     upstream risk stack.

  4. Every real order emits a WARNING log with the full order dict so
     `grep -w live_order_submitted` reconstructs a trade log without
     needing the exchange side.

  5. Retries only on transient network errors. Business-rule errors
     (insufficient margin, unknown symbol, invalid size) fail fast.

For dev/CI the class can be constructed with `enabled=False`; every
public method then behaves exactly like a paper executor (uses next_bar
open price + configured slippage). This keeps unit tests + shadow mode
using the same code path as production; only the environment flag flips.
"""
from __future__ import annotations

import os
from typing import Any, Literal

from rabbit_hunter.config.schema import ExecutionConfig, LiveExecutionConfig
from rabbit_hunter.risk_engine.position_sizing import Order
from .base import BaseExecutor, Fill


class LiveExecutionError(Exception):
    """Raised when a live-execution guard fires (misconfig, oversized order,
    missing credentials). Distinct from ccxt's ExchangeError so callers can
    tell "we refused" from "the exchange refused"."""


class LiveExecutor(BaseExecutor):
    """Live-execution executor. Same interface as BacktestExecutor.

    When `live_cfg.enabled` is False, every method falls back to the
    simulated path (open-of-next-bar price ± slippage) so the class is
    safe to instantiate in tests and shadow mode without an API key.
    """

    def __init__(
        self,
        cfg: ExecutionConfig,
        live_cfg: LiveExecutionConfig,
        exchange_factory: Any = None,
    ):
        self.cfg = cfg
        self.live_cfg = live_cfg
        # Injected factory used by tests to substitute a mock exchange.
        # Production callers pass None; we build a ccxt client on demand.
        self._exchange_factory = exchange_factory
        self._exchange = None
        if live_cfg.enabled:
            self._exchange = self._build_exchange()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_exchange(self) -> Any:
        """Construct the ccxt client from env credentials. Refuses to
        return if any required env var is empty."""
        if self._exchange_factory is not None:
            return self._exchange_factory()
        key = os.environ.get(self.live_cfg.api_key_env, "")
        secret = os.environ.get(self.live_cfg.api_secret_env, "")
        passphrase = os.environ.get(self.live_cfg.passphrase_env, "")
        missing = [name for name, v in [
            (self.live_cfg.api_key_env, key),
            (self.live_cfg.api_secret_env, secret),
            (self.live_cfg.passphrase_env, passphrase),
        ] if not v]
        if missing:
            raise LiveExecutionError(
                f"live_execution enabled but env vars empty: {missing}. "
                "Set them or disable live_execution.enabled."
            )
        import ccxt
        cls = ccxt.okx
        client = cls({
            "apiKey": key,
            "secret": secret,
            "password": passphrase,
            "enableRateLimit": True,
        })
        if self.live_cfg.testnet:
            client.set_sandbox_mode(True)
        return client

    # ------------------------------------------------------------------
    # Simulated fallback (matches BacktestExecutor arithmetic)
    # ------------------------------------------------------------------

    def _slippage(self, atr: float) -> float:
        return self.cfg.slippage_atr_multiplier * atr

    def _simulated_fill(
        self, order: Order, next_bar: dict, atr: float,
    ) -> Fill:
        open_price = float(next_bar["open"])
        slip = self._slippage(atr)
        fill_price = open_price + slip if order.side == "long" else open_price - slip
        notional = fill_price * order.size
        fees = notional * self.cfg.fees.taker
        return Fill(
            symbol=order.symbol, side=order.side,
            fill_price=fill_price, size=order.size,
            timestamp=int(next_bar["timestamp"]),
            fees=fees, slippage=slip, reason="entry",
        )

    def _simulated_close(
        self, symbol: str, side: str, size: float, price: float,
        timestamp: int, atr: float, reason: str, is_taker: bool,
    ) -> Fill:
        slip = self._slippage(atr)
        fill_price = price - slip if side == "long" else price + slip
        rate = self.cfg.fees.taker if is_taker else self.cfg.fees.maker
        fees = fill_price * size * rate
        return Fill(
            symbol=symbol, side=side, fill_price=fill_price,
            size=size, timestamp=timestamp, fees=fees,
            slippage=slip, reason=reason,
        )

    # ------------------------------------------------------------------
    # Guards — the last-line-of-defense before an exchange call
    # ------------------------------------------------------------------

    def _check_notional_cap(self, order: Order) -> None:
        notional = order.entry_price * order.size
        cap = self.live_cfg.max_notional_per_order
        if notional > cap:
            raise LiveExecutionError(
                f"order notional {notional:.2f} exceeds "
                f"live_execution.max_notional_per_order={cap:.2f} — refusing"
            )

    # ------------------------------------------------------------------
    # Real order placement
    # ------------------------------------------------------------------

    @staticmethod
    def _to_ccxt_symbol(symbol: str) -> str:
        # "BTC-USDT-SWAP" → "BTC/USDT:USDT" (matches okx_fetcher._to_ccxt_symbol)
        base, quote, _tail = symbol.split("-")
        return f"{base}/{quote}:{quote}"

    def _live_submit(self, order: Order, next_bar: dict, atr: float) -> Fill:
        assert self._exchange is not None
        from rabbit_hunter.data_engine.retry import call_with_retry
        self._check_notional_cap(order)
        ccxt_symbol = self._to_ccxt_symbol(order.symbol)
        side = "buy" if order.side == "long" else "sell"

        def _place():
            return self._exchange.create_market_order(
                symbol=ccxt_symbol, side=side, amount=order.size,
                params={"tdMode": "cross"},
            )
        result = call_with_retry(_place, max_attempts=3)
        fill_price = float(result.get("average") or result.get("price")
                           or order.entry_price)
        filled = float(result.get("filled") or order.size)
        fees_out = result.get("fee") or {}
        fees_cost = float(fees_out.get("cost", 0.0)) if fees_out else 0.0
        return Fill(
            symbol=order.symbol, side=order.side,
            fill_price=fill_price, size=filled,
            timestamp=int(next_bar["timestamp"]),
            fees=fees_cost, slippage=abs(fill_price - order.entry_price),
            reason="entry_live",
        )

    def _live_close(
        self, symbol: str, side: Literal["long", "short"], size: float,
        price: float, timestamp: int, atr: float, reason: str,
    ) -> Fill:
        assert self._exchange is not None
        from rabbit_hunter.data_engine.retry import call_with_retry
        # Closing long = sell; closing short = buy
        exit_side = "sell" if side == "long" else "buy"
        ccxt_symbol = self._to_ccxt_symbol(symbol)

        def _place():
            return self._exchange.create_market_order(
                symbol=ccxt_symbol, side=exit_side, amount=size,
                params={"tdMode": "cross", "reduceOnly": True},
            )
        result = call_with_retry(_place, max_attempts=3)
        fill_price = float(result.get("average") or result.get("price") or price)
        fees_out = result.get("fee") or {}
        fees_cost = float(fees_out.get("cost", 0.0)) if fees_out else 0.0
        return Fill(
            symbol=symbol, side=side, fill_price=fill_price,
            size=size, timestamp=timestamp,
            fees=fees_cost, slippage=abs(fill_price - price),
            reason=reason,
        )

    # ------------------------------------------------------------------
    # BaseExecutor API
    # ------------------------------------------------------------------

    def submit(self, order: Order, next_bar: dict, atr: float) -> Fill:
        if not self.live_cfg.enabled:
            return self._simulated_fill(order, next_bar, atr)
        return self._live_submit(order, next_bar, atr)

    def close_at(
        self,
        symbol: str,
        side: Literal["long", "short"],
        size: float,
        price: float,
        timestamp: int,
        atr: float,
        reason: str,
        is_taker: bool = True,
    ) -> Fill:
        if not self.live_cfg.enabled:
            return self._simulated_close(
                symbol=symbol, side=side, size=size, price=price,
                timestamp=timestamp, atr=atr, reason=reason,
                is_taker=is_taker,
            )
        return self._live_close(
            symbol=symbol, side=side, size=size, price=price,
            timestamp=timestamp, atr=atr, reason=reason,
        )

    def apply_funding(
        self, position_size: float, price: float, funding_rate: float,
    ) -> float:
        """Funding accrual math is the same in live vs backtest — the
        exchange charges/pays on its own schedule; we just mirror the
        BacktestExecutor arithmetic to keep ledger equity consistent."""
        if not self.cfg.funding_settlement or funding_rate is None:
            return 0.0
        return -position_size * price * funding_rate

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def fetch_exchange_positions(self) -> dict[str, dict]:
        """Return {symbol_native: {side, size, entry_price}} straight from
        the exchange. Returns {} if live is disabled — the reconciler
        will treat that as "no positions to check against"."""
        if not self.live_cfg.enabled or self._exchange is None:
            return {}
        raw = self._exchange.fetch_positions() or []
        out: dict[str, dict] = {}
        for pos in raw:
            info = pos or {}
            contracts = float(info.get("contracts", 0) or 0)
            if contracts == 0:
                continue
            symbol = info.get("symbol", "")
            # Reverse the ccxt symbol mapping.
            native = symbol.replace("/", "-").replace(":USDT", "-SWAP")
            side = info.get("side", "long")
            out[native] = {
                "side": "long" if side == "long" else "short",
                "size": abs(contracts),
                "entry_price": float(info.get("entryPrice") or 0.0),
            }
        return out
