# Rabbit Hunter Phase 1a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付 Phase 1a：引擎骨架 + trend_following 策略，`rabbit backtest` 能跑完 BTC/ETH 2 年 1H 数据，输出完整 `reports/YYYY-MM-DD-HHMM/` 目录（含 `report.md` + `trades.parquet` + `snapshots.parquet` + `ai_context.md` + `charts/*.png`）。

**Architecture:** 单进程模块化 Python 包；Data → Feature → Scoring → Router → Risk → BacktestExecutor 六层同步管线；Parquet + DuckDB 存储；Docker 单容器。

**Tech Stack:** Python 3.11 / pandas / pandas-ta / ccxt / duckdb / pyarrow / pydantic / typer / structlog / matplotlib / jinja2 / pytest / Docker

## Global Constraints

- Python 版本：3.11+（Docker 基底 `python:3.11-slim`）
- 交易所：OKX 永续合约，标的固定为 `BTC-USDT-SWAP` + `ETH-USDT-SWAP`
- 主 K 线周期：`1H`；确认周期：`15m`
- 回测窗口：`2024-07-05` → `2026-07-05`（2 年）
- 时间戳统一 UTC ISO8601
- Feature Engine 语义版本从 `0.1.0` 起
- 所有配置走 Pydantic 校验，不允许运行时才发现字段错误
- 单元测试 `<1s`；集成测试 `~10s`；回归回测 `~1min`
- 日志：`structlog` JSON 格式
- 报告目录命名：`reports/{YYYY-MM-DD-HHMM}/`
- Feature Engine 无状态、逐 tick：`compute(bars_up_to_t) -> features_at_t`
- 远程仓库：`https://github.com/xiaomimi123/rabbit-Hunter-5.0-.git`（沿用 5.0 仓库名）
- Commit 规范：Conventional Commits（`feat:` / `test:` / `chore:` / `docs:` / `fix:`）

---

## Task 索引

1. Repository init + toolchain（git / pyproject / Docker / Makefile）
2. Config schema（Pydantic + YAML）
3. Data Engine · OKX Fetcher
4. Data Engine · Quality check
5. Data Engine · Storage（Parquet + DuckDB）
6. Feature Engine · Indicators
7. Feature Engine · Price Action features
8. Feature Engine · Regime labeling
9. Feature Engine · Pipeline + Cache + Baseline snapshot
10. Scoring · BaseStrategy + Hard rules
11. Scoring · Trend Following strategy
12. Strategy Router · weighted_avg composer
13. Risk Engine · Position sizing + Daily circuit
14. Execution · BacktestExecutor
15. Ledger + Backtest main loop
16. Observability · structlog + Snapshot writer
17. Report Layer · 全套输出（report.md / trades.parquet / ai_context.md / charts）
18. CLI（typer）
19. End-to-end integration test + smoke run
20. Merge to `main` + tag `v0.1a`

---

## Task 1: Repository init + toolchain

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `Makefile`
- Create: `docker/Dockerfile`
- Create: `docker/docker-compose.yml`
- Create: `rabbit_hunter/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: —
- Produces: 一个可 `pip install -e .` 的项目骨架；`make test` 能跑起来。

- [ ] **Step 1: Init git 仓库并加 remote**

```bash
cd /Users/lizhishaoniange/Documents/RabbitHunter5.1
git init
git branch -M main
git remote add origin https://github.com/xiaomimi123/rabbit-Hunter-5.0-.git
```

- [ ] **Step 2: 写 `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.coverage
htmlcov/
.venv/
venv/

# Rabbit Hunter runtime data (不入库)
data/
snapshots/
reports/

# IDE
.vscode/
.idea/
*.swp
.DS_Store
```

- [ ] **Step 3: 写 `pyproject.toml`**

```toml
[project]
name = "rabbit-hunter"
version = "0.1a0"
description = "Rule-based crypto perpetual quant engine (Phase 1a)"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.2",
    "pyarrow>=15.0",
    "duckdb>=0.10",
    "pandas-ta>=0.3.14b0",
    "ccxt>=4.2",
    "pydantic>=2.6",
    "pyyaml>=6.0",
    "structlog>=24.1",
    "typer>=0.12",
    "matplotlib>=3.8",
    "jinja2>=3.1",
    "scipy>=1.12",
    "numpy>=1.26,<2.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=4.1"]

[project.scripts]
rabbit = "rabbit_hunter.cli:app"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["rabbit_hunter*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --strict-markers"
```

- [ ] **Step 4: 写 `docker/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY rabbit_hunter ./rabbit_hunter
RUN pip install --no-cache-dir -e ".[dev]"

CMD ["rabbit", "--help"]
```

- [ ] **Step 5: 写 `docker/docker-compose.yml`**

```yaml
version: "3.9"
services:
  rabbit:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    volumes:
      - ../data:/app/data
      - ../snapshots:/app/snapshots
      - ../reports:/app/reports
      - ../configs:/app/configs
      - ../rabbit_hunter:/app/rabbit_hunter
      - ../tests:/app/tests
    working_dir: /app
    tty: true
```

- [ ] **Step 6: 写 `Makefile`**

```makefile
.PHONY: build test fetch backtest features shell clean

DC := docker compose -f docker/docker-compose.yml

build:
	$(DC) build

test:
	$(DC) run --rm rabbit pytest

fetch:
	$(DC) run --rm rabbit rabbit fetch

backtest:
	$(DC) run --rm rabbit rabbit backtest

features:
	$(DC) run --rm rabbit rabbit features build

shell:
	$(DC) run --rm rabbit bash

clean:
	rm -rf data/features snapshots reports
```

- [ ] **Step 7: 写 `README.md`（最小版）**

```markdown
# Rabbit Hunter V5.1 · Phase 1a

Rule-based crypto perpetual quant engine.

## Quickstart

```bash
make build
make fetch      # 拉 2 年 BTC/ETH 1H 数据
make backtest   # 跑回测，输出 reports/YYYY-MM-DD-HHMM/
make test       # 跑测试
```

详见 `docs/superpowers/specs/2026-07-05-rabbit-hunter-phase-1-design.md`。
```

- [ ] **Step 8: 写空的 `rabbit_hunter/__init__.py`**

```python
__version__ = "0.1a0"
```

- [ ] **Step 9: 写 smoke 测试 `tests/test_smoke.py`**

```python
import rabbit_hunter


def test_package_importable():
    assert rabbit_hunter.__version__ == "0.1a0"
```

- [ ] **Step 10: 验证包能装、smoke 能过（本机验证）**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/test_smoke.py -v
```

Expected: `tests/test_smoke.py::test_package_importable PASSED`

- [ ] **Step 11: 首次 commit**

```bash
git add .gitignore pyproject.toml README.md Makefile docker/ rabbit_hunter/ tests/
git commit -m "chore: bootstrap Phase 1a project scaffold"
```

---

## Task 2: Config schema（Pydantic + YAML）

**Files:**
- Create: `rabbit_hunter/config/__init__.py`
- Create: `rabbit_hunter/config/schema.py`
- Create: `rabbit_hunter/config/loader.py`
- Create: `configs/default.yaml`
- Create: `configs/strategies/trend_following.yaml`
- Create: `tests/config/__init__.py`
- Create: `tests/config/test_loader.py`

**Interfaces:**
- Consumes: —
- Produces: `load_config(path: str) -> AppConfig`；`AppConfig` 含 `.data / .feature_engine / .strategy_router / .risk / .execution / .backtest / .report` 子对象。

- [ ] **Step 1: 写失败测试 `tests/config/test_loader.py`**

```python
from pathlib import Path
from rabbit_hunter.config.loader import load_config


def test_load_default_config(tmp_path):
    cfg_path = Path("configs/default.yaml")
    cfg = load_config(cfg_path)
    assert cfg.data.symbols == ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]
    assert cfg.data.main_interval == "1H"
    assert cfg.data.confirm_interval == "15m"
    assert cfg.risk.risk_per_trade_pct == 1.0
    assert cfg.backtest.initial_capital == 10000
    assert "trend_following" in cfg.strategy_router.enabled_strategies
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/config/test_loader.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 写 `rabbit_hunter/config/schema.py`**

```python
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exchange: Literal["okx"] = "okx"
    symbols: list[str]
    main_interval: str
    confirm_interval: str
    history_window_days: int = Field(gt=0)


class FeatureEngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    cache_enabled: bool = True


class StrategyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    weight: float = Field(ge=0)
    config_file: str


class StrategyRouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    composer: Literal["weighted_avg", "unanimous", "regime_switch", "max_score"] = "weighted_avg"
    enabled_strategies: dict[str, StrategyEntry]


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    risk_per_trade_pct: float = Field(gt=0)
    atr_stop_multiplier: float = Field(gt=0)
    reward_risk_ratio: float = Field(gt=0)
    max_leverage: float = Field(gt=0)
    daily_max_loss_pct: float = Field(gt=0)
    hold_timeout_bars: int = Field(gt=0)


class FeeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    maker: float  # 小数形式：0.0002 表示 0.02%
    taker: float


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fees: FeeConfig
    slippage_atr_multiplier: float = Field(ge=0)
    funding_settlement: bool = True


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: str  # ISO date
    end: str
    initial_capital: float = Field(gt=0)


class ReportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_dir: str = "reports"
    ai_context: bool = True


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: DataConfig
    feature_engine: FeatureEngineConfig
    strategy_router: StrategyRouterConfig
    risk: RiskConfig
    execution: ExecutionConfig
    backtest: BacktestConfig
    report: ReportConfig
```

- [ ] **Step 4: 写 `rabbit_hunter/config/loader.py`**

```python
from pathlib import Path
import yaml
from .schema import AppConfig


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
```

- [ ] **Step 5: 写 `rabbit_hunter/config/__init__.py`**

```python
from .loader import load_config
from .schema import AppConfig

__all__ = ["load_config", "AppConfig"]
```

- [ ] **Step 6: 写 `configs/default.yaml`**

```yaml
data:
  exchange: okx
  symbols: [BTC-USDT-SWAP, ETH-USDT-SWAP]
  main_interval: "1H"
  confirm_interval: "15m"
  history_window_days: 730

feature_engine:
  version: "0.1.0"
  cache_enabled: true

strategy_router:
  composer: weighted_avg
  enabled_strategies:
    trend_following:
      weight: 1.0
      config_file: strategies/trend_following.yaml

risk:
  risk_per_trade_pct: 1.0
  atr_stop_multiplier: 1.5
  reward_risk_ratio: 2.0
  max_leverage: 3
  daily_max_loss_pct: 3.0
  hold_timeout_bars: 48

execution:
  fees:
    maker: 0.0002
    taker: 0.0005
  slippage_atr_multiplier: 0.1
  funding_settlement: true

backtest:
  start: "2024-07-05"
  end: "2026-07-05"
  initial_capital: 10000

report:
  output_dir: reports
  ai_context: true
```

- [ ] **Step 7: 写 `configs/strategies/trend_following.yaml`**

```yaml
name: trend_following
version: "0.1.0"
params:
  ema_fast: 20
  ema_slow: 60
  ema_trend: 200
  adx_threshold: 25
  volume_ratio_threshold: 1.2
  confirm_ema_fast: 20
  confirm_adx_threshold: 20
```

- [ ] **Step 8: 跑测试确认通过**

Run: `pytest tests/config/test_loader.py -v`
Expected: PASS

- [ ] **Step 9: 追加负例测试**

在 `tests/config/test_loader.py` 底部追加：

```python
import pytest
from pydantic import ValidationError


def test_load_config_rejects_unknown_field(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "data:\n"
        "  exchange: okx\n"
        "  symbols: [BTC-USDT-SWAP]\n"
        "  main_interval: '1H'\n"
        "  confirm_interval: '15m'\n"
        "  history_window_days: 30\n"
        "  unknown_field: oops\n"
        "feature_engine: {version: '0.1.0'}\n"
        "strategy_router:\n"
        "  composer: weighted_avg\n"
        "  enabled_strategies: {}\n"
        "risk: {risk_per_trade_pct: 1, atr_stop_multiplier: 1.5,"
        " reward_risk_ratio: 2, max_leverage: 3, daily_max_loss_pct: 3,"
        " hold_timeout_bars: 48}\n"
        "execution:\n"
        "  fees: {maker: 0.0002, taker: 0.0005}\n"
        "  slippage_atr_multiplier: 0.1\n"
        "backtest: {start: '2024-01-01', end: '2024-06-01', initial_capital: 1000}\n"
        "report: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(bad)
```

Run: `pytest tests/config/test_loader.py -v`
Expected: 两个用例全 PASS

- [ ] **Step 10: Commit**

```bash
git add rabbit_hunter/config configs tests/config
git commit -m "feat(config): pydantic schema + YAML loader"
```

---

## Task 3: Data Engine · OKX Fetcher

**Files:**
- Create: `rabbit_hunter/data_engine/__init__.py`
- Create: `rabbit_hunter/data_engine/okx_fetcher.py`
- Create: `tests/data_engine/__init__.py`
- Create: `tests/data_engine/test_okx_fetcher.py`

**Interfaces:**
- Consumes: —
- Produces:
  - `fetch_ohlcv(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame`
    - 列：`timestamp` (int64 ms), `open`, `high`, `low`, `close`, `volume`
  - `fetch_funding_rate_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame`
    - 列：`timestamp`, `funding_rate`
  - `fetch_open_interest_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame`
    - 列：`timestamp`, `oi`

- [ ] **Step 1: 写失败测试 `tests/data_engine/test_okx_fetcher.py`**

```python
from unittest.mock import MagicMock, patch
import pandas as pd
from rabbit_hunter.data_engine.okx_fetcher import fetch_ohlcv


def _fake_ohlcv_batch(base_ms: int, n: int):
    return [
        [base_ms + i * 3_600_000, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0 + i]
        for i in range(n)
    ]


def test_fetch_ohlcv_pages_and_stops_at_end():
    start_ms = 1_700_000_000_000
    end_ms = start_ms + 3 * 3_600_000  # 3 小时窗口
    mock_ex = MagicMock()
    # 第一次返回 2 根，第二次返回 1 根（末尾），第三次为空
    mock_ex.fetch_ohlcv.side_effect = [
        _fake_ohlcv_batch(start_ms, 2),
        _fake_ohlcv_batch(start_ms + 2 * 3_600_000, 1),
        [],
    ]
    with patch("rabbit_hunter.data_engine.okx_fetcher._build_exchange", return_value=mock_ex):
        df = fetch_ohlcv("BTC-USDT-SWAP", "1H", start_ms, end_ms)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 3
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].iloc[0] == start_ms
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/data_engine/test_okx_fetcher.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 写 `rabbit_hunter/data_engine/okx_fetcher.py`**

```python
from __future__ import annotations
import time
from typing import Any
import ccxt
import pandas as pd

_OKX_INTERVAL_MAP = {"1H": "1h", "15m": "15m", "1h": "1h", "5m": "5m", "1D": "1d"}
_LIMIT = 200
_SLEEP_MS = 200


def _build_exchange() -> Any:
    return ccxt.okx({"enableRateLimit": True})


def _to_ccxt_symbol(symbol: str) -> str:
    # OKX 永续："BTC-USDT-SWAP" -> ccxt: "BTC/USDT:USDT"
    base, quote, tail = symbol.split("-")
    return f"{base}/{quote}:{quote}"


def fetch_ohlcv(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    ex = _build_exchange()
    ccxt_symbol = _to_ccxt_symbol(symbol)
    tf = _OKX_INTERVAL_MAP[interval]
    all_rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = ex.fetch_ohlcv(ccxt_symbol, timeframe=tf, since=cursor, limit=_LIMIT)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        time.sleep(_SLEEP_MS / 1000)
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df[df["timestamp"] < end_ms].reset_index(drop=True)
    df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)
    return df


def fetch_funding_rate_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    ex = _build_exchange()
    ccxt_symbol = _to_ccxt_symbol(symbol)
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = ex.fetch_funding_rate_history(ccxt_symbol, since=cursor, limit=_LIMIT)
        if not batch:
            break
        for r in batch:
            rows.append({"timestamp": r["timestamp"], "funding_rate": r["fundingRate"]})
        last_ts = batch[-1]["timestamp"]
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        time.sleep(_SLEEP_MS / 1000)
    df = pd.DataFrame(rows, columns=["timestamp", "funding_rate"])
    df = df[df["timestamp"] < end_ms].drop_duplicates(subset="timestamp").reset_index(drop=True)
    return df


def fetch_open_interest_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """OKX Open Interest 历史。ccxt: fetch_open_interest_history。"""
    ex = _build_exchange()
    ccxt_symbol = _to_ccxt_symbol(symbol)
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = ex.fetch_open_interest_history(ccxt_symbol, timeframe="1h", since=cursor, limit=_LIMIT)
        if not batch:
            break
        for r in batch:
            rows.append({"timestamp": r["timestamp"], "oi": r["openInterestAmount"]})
        last_ts = batch[-1]["timestamp"]
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        time.sleep(_SLEEP_MS / 1000)
    df = pd.DataFrame(rows, columns=["timestamp", "oi"])
    df = df[df["timestamp"] < end_ms].drop_duplicates(subset="timestamp").reset_index(drop=True)
    return df
```

- [ ] **Step 4: 写空 `__init__.py`**

`rabbit_hunter/data_engine/__init__.py`：

```python
from .okx_fetcher import fetch_ohlcv, fetch_funding_rate_history, fetch_open_interest_history

__all__ = ["fetch_ohlcv", "fetch_funding_rate_history", "fetch_open_interest_history"]
```

- [ ] **Step 5: 跑测试确认通过**

Run: `pytest tests/data_engine/test_okx_fetcher.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add rabbit_hunter/data_engine tests/data_engine
git commit -m "feat(data): OKX ccxt fetcher with pagination for OHLCV/funding/OI"
```

---

## Task 4: Data Engine · Quality check

**Files:**
- Create: `rabbit_hunter/data_engine/quality.py`
- Modify: `rabbit_hunter/data_engine/__init__.py`
- Create: `tests/data_engine/test_quality.py`

**Interfaces:**
- Consumes: `pd.DataFrame` with columns `[timestamp, open, high, low, close, volume]`
- Produces:
  - `check_ohlcv(df, interval: str) -> QualityReport`
  - `QualityReport` dataclass: `.clean_df: pd.DataFrame`, `.issues: list[dict]`, `.is_ok: bool`

- [ ] **Step 1: 写失败测试**

`tests/data_engine/test_quality.py`：

```python
import pandas as pd
import pytest
from rabbit_hunter.data_engine.quality import check_ohlcv


def _mk(ts_seq, close_val=100.0):
    return pd.DataFrame({
        "timestamp": ts_seq,
        "open": [close_val] * len(ts_seq),
        "high": [close_val + 1] * len(ts_seq),
        "low": [close_val - 1] * len(ts_seq),
        "close": [close_val] * len(ts_seq),
        "volume": [10.0] * len(ts_seq),
    })


def test_clean_bars_pass():
    df = _mk([0, 3_600_000, 7_200_000])
    r = check_ohlcv(df, "1H")
    assert r.is_ok
    assert len(r.clean_df) == 3
    assert r.issues == []


def test_gap_detected():
    df = _mk([0, 3_600_000, 10_800_000])  # 缺一根
    r = check_ohlcv(df, "1H")
    assert not r.is_ok
    assert any(i["type"] == "gap" for i in r.issues)


def test_bad_prices_dropped():
    df = _mk([0, 3_600_000], close_val=100.0)
    df.loc[1, "close"] = -1.0
    r = check_ohlcv(df, "1H")
    assert len(r.clean_df) == 1
    assert any(i["type"] == "invalid_price" for i in r.issues)


def test_duplicate_timestamps_dropped():
    df = _mk([0, 3_600_000, 3_600_000])
    r = check_ohlcv(df, "1H")
    assert len(r.clean_df) == 2
    assert any(i["type"] == "duplicate_timestamp" for i in r.issues)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/data_engine/test_quality.py -v`
Expected: FAIL

- [ ] **Step 3: 写 `rabbit_hunter/data_engine/quality.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

_INTERVAL_MS = {"1H": 3_600_000, "15m": 900_000, "5m": 300_000, "1D": 86_400_000}


@dataclass
class QualityReport:
    clean_df: pd.DataFrame
    issues: list[dict] = field(default_factory=list)

    @property
    def is_ok(self) -> bool:
        return len(self.issues) == 0


def check_ohlcv(df: pd.DataFrame, interval: str) -> QualityReport:
    issues: list[dict] = []
    step = _INTERVAL_MS[interval]

    # 排序
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 重复时间戳
    dup_mask = df["timestamp"].duplicated(keep="first")
    if dup_mask.any():
        for ts in df.loc[dup_mask, "timestamp"].tolist():
            issues.append({"type": "duplicate_timestamp", "timestamp": int(ts)})
    df = df[~dup_mask].reset_index(drop=True)

    # 无效价格 / 无效成交量 / NaN
    price_cols = ["open", "high", "low", "close"]
    bad_price_mask = (
        df[price_cols].le(0).any(axis=1)
        | df[price_cols].isna().any(axis=1)
        | df["volume"].lt(0)
        | df["volume"].isna()
    )
    if bad_price_mask.any():
        for ts in df.loc[bad_price_mask, "timestamp"].tolist():
            issues.append({"type": "invalid_price", "timestamp": int(ts)})
        df = df[~bad_price_mask].reset_index(drop=True)

    # High/Low 关系检查
    hilo_bad = (df["high"] < df["low"]) | (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    if hilo_bad.any():
        for ts in df.loc[hilo_bad, "timestamp"].tolist():
            issues.append({"type": "invalid_hilo", "timestamp": int(ts)})
        df = df[~hilo_bad].reset_index(drop=True)

    # 跳空
    if len(df) >= 2:
        diffs = df["timestamp"].diff().iloc[1:]
        gap_mask = diffs > step
        for pos, is_gap in enumerate(gap_mask, start=1):
            if is_gap:
                issues.append({
                    "type": "gap",
                    "before_ts": int(df["timestamp"].iloc[pos - 1]),
                    "after_ts": int(df["timestamp"].iloc[pos]),
                    "missing_bars": int(diffs.iloc[pos - 1] // step) - 1,
                })

    return QualityReport(clean_df=df, issues=issues)
```

- [ ] **Step 4: 在 `rabbit_hunter/data_engine/__init__.py` 追加导出**

```python
from .quality import check_ohlcv, QualityReport
```
（追加到已有的 `__all__` 里）

- [ ] **Step 5: 跑测试确认通过**

Run: `pytest tests/data_engine/test_quality.py -v`
Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
git add rabbit_hunter/data_engine tests/data_engine
git commit -m "feat(data): OHLCV quality check (gap/dup/invalid_price/hilo)"
```

---

## Task 5: Data Engine · Storage（Parquet + DuckDB）

**Files:**
- Create: `rabbit_hunter/data_engine/storage.py`
- Modify: `rabbit_hunter/data_engine/__init__.py`
- Create: `tests/data_engine/test_storage.py`

**Interfaces:**
- Consumes: `pd.DataFrame`
- Produces:
  - `write_ohlcv(df, root: Path, symbol: str, interval: str) -> list[Path]`（按 year/month 分区）
  - `read_ohlcv(root: Path, symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame`

- [ ] **Step 1: 写失败测试 `tests/data_engine/test_storage.py`**

```python
from datetime import datetime, timezone
import pandas as pd
from rabbit_hunter.data_engine.storage import write_ohlcv, read_ohlcv


def _mk_df(start_dt: datetime, n: int, step_h: int = 1):
    ts = [int((start_dt.timestamp() + i * step_h * 3600) * 1000) for i in range(n)]
    return pd.DataFrame({
        "timestamp": ts,
        "open": [1.0 + i for i in range(n)],
        "high": [1.5 + i for i in range(n)],
        "low": [0.5 + i for i in range(n)],
        "close": [1.2 + i for i in range(n)],
        "volume": [10.0 + i for i in range(n)],
    })


def test_write_and_read_roundtrip(tmp_path):
    df = _mk_df(datetime(2025, 1, 1, tzinfo=timezone.utc), 48)  # 2 天
    paths = write_ohlcv(df, tmp_path, "BTC-USDT-SWAP", "1H")
    assert len(paths) >= 1
    for p in paths:
        assert p.exists()
    start_ms = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime(2025, 1, 3, tzinfo=timezone.utc).timestamp() * 1000)
    df_back = read_ohlcv(tmp_path, "BTC-USDT-SWAP", "1H", start_ms, end_ms)
    assert len(df_back) == 48
    assert df_back["timestamp"].is_monotonic_increasing


def test_write_crossing_month_creates_two_partitions(tmp_path):
    df = _mk_df(datetime(2025, 1, 31, 20, tzinfo=timezone.utc), 12)
    paths = write_ohlcv(df, tmp_path, "BTC-USDT-SWAP", "1H")
    partition_paths = {p.parent.name for p in paths}
    # 应该同时落在 month=01 和 month=02
    assert any("month=01" in p.as_posix() for p in paths)
    assert any("month=02" in p.as_posix() for p in paths)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/data_engine/test_storage.py -v`
Expected: FAIL

- [ ] **Step 3: 写 `rabbit_hunter/data_engine/storage.py`**

```python
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import duckdb


def _partition_path(root: Path, symbol: str, interval: str, year: int, month: int) -> Path:
    return root / "raw" / "okx" / symbol / interval / f"year={year}" / f"month={month:02d}.parquet"


def write_ohlcv(df: pd.DataFrame, root: Path, symbol: str, interval: str) -> list[Path]:
    if df.empty:
        return []
    df = df.copy()
    dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["_year"] = dt.dt.year
    df["_month"] = dt.dt.month
    written: list[Path] = []
    for (year, month), grp in df.groupby(["_year", "_month"], sort=True):
        out = _partition_path(root, symbol, interval, int(year), int(month))
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            existing = pd.read_parquet(out)
            combined = pd.concat([existing, grp.drop(columns=["_year", "_month"])])
            combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        else:
            combined = grp.drop(columns=["_year", "_month"]).sort_values("timestamp").reset_index(drop=True)
        combined.to_parquet(out, index=False)
        written.append(out)
    return written


def read_ohlcv(root: Path, symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    glob = str(root / "raw" / "okx" / symbol / interval / "year=*" / "month=*.parquet")
    con = duckdb.connect()
    q = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM parquet_scan('{glob}')
        WHERE timestamp >= {start_ms} AND timestamp < {end_ms}
        ORDER BY timestamp
    """
    df = con.execute(q).fetch_df()
    con.close()
    return df.reset_index(drop=True)
```

- [ ] **Step 4: 在 `__init__.py` 追加导出**

```python
from .storage import write_ohlcv, read_ohlcv
```

- [ ] **Step 5: 跑测试确认通过**

Run: `pytest tests/data_engine/test_storage.py -v`
Expected: 2 PASS

- [ ] **Step 6: Commit**

```bash
git add rabbit_hunter/data_engine tests/data_engine
git commit -m "feat(data): parquet partitioned storage + duckdb read"
```

---

## Task 6: Feature Engine · Indicators

**Files:**
- Create: `rabbit_hunter/feature_engine/__init__.py`
- Create: `rabbit_hunter/feature_engine/indicators.py`
- Create: `tests/feature_engine/__init__.py`
- Create: `tests/feature_engine/test_indicators.py`

**Interfaces:**
- Consumes: `pd.DataFrame` with `[open, high, low, close, volume]`
- Produces: `compute_indicators(df) -> pd.DataFrame`（追加列，不修改原列）：
  - `ema20`, `ema60`, `ema200`, `ema20_slope`
  - `adx`, `di_plus`, `di_minus`
  - `rsi_14`
  - `bb_upper`, `bb_middle`, `bb_lower`, `bb_width`, `bb_pct`, `zscore_20`
  - `atr_14`, `atr_pct`
  - `volume_ratio_20`

- [ ] **Step 1: 写失败测试 `tests/feature_engine/test_indicators.py`**

```python
import numpy as np
import pandas as pd
from rabbit_hunter.feature_engine.indicators import compute_indicators


def _mk_trend_df(n: int = 300):
    close = np.linspace(100, 200, n)
    high = close + 0.5
    low = close - 0.5
    open_ = close - 0.2
    vol = np.full(n, 100.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})


def test_indicators_columns_present():
    df = _mk_trend_df()
    out = compute_indicators(df)
    for col in [
        "ema20", "ema60", "ema200", "ema20_slope",
        "adx", "di_plus", "di_minus",
        "rsi_14",
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct", "zscore_20",
        "atr_14", "atr_pct",
        "volume_ratio_20",
    ]:
        assert col in out.columns, f"missing {col}"


def test_ema_stack_in_uptrend():
    df = _mk_trend_df()
    out = compute_indicators(df).iloc[-1]
    # 稳定上涨中 EMA20 > EMA60 > EMA200
    assert out["ema20"] > out["ema60"] > out["ema200"]


def test_no_lookahead_last_row_stable():
    df = _mk_trend_df()
    full = compute_indicators(df).iloc[-1]
    partial = compute_indicators(df.iloc[:-1])
    partial_last_after_extend = compute_indicators(df).iloc[-2]
    # 去掉最后一行再算，再对同一位置的历史行取值，应与全量的对应行完全一致
    for col in ["ema20", "adx", "rsi_14", "atr_14"]:
        assert np.isclose(partial.iloc[-1][col], partial_last_after_extend[col], equal_nan=True), col
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/feature_engine/test_indicators.py -v`
Expected: FAIL

- [ ] **Step 3: 写 `rabbit_hunter/feature_engine/indicators.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
import pandas_ta as ta


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # EMA
    out["ema20"] = ta.ema(out["close"], length=20)
    out["ema60"] = ta.ema(out["close"], length=60)
    out["ema200"] = ta.ema(out["close"], length=200)
    out["ema20_slope"] = out["ema20"].diff()

    # ADX / DI
    adx_df = ta.adx(out["high"], out["low"], out["close"], length=14)
    if adx_df is not None:
        out["adx"] = adx_df["ADX_14"]
        out["di_plus"] = adx_df["DMP_14"]
        out["di_minus"] = adx_df["DMN_14"]
    else:
        out["adx"] = np.nan
        out["di_plus"] = np.nan
        out["di_minus"] = np.nan

    # RSI
    out["rsi_14"] = ta.rsi(out["close"], length=14)

    # Bollinger Bands
    bb = ta.bbands(out["close"], length=20, std=2.0)
    if bb is not None:
        out["bb_lower"] = bb["BBL_20_2.0"]
        out["bb_middle"] = bb["BBM_20_2.0"]
        out["bb_upper"] = bb["BBU_20_2.0"]
        out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_middle"]
        out["bb_pct"] = (out["close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])
    else:
        for c in ["bb_lower", "bb_middle", "bb_upper", "bb_width", "bb_pct"]:
            out[c] = np.nan

    # Z-Score(20)
    rolling_mean = out["close"].rolling(20).mean()
    rolling_std = out["close"].rolling(20).std()
    out["zscore_20"] = (out["close"] - rolling_mean) / rolling_std

    # ATR
    out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)
    out["atr_pct"] = out["atr_14"] / out["close"]

    # Volume ratio
    out["volume_ratio_20"] = out["volume"] / out["volume"].rolling(20).mean()

    return out
```

- [ ] **Step 4: 写 `rabbit_hunter/feature_engine/__init__.py`**

```python
from .indicators import compute_indicators

__all__ = ["compute_indicators"]
```

- [ ] **Step 5: 跑测试确认通过**

Run: `pytest tests/feature_engine/test_indicators.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add rabbit_hunter/feature_engine tests/feature_engine
git commit -m "feat(feature): pandas-ta indicators (EMA/ADX/RSI/BB/ATR/vol_ratio)"
```

---

## Task 7: Feature Engine · Price Action features

**Files:**
- Create: `rabbit_hunter/feature_engine/price_action.py`
- Modify: `rabbit_hunter/feature_engine/__init__.py`
- Create: `tests/feature_engine/test_price_action.py`

**Interfaces:**
- Consumes: `pd.DataFrame` with `[open, high, low, close]`（可含 indicators 输出的列，本函数不依赖）
- Produces: `compute_price_action(df) -> pd.DataFrame`（追加列）：
  - `pattern_engulfing_bull`, `pattern_engulfing_bear` (0/1)
  - `pattern_pinbar` (0/1), `pattern_inside_bar` (0/1), `pattern_doji` (0/1), `pattern_marubozu` (0/1)
  - `swing_high_last` (float), `swing_low_last` (float)（回看 20 根内最近的 swing 价，无则 NaN）
  - `structure_regime` (str: `uptrend` / `downtrend` / `range`)
  - `bos_flag` (0/1), `choch_flag` (0/1)

- [ ] **Step 1: 写失败测试 `tests/feature_engine/test_price_action.py`**

```python
import numpy as np
import pandas as pd
from rabbit_hunter.feature_engine.price_action import compute_price_action


def _bar(o, h, l, c):
    return {"open": o, "high": h, "low": l, "close": c}


def test_bullish_engulfing():
    df = pd.DataFrame([
        _bar(100, 100.5, 99, 99),      # 阴
        _bar(98.5, 101.5, 98.4, 101),  # 阳，实体吞没前一根
    ])
    out = compute_price_action(df)
    assert out["pattern_engulfing_bull"].iloc[-1] == 1
    assert out["pattern_engulfing_bear"].iloc[-1] == 0


def test_bearish_engulfing():
    df = pd.DataFrame([
        _bar(99, 101, 99, 101),
        _bar(101.5, 101.6, 98, 98.5),
    ])
    out = compute_price_action(df)
    assert out["pattern_engulfing_bear"].iloc[-1] == 1


def test_inside_bar():
    df = pd.DataFrame([
        _bar(100, 105, 95, 102),
        _bar(101, 104, 96, 103),  # 完全在前一根 high/low 内
    ])
    out = compute_price_action(df)
    assert out["pattern_inside_bar"].iloc[-1] == 1


def test_doji():
    df = pd.DataFrame([_bar(100, 100.5, 99.5, 100.001)])
    out = compute_price_action(df)
    assert out["pattern_doji"].iloc[-1] == 1


def test_structure_and_bos():
    # 构造 HH-HL 上升结构后跌破前低 → BOS 下
    closes = [100, 102, 101, 104, 103, 106, 105, 108, 100]
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    opens = [c - 0.2 for c in closes]
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})
    out = compute_price_action(df)
    assert out["structure_regime"].iloc[-1] in {"uptrend", "downtrend", "range"}
    # 最后一根低于前几根，至少能产出 bos_flag=1 或 choch_flag=1 之一
    assert out["bos_flag"].iloc[-1] == 1 or out["choch_flag"].iloc[-1] == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/feature_engine/test_price_action.py -v`
Expected: FAIL

- [ ] **Step 3: 写 `rabbit_hunter/feature_engine/price_action.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd


def _engulfing(o1, c1, o2, c2, bullish: bool) -> int:
    body1_top = max(o1, c1); body1_bot = min(o1, c1)
    body2_top = max(o2, c2); body2_bot = min(o2, c2)
    if bullish:
        return int(c1 < o1 and c2 > o2 and body2_top >= body1_top and body2_bot <= body1_bot)
    else:
        return int(c1 > o1 and c2 < o2 and body2_top >= body1_top and body2_bot <= body1_bot)


def _pinbar(o, h, l, c) -> int:
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    rng = h - l
    if rng == 0:
        return 0
    return int((upper > 2 * body and upper / rng > 0.6) or (lower > 2 * body and lower / rng > 0.6))


def _doji(o, h, l, c) -> int:
    body = abs(c - o)
    rng = h - l
    if rng == 0:
        return 0
    return int(body / rng < 0.1)


def _marubozu(o, h, l, c) -> int:
    body = abs(c - o)
    rng = h - l
    if rng == 0:
        return 0
    return int(body / rng > 0.9)


def _swing_points(highs: np.ndarray, lows: np.ndarray, lookback: int = 3) -> tuple[np.ndarray, np.ndarray]:
    n = len(highs)
    swing_h = np.zeros(n, dtype=bool)
    swing_l = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback: i + lookback + 1]
        window_l = lows[i - lookback: i + lookback + 1]
        if highs[i] == window_h.max() and (highs[i] > highs[i - 1]):
            swing_h[i] = True
        if lows[i] == window_l.min() and (lows[i] < lows[i - 1]):
            swing_l[i] = True
    return swing_h, swing_l


def compute_price_action(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    o = out["open"].to_numpy(dtype=float)
    h = out["high"].to_numpy(dtype=float)
    l = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)

    engulf_bull = np.zeros(n, dtype=int)
    engulf_bear = np.zeros(n, dtype=int)
    inside = np.zeros(n, dtype=int)
    pin = np.zeros(n, dtype=int)
    doji = np.zeros(n, dtype=int)
    maru = np.zeros(n, dtype=int)

    for i in range(n):
        if i >= 1:
            engulf_bull[i] = _engulfing(o[i-1], c[i-1], o[i], c[i], True)
            engulf_bear[i] = _engulfing(o[i-1], c[i-1], o[i], c[i], False)
            inside[i] = int(h[i] <= h[i-1] and l[i] >= l[i-1])
        pin[i] = _pinbar(o[i], h[i], l[i], c[i])
        doji[i] = _doji(o[i], h[i], l[i], c[i])
        maru[i] = _marubozu(o[i], h[i], l[i], c[i])

    out["pattern_engulfing_bull"] = engulf_bull
    out["pattern_engulfing_bear"] = engulf_bear
    out["pattern_pinbar"] = pin
    out["pattern_inside_bar"] = inside
    out["pattern_doji"] = doji
    out["pattern_marubozu"] = maru

    # Swing points + last swing price（回看 20 根内）
    swing_h_mask, swing_l_mask = _swing_points(h, l, lookback=3)
    swing_high_last = np.full(n, np.nan)
    swing_low_last = np.full(n, np.nan)
    last_h = np.nan; last_l = np.nan
    for i in range(n):
        if swing_h_mask[i]:
            last_h = h[i]
        if swing_l_mask[i]:
            last_l = l[i]
        swing_high_last[i] = last_h
        swing_low_last[i] = last_l
    out["swing_high_last"] = swing_high_last
    out["swing_low_last"] = swing_low_last

    # Structure regime + BOS / CHoCH（简化实现）
    structure = np.array(["range"] * n, dtype=object)
    bos = np.zeros(n, dtype=int)
    choch = np.zeros(n, dtype=int)
    prev_regime = "range"
    highs_seen: list[float] = []
    lows_seen: list[float] = []
    for i in range(n):
        if swing_h_mask[i]:
            highs_seen.append(h[i])
        if swing_l_mask[i]:
            lows_seen.append(l[i])
        recent_h = highs_seen[-2:]
        recent_l = lows_seen[-2:]
        if len(recent_h) == 2 and len(recent_l) == 2:
            if recent_h[-1] > recent_h[-2] and recent_l[-1] > recent_l[-2]:
                regime = "uptrend"
            elif recent_h[-1] < recent_h[-2] and recent_l[-1] < recent_l[-2]:
                regime = "downtrend"
            else:
                regime = "range"
        else:
            regime = "range"
        structure[i] = regime

        if prev_regime == "uptrend" and not np.isnan(last_l) and l[i] < last_l:
            bos[i] = 1
        if prev_regime == "downtrend" and not np.isnan(last_h) and h[i] > last_h:
            bos[i] = 1
        if regime != prev_regime and prev_regime != "range" and regime != "range":
            choch[i] = 1
        prev_regime = regime

    out["structure_regime"] = structure
    out["bos_flag"] = bos
    out["choch_flag"] = choch
    return out
```

- [ ] **Step 4: 追加到 `__init__.py` 导出**

```python
from .price_action import compute_price_action
```

- [ ] **Step 5: 跑测试确认通过**

Run: `pytest tests/feature_engine/test_price_action.py -v`
Expected: 5 PASS

- [ ] **Step 6: Commit**

```bash
git add rabbit_hunter/feature_engine tests/feature_engine
git commit -m "feat(feature): price action patterns + structure/BOS/CHoCH"
```

---

## Task 8: Feature Engine · Regime labeling

**Files:**
- Create: `rabbit_hunter/feature_engine/regime.py`
- Modify: `rabbit_hunter/feature_engine/__init__.py`
- Create: `tests/feature_engine/test_regime.py`

**Interfaces:**
- Consumes: DataFrame containing `adx`, `atr_pct`（Task 6 输出的列）
- Produces:
  - `compute_regime(df) -> pd.DataFrame`（追加 `regime`, `session`, `day_of_week` 列）
  - `regime ∈ {"trending", "ranging", "high_vol", "low_vol"}`
    - `high_vol`: `atr_pct >= atr_pct.rolling(500).quantile(0.9)`
    - `low_vol`: `atr_pct <= atr_pct.rolling(500).quantile(0.1)`
    - `trending`: 否则 `adx > 25`
    - `ranging`: 其他
  - `session ∈ {"asia", "europe", "us"}`：由 UTC 小时决定（0-7 asia，8-15 europe，16-23 us）
  - `day_of_week` int 0-6

DataFrame 必须含 `timestamp` 列（毫秒 int）。

- [ ] **Step 1: 写失败测试 `tests/feature_engine/test_regime.py`**

```python
import numpy as np
import pandas as pd
from rabbit_hunter.feature_engine.regime import compute_regime


def _mk(n=600, adx_val=30.0, atr_pct_val=0.02):
    ts_ms = [i * 3_600_000 for i in range(n)]
    return pd.DataFrame({
        "timestamp": ts_ms,
        "adx": [adx_val] * n,
        "atr_pct": [atr_pct_val] * n,
    })


def test_trending_when_adx_high():
    df = _mk()
    out = compute_regime(df)
    assert out["regime"].iloc[-1] == "trending"


def test_ranging_when_adx_low():
    df = _mk(adx_val=10.0)
    out = compute_regime(df)
    assert out["regime"].iloc[-1] == "ranging"


def test_high_vol_wins_over_trend():
    df = _mk(adx_val=30.0)
    # 尾巴放极大 atr_pct
    df.loc[df.index[-1], "atr_pct"] = 1.0
    out = compute_regime(df)
    assert out["regime"].iloc[-1] == "high_vol"


def test_session_and_dow():
    df = _mk(n=48)
    out = compute_regime(df)
    for s in out["session"]:
        assert s in {"asia", "europe", "us"}
    for d in out["day_of_week"]:
        assert 0 <= d <= 6
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/feature_engine/test_regime.py -v`
Expected: FAIL

- [ ] **Step 3: 写 `rabbit_hunter/feature_engine/regime.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd


def _session_of(hour: int) -> str:
    if 0 <= hour < 8:
        return "asia"
    if 8 <= hour < 16:
        return "europe"
    return "us"


def compute_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    q_hi = out["atr_pct"].rolling(500, min_periods=50).quantile(0.9)
    q_lo = out["atr_pct"].rolling(500, min_periods=50).quantile(0.1)

    def label(row_atr, q_high, q_low, adx) -> str:
        if pd.notna(q_high) and row_atr >= q_high:
            return "high_vol"
        if pd.notna(q_low) and row_atr <= q_low:
            return "low_vol"
        if pd.notna(adx) and adx > 25:
            return "trending"
        return "ranging"

    regimes = [label(a, qh, ql, adx) for a, qh, ql, adx in zip(out["atr_pct"], q_hi, q_lo, out["adx"])]
    out["regime"] = regimes

    ts = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    out["session"] = [_session_of(t.hour) for t in ts]
    out["day_of_week"] = ts.dt.dayofweek.astype(int).to_numpy()
    return out
```

- [ ] **Step 4: 追加导出**

在 `rabbit_hunter/feature_engine/__init__.py` 追加：

```python
from .regime import compute_regime
```

- [ ] **Step 5: 跑测试确认通过**

Run: `pytest tests/feature_engine/test_regime.py -v`
Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
git add rabbit_hunter/feature_engine tests/feature_engine
git commit -m "feat(feature): auto regime + session + day_of_week labeling"
```

---

## Task 9: Feature Engine · Pipeline + Cache + Baseline snapshot

**Files:**
- Create: `rabbit_hunter/feature_engine/pipeline.py`
- Modify: `rabbit_hunter/feature_engine/__init__.py`
- Create: `tests/feature_engine/test_pipeline.py`
- Create: `tests/baselines/features_v0_1_0.csv`（Step 5 生成后 commit）

**Interfaces:**
- Consumes:
  - `raw_ohlcv: pd.DataFrame`（含 `timestamp`, OHLCV，可选 `funding_rate`, `oi`）
  - `confirm_ohlcv: pd.DataFrame | None`（15m K 线，用于跨周期对齐）
- Produces:
  - `build_features(raw, confirm=None, engine_version="0.1.0") -> pd.DataFrame`
    - 输出列：原始 OHLCV + 所有 indicators + 所有 PA + regime/session/day_of_week + `ema20_1h_on_15m`, `adx_1h_on_15m`（若给了 confirm 就填，否则 NaN）+ `funding_rate` + `oi_change_pct`（若原始给了就填）
  - `load_or_compute_features(root, symbol, interval, engine_version, ...) -> pd.DataFrame`
    - 缓存路径：`{root}/features/{symbol}/{interval}/features_v{engine_version}.parquet`
    - 若缓存存在且版本匹配 → 直接读；否则重算并写缓存

- [ ] **Step 1: 写失败测试 `tests/feature_engine/test_pipeline.py`**

```python
from pathlib import Path
import numpy as np
import pandas as pd
from rabbit_hunter.feature_engine.pipeline import build_features, load_or_compute_features


def _mk_raw(n=400, base=100.0):
    ts = [i * 3_600_000 for i in range(n)]
    close = np.linspace(base, base + 100, n)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 10.0),
        "funding_rate": np.full(n, 0.0001),
        "oi": np.linspace(1000, 1500, n),
    })


def test_build_features_columns_present():
    raw = _mk_raw()
    feats = build_features(raw)
    for col in [
        "ema20", "adx", "rsi_14", "atr_14",
        "pattern_engulfing_bull", "structure_regime",
        "regime", "session", "day_of_week",
        "funding_rate", "oi_change_pct",
    ]:
        assert col in feats.columns, f"missing {col}"
    assert len(feats) == len(raw)


def test_no_lookahead_prefix_matches():
    raw = _mk_raw()
    full = build_features(raw)
    prefix = build_features(raw.iloc[:-10])
    for col in ["ema20", "adx", "rsi_14", "atr_14"]:
        np.testing.assert_allclose(
            full[col].iloc[:-10].to_numpy(),
            prefix[col].to_numpy(),
            equal_nan=True,
        )


def test_cache_hit_returns_same(tmp_path):
    raw = _mk_raw()

    def fetch():
        return raw

    a = load_or_compute_features(
        root=tmp_path, symbol="TEST-SWAP", interval="1H",
        engine_version="0.1.0", fetch_raw=fetch,
    )
    # 第二次调用不应触发 fetch（用异常检测）
    def fetch_should_not_run():
        raise AssertionError("cache should have hit")

    b = load_or_compute_features(
        root=tmp_path, symbol="TEST-SWAP", interval="1H",
        engine_version="0.1.0", fetch_raw=fetch_should_not_run,
    )
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/feature_engine/test_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: 写 `rabbit_hunter/feature_engine/pipeline.py`**

```python
from __future__ import annotations
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd

from .indicators import compute_indicators
from .price_action import compute_price_action
from .regime import compute_regime


def _align_1h_on_15m(main_1h: pd.DataFrame, confirm_15m: pd.DataFrame) -> pd.DataFrame:
    """把 1H 的 ema20/adx 前向填充到 15m 时间轴。confirm_15m 必须含 timestamp。"""
    right = main_1h[["timestamp", "ema20", "adx"]].rename(
        columns={"ema20": "ema20_1h_on_15m", "adx": "adx_1h_on_15m"}
    ).sort_values("timestamp")
    left = confirm_15m[["timestamp"]].sort_values("timestamp")
    merged = pd.merge_asof(left, right, on="timestamp", direction="backward")
    return merged


def build_features(
    raw: pd.DataFrame,
    confirm: pd.DataFrame | None = None,
    engine_version: str = "0.1.0",
) -> pd.DataFrame:
    df = raw.copy().reset_index(drop=True)
    df = compute_indicators(df)
    df = compute_price_action(df)
    df = compute_regime(df)

    if "funding_rate" not in df.columns:
        df["funding_rate"] = np.nan
    if "oi" in df.columns:
        df["oi_change_pct"] = df["oi"].pct_change().fillna(0.0)
    else:
        df["oi_change_pct"] = np.nan

    if confirm is not None and not confirm.empty:
        # 计算 15m 上的 indicators
        confirm_ind = compute_indicators(confirm.copy().reset_index(drop=True))
        aligned = _align_1h_on_15m(df, confirm_ind)
        df = df.merge(aligned, on="timestamp", how="left")
    else:
        df["ema20_1h_on_15m"] = np.nan
        df["adx_1h_on_15m"] = np.nan

    df.attrs["engine_version"] = engine_version
    return df


def _cache_path(root: Path, symbol: str, interval: str, engine_version: str) -> Path:
    return root / "features" / symbol / interval / f"features_v{engine_version}.parquet"


def load_or_compute_features(
    root: Path,
    symbol: str,
    interval: str,
    engine_version: str,
    fetch_raw: Callable[[], pd.DataFrame],
    fetch_confirm: Callable[[], pd.DataFrame] | None = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    cache = _cache_path(root, symbol, interval, engine_version)
    if cache.exists() and not force_recompute:
        return pd.read_parquet(cache)
    raw = fetch_raw()
    confirm = fetch_confirm() if fetch_confirm is not None else None
    feats = build_features(raw, confirm, engine_version=engine_version)
    cache.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(cache, index=False)
    return feats
```

- [ ] **Step 4: 追加导出**

在 `rabbit_hunter/feature_engine/__init__.py`：

```python
from .pipeline import build_features, load_or_compute_features
```

- [ ] **Step 5: 生成 baseline 快照并加锁死回归测试**

在 `tests/feature_engine/test_pipeline.py` 末尾追加：

```python
def test_baseline_snapshot_stable():
    """
    锁死特征值。任何影响历史特征的改动都必须显式更新 baseline。
    """
    raw = _mk_raw(n=250)
    feats = build_features(raw)
    baseline_path = Path(__file__).resolve().parents[1] / "baselines" / "features_v0_1_0.csv"
    check_cols = ["timestamp", "ema20", "ema60", "adx", "rsi_14", "atr_14", "regime"]
    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        feats[check_cols].tail(50).to_csv(baseline_path, index=False)
    expected = pd.read_csv(baseline_path)
    actual = feats[check_cols].tail(50).reset_index(drop=True)
    for c in ["ema20", "ema60", "adx", "rsi_14", "atr_14"]:
        np.testing.assert_allclose(actual[c].to_numpy(), expected[c].to_numpy(), rtol=1e-9, atol=1e-9, equal_nan=True)
    assert (actual["regime"].to_numpy() == expected["regime"].to_numpy()).all()
```

- [ ] **Step 6: 跑测试确认通过（第一次会自动生成 baseline）**

Run: `pytest tests/feature_engine/test_pipeline.py -v`
Expected: 4 PASS，且 `tests/baselines/features_v0_1_0.csv` 被创建

- [ ] **Step 7: Commit（baseline 也入库）**

```bash
git add rabbit_hunter/feature_engine tests/feature_engine tests/baselines
git commit -m "feat(feature): pipeline + cache + baseline snapshot lock"
```

---

## Task 10: Scoring · BaseStrategy + Hard rules

**Files:**
- Create: `rabbit_hunter/scoring_engine/__init__.py`
- Create: `rabbit_hunter/scoring_engine/base.py`
- Create: `rabbit_hunter/scoring_engine/rules_hard.py`
- Create: `tests/scoring_engine/__init__.py`
- Create: `tests/scoring_engine/test_base.py`
- Create: `tests/scoring_engine/test_rules_hard.py`

**Interfaces:**
- Produces:
  - `ScoreOutput` dataclass: `long: float, short: float, components: dict, metadata: dict`
  - `BaseStrategy` ABC: `.name: str`, `.version: str`, `.score(features_row, features_history) -> ScoreOutput`
  - `pass_hard_rules(features_row: dict, params: HardRulesParams) -> tuple[bool, list[str]]`
  - `HardRulesParams` dataclass: `min_quote_volume_24h: float`, `atr_pct_max_multiplier: float`, `atr_pct_baseline_window: int`

- [ ] **Step 1: 写失败测试 `tests/scoring_engine/test_base.py`**

```python
import pandas as pd
from rabbit_hunter.scoring_engine.base import BaseStrategy, ScoreOutput


class Dummy(BaseStrategy):
    name = "dummy"
    version = "0.1.0"

    def score(self, features_row, features_history):
        return ScoreOutput(long=0.5, short=0.1, components={"x": 0.4}, metadata={"note": "hi"})


def test_score_output_frozen_and_typed():
    d = Dummy()
    out = d.score({}, pd.DataFrame())
    assert out.long == 0.5
    assert out.short == 0.1
    assert out.components == {"x": 0.4}
    assert out.metadata == {"note": "hi"}
```

- [ ] **Step 2: 写失败测试 `tests/scoring_engine/test_rules_hard.py`**

```python
import pandas as pd
from rabbit_hunter.scoring_engine.rules_hard import pass_hard_rules, HardRulesParams


def _row(**overrides):
    base = {
        "quote_volume_24h": 1_000_000_000.0,
        "atr_pct": 0.02,
        "atr_pct_baseline": 0.02,
    }
    base.update(overrides)
    return base


def test_pass_normal():
    ok, reasons = pass_hard_rules(_row(), HardRulesParams(
        min_quote_volume_24h=1_000_000.0,
        atr_pct_max_multiplier=5.0,
        atr_pct_baseline_window=500,
    ))
    assert ok and reasons == []


def test_reject_low_liquidity():
    ok, reasons = pass_hard_rules(
        _row(quote_volume_24h=100.0),
        HardRulesParams(min_quote_volume_24h=1_000.0, atr_pct_max_multiplier=5.0, atr_pct_baseline_window=500),
    )
    assert not ok and "low_liquidity" in reasons


def test_reject_extreme_volatility():
    ok, reasons = pass_hard_rules(
        _row(atr_pct=1.0, atr_pct_baseline=0.02),
        HardRulesParams(min_quote_volume_24h=1_000.0, atr_pct_max_multiplier=5.0, atr_pct_baseline_window=500),
    )
    assert not ok and "extreme_volatility" in reasons
```

- [ ] **Step 3: 跑测试确认失败**

Run: `pytest tests/scoring_engine -v`
Expected: FAIL

- [ ] **Step 4: 写 `rabbit_hunter/scoring_engine/base.py`**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass(frozen=True)
class ScoreOutput:
    long: float
    short: float
    components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    name: str = ""
    version: str = "0.0.0"

    @abstractmethod
    def score(self, features_row: dict, features_history: pd.DataFrame) -> ScoreOutput: ...
```

- [ ] **Step 5: 写 `rabbit_hunter/scoring_engine/rules_hard.py`**

```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class HardRulesParams:
    min_quote_volume_24h: float
    atr_pct_max_multiplier: float
    atr_pct_baseline_window: int


def pass_hard_rules(features_row: dict, params: HardRulesParams) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    qv = features_row.get("quote_volume_24h")
    if qv is None or qv < params.min_quote_volume_24h:
        reasons.append("low_liquidity")

    atr_pct = features_row.get("atr_pct")
    baseline = features_row.get("atr_pct_baseline")
    if atr_pct is not None and baseline is not None and baseline > 0:
        if atr_pct > params.atr_pct_max_multiplier * baseline:
            reasons.append("extreme_volatility")

    return (len(reasons) == 0, reasons)
```

- [ ] **Step 6: 写 `rabbit_hunter/scoring_engine/__init__.py`**

```python
from .base import BaseStrategy, ScoreOutput
from .rules_hard import pass_hard_rules, HardRulesParams

__all__ = ["BaseStrategy", "ScoreOutput", "pass_hard_rules", "HardRulesParams"]
```

- [ ] **Step 7: 跑测试确认通过**

Run: `pytest tests/scoring_engine -v`
Expected: 4 PASS

- [ ] **Step 8: Commit**

```bash
git add rabbit_hunter/scoring_engine tests/scoring_engine
git commit -m "feat(scoring): BaseStrategy interface + hard rules engine"
```

---

## Task 11: Scoring · Trend Following strategy

**Files:**
- Create: `rabbit_hunter/scoring_engine/strategies/__init__.py`
- Create: `rabbit_hunter/scoring_engine/strategies/trend_following.py`
- Create: `tests/scoring_engine/test_trend_following.py`

**Interfaces:**
- Consumes: `features_row` dict 包含 `ema20`, `ema60`, `ema200`, `adx`, `di_plus`, `di_minus`, `volume_ratio_20`, `ema20_1h_on_15m`, `adx_1h_on_15m`；`features_history` 至少 200 根
- Produces:
  - `TrendFollowing(params: TFParams)` 实现 `BaseStrategy`
  - `TFParams`: `ema_fast`, `ema_slow`, `ema_trend`, `adx_threshold`, `volume_ratio_threshold`, `confirm_ema_fast`, `confirm_adx_threshold`（对齐 `configs/strategies/trend_following.yaml`）
  - long/short score: 0.0~1.0；使用 EMA 顺序 + ADX + 成交量 + 15m 确认加权求和

- [ ] **Step 1: 写失败测试 `tests/scoring_engine/test_trend_following.py`**

```python
import pandas as pd
from rabbit_hunter.scoring_engine.strategies.trend_following import TrendFollowing, TFParams


def _params():
    return TFParams(
        ema_fast=20, ema_slow=60, ema_trend=200,
        adx_threshold=25, volume_ratio_threshold=1.2,
        confirm_ema_fast=20, confirm_adx_threshold=20,
    )


def _row(**kw):
    base = {
        "close": 100.0,
        "ema20": 105.0, "ema60": 100.0, "ema200": 90.0,
        "adx": 30.0, "di_plus": 25.0, "di_minus": 15.0,
        "volume_ratio_20": 1.5,
        "ema20_1h_on_15m": 105.0, "adx_1h_on_15m": 22.0,
    }
    base.update(kw)
    return base


def test_long_score_high_in_uptrend():
    tf = TrendFollowing(_params())
    out = tf.score(_row(), pd.DataFrame())
    assert out.long > 0.6
    assert out.short < 0.2


def test_short_score_high_in_downtrend():
    tf = TrendFollowing(_params())
    out = tf.score(
        _row(ema20=90, ema60=100, ema200=110, di_plus=15, di_minus=25,
             ema20_1h_on_15m=90),
        pd.DataFrame(),
    )
    assert out.short > 0.6
    assert out.long < 0.2


def test_low_score_when_adx_below_threshold():
    tf = TrendFollowing(_params())
    out = tf.score(_row(adx=15), pd.DataFrame())
    # ADX 弱 → 两边分都低
    assert out.long < 0.5
    assert out.short < 0.5


def test_components_present():
    tf = TrendFollowing(_params())
    out = tf.score(_row(), pd.DataFrame())
    for k in ("ema_stack", "adx", "volume", "confirm"):
        assert k in out.components
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/scoring_engine/test_trend_following.py -v`
Expected: FAIL

- [ ] **Step 3: 写 `rabbit_hunter/scoring_engine/strategies/trend_following.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from ..base import BaseStrategy, ScoreOutput


@dataclass(frozen=True)
class TFParams:
    ema_fast: int
    ema_slow: int
    ema_trend: int
    adx_threshold: float
    volume_ratio_threshold: float
    confirm_ema_fast: int
    confirm_adx_threshold: float


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class TrendFollowing(BaseStrategy):
    name = "trend_following"
    version = "0.1.0"

    def __init__(self, params: TFParams):
        self.params = params

    def score(self, features_row: dict, features_history: pd.DataFrame) -> ScoreOutput:
        p = self.params
        ema_f = features_row.get("ema20")
        ema_s = features_row.get("ema60")
        ema_t = features_row.get("ema200")
        adx = features_row.get("adx")
        di_plus = features_row.get("di_plus")
        di_minus = features_row.get("di_minus")
        vol_ratio = features_row.get("volume_ratio_20")
        confirm_ema = features_row.get("ema20_1h_on_15m")
        confirm_adx = features_row.get("adx_1h_on_15m")
        price = features_row.get("close")

        # ema_stack: 1 if bullish stack, 0 neutral, -1 bearish
        if ema_f is None or ema_s is None or ema_t is None:
            ema_stack_long = 0.0
            ema_stack_short = 0.0
        else:
            if ema_f > ema_s > ema_t:
                ema_stack_long = 1.0; ema_stack_short = 0.0
            elif ema_f < ema_s < ema_t:
                ema_stack_long = 0.0; ema_stack_short = 1.0
            else:
                ema_stack_long = 0.0; ema_stack_short = 0.0

        # adx_score
        if adx is None:
            adx_score = 0.0
        else:
            adx_score = _clip01((adx - p.adx_threshold) / max(p.adx_threshold, 1e-9))
            adx_score = _clip01(adx_score)

        # DI 方向偏向
        if di_plus is None or di_minus is None:
            di_long = 0.5; di_short = 0.5
        else:
            total = di_plus + di_minus
            if total <= 0:
                di_long = 0.5; di_short = 0.5
            else:
                di_long = di_plus / total
                di_short = di_minus / total

        # volume_score
        if vol_ratio is None:
            vol_score = 0.0
        else:
            vol_score = _clip01((vol_ratio - p.volume_ratio_threshold) / max(p.volume_ratio_threshold, 1e-9))

        # 15m confirm
        if confirm_ema is None or price is None:
            confirm_long = 0.0; confirm_short = 0.0
        else:
            confirm_long = 1.0 if price > confirm_ema else 0.0
            confirm_short = 1.0 if price < confirm_ema else 0.0
        if confirm_adx is not None and confirm_adx < p.confirm_adx_threshold:
            confirm_long *= 0.5
            confirm_short *= 0.5

        # 权重（合计 1.0）
        w_ema, w_adx, w_vol, w_conf = 0.4, 0.25, 0.15, 0.20
        long_score = (
            w_ema * ema_stack_long
            + w_adx * adx_score * di_long * 2  # DI 已归一到 0.5 中枢，乘 2 归一到 [0,1]
            + w_vol * vol_score
            + w_conf * confirm_long
        )
        short_score = (
            w_ema * ema_stack_short
            + w_adx * adx_score * di_short * 2
            + w_vol * vol_score * (ema_stack_short)  # 只在空头结构下算成交量
            + w_conf * confirm_short
        )

        return ScoreOutput(
            long=_clip01(long_score),
            short=_clip01(short_score),
            components={
                "ema_stack": ema_stack_long - ema_stack_short,
                "adx": adx_score,
                "volume": vol_score,
                "confirm": confirm_long - confirm_short,
            },
            metadata={"adx_value": adx, "vol_ratio": vol_ratio},
        )
```

- [ ] **Step 4: 写 `rabbit_hunter/scoring_engine/strategies/__init__.py`**

```python
from .trend_following import TrendFollowing, TFParams

__all__ = ["TrendFollowing", "TFParams"]
```

- [ ] **Step 5: 跑测试确认通过**

Run: `pytest tests/scoring_engine/test_trend_following.py -v`
Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
git add rabbit_hunter/scoring_engine tests/scoring_engine
git commit -m "feat(scoring): trend_following strategy v0.1.0"
```

---

## Task 12: Strategy Router · weighted_avg composer

**Files:**
- Create: `rabbit_hunter/strategy_router/__init__.py`
- Create: `rabbit_hunter/strategy_router/router.py`
- Create: `tests/strategy_router/__init__.py`
- Create: `tests/strategy_router/test_router.py`

**Interfaces:**
- Consumes:
  - `AppConfig`（Task 2 产物）
  - 策略实例集合（Task 11 产物 + 未来 1b/1c）
  - `features_row: dict`, `features_history: pd.DataFrame`
- Produces:
  - `Intent` dataclass: `symbol`, `action: Literal['open_long','open_short','close','wait']`, `conviction: float`, `contributing_strategies: dict[str,float]`, `features_snapshot: dict`
  - `StrategyRouter.__init__(app_config, strategy_configs: dict[str,dict], strategies: list[BaseStrategy])`
  - `.route(symbol: str, features_row, features_history, open_action_threshold=0.5) -> Intent`

- [ ] **Step 1: 写失败测试 `tests/strategy_router/test_router.py`**

```python
from dataclasses import dataclass
import pandas as pd
from rabbit_hunter.scoring_engine.base import BaseStrategy, ScoreOutput
from rabbit_hunter.strategy_router.router import StrategyRouter, Intent


class StubLong(BaseStrategy):
    name = "stub_long"; version = "0.1"
    def score(self, fr, fh): return ScoreOutput(long=0.8, short=0.1)

class StubShort(BaseStrategy):
    name = "stub_short"; version = "0.1"
    def score(self, fr, fh): return ScoreOutput(long=0.1, short=0.7)

class StubWait(BaseStrategy):
    name = "stub_wait"; version = "0.1"
    def score(self, fr, fh): return ScoreOutput(long=0.2, short=0.2)


def _weights(mapping):
    return {name: {"weight": w} for name, w in mapping.items()}


def test_weighted_avg_long_wins():
    r = StrategyRouter(
        composer="weighted_avg",
        strategy_weights=_weights({"stub_long": 1.0, "stub_wait": 1.0}),
        strategies=[StubLong(), StubWait()],
    )
    intent = r.route("BTC-USDT-SWAP", {"close": 100}, pd.DataFrame(), open_action_threshold=0.4)
    assert intent.action == "open_long"
    assert intent.symbol == "BTC-USDT-SWAP"
    assert 0.4 <= intent.conviction <= 1.0
    assert intent.contributing_strategies == {"stub_long": 0.8, "stub_wait": 0.2}


def test_weighted_avg_short_wins():
    r = StrategyRouter(
        composer="weighted_avg",
        strategy_weights=_weights({"stub_short": 1.0}),
        strategies=[StubShort()],
    )
    intent = r.route("ETH-USDT-SWAP", {"close": 100}, pd.DataFrame(), open_action_threshold=0.5)
    assert intent.action == "open_short"


def test_below_threshold_returns_wait():
    r = StrategyRouter(
        composer="weighted_avg",
        strategy_weights=_weights({"stub_wait": 1.0}),
        strategies=[StubWait()],
    )
    intent = r.route("BTC-USDT-SWAP", {"close": 100}, pd.DataFrame(), open_action_threshold=0.5)
    assert intent.action == "wait"


def test_features_snapshot_passed_through():
    r = StrategyRouter(
        composer="weighted_avg",
        strategy_weights=_weights({"stub_long": 1.0}),
        strategies=[StubLong()],
    )
    intent = r.route("BTC-USDT-SWAP", {"close": 100, "ema20": 105}, pd.DataFrame(), open_action_threshold=0.5)
    assert intent.features_snapshot["ema20"] == 105
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/strategy_router -v`
Expected: FAIL

- [ ] **Step 3: 写 `rabbit_hunter/strategy_router/router.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal
import pandas as pd
from rabbit_hunter.scoring_engine.base import BaseStrategy, ScoreOutput


Action = Literal["open_long", "open_short", "close", "wait"]


@dataclass(frozen=True)
class Intent:
    symbol: str
    action: Action
    conviction: float
    contributing_strategies: dict[str, float] = field(default_factory=dict)
    features_snapshot: dict = field(default_factory=dict)
    score_components: dict = field(default_factory=dict)


class StrategyRouter:
    def __init__(
        self,
        composer: str,
        strategy_weights: dict[str, dict],
        strategies: list[BaseStrategy],
    ):
        if composer != "weighted_avg":
            # Phase 1a 只实现 weighted_avg；其他 composer 留给 1b/1c
            raise NotImplementedError(f"composer {composer} not implemented in Phase 1a")
        self.composer = composer
        self.weights = {name: float(cfg["weight"]) for name, cfg in strategy_weights.items()}
        self.strategies = [s for s in strategies if s.name in self.weights]

    def route(
        self,
        symbol: str,
        features_row: dict,
        features_history: pd.DataFrame,
        open_action_threshold: float = 0.5,
    ) -> Intent:
        outputs: dict[str, ScoreOutput] = {}
        for s in self.strategies:
            outputs[s.name] = s.score(features_row, features_history)

        total_w = sum(self.weights[n] for n in outputs) or 1.0
        long_sum = sum(outputs[n].long * self.weights[n] for n in outputs) / total_w
        short_sum = sum(outputs[n].short * self.weights[n] for n in outputs) / total_w

        contributing = {n: (outputs[n].long - outputs[n].short) for n in outputs}
        # 简化：contributing 展示每个策略的 long 分（trades.parquet 用）
        contributing_display = {n: outputs[n].long for n in outputs}

        if long_sum >= open_action_threshold and long_sum > short_sum:
            action: Action = "open_long"
            conviction = long_sum
        elif short_sum >= open_action_threshold and short_sum > long_sum:
            action = "open_short"
            conviction = short_sum
        else:
            action = "wait"
            conviction = max(long_sum, short_sum)

        # 合并所有策略的 components
        merged_components: dict[str, Any] = {}
        for n, out in outputs.items():
            for k, v in out.components.items():
                merged_components[f"{n}.{k}"] = v

        return Intent(
            symbol=symbol,
            action=action,
            conviction=conviction,
            contributing_strategies=contributing_display,
            features_snapshot=dict(features_row),
            score_components=merged_components,
        )
```

- [ ] **Step 4: 写 `rabbit_hunter/strategy_router/__init__.py`**

```python
from .router import StrategyRouter, Intent

__all__ = ["StrategyRouter", "Intent"]
```

- [ ] **Step 5: 跑测试确认通过**

Run: `pytest tests/strategy_router -v`
Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
git add rabbit_hunter/strategy_router tests/strategy_router
git commit -m "feat(router): weighted_avg composer + Intent"
```

---

## Task 13: Risk Engine · Position sizing + Daily circuit

**Files:**
- Create: `rabbit_hunter/risk_engine/__init__.py`
- Create: `rabbit_hunter/risk_engine/position_sizing.py`
- Create: `rabbit_hunter/risk_engine/daily_circuit.py`
- Create: `rabbit_hunter/risk_engine/engine.py`
- Create: `tests/risk_engine/__init__.py`
- Create: `tests/risk_engine/test_engine.py`

**Interfaces:**
- Consumes:
  - `Intent`（Task 12 产物）
  - `equity: float`（当前账户净值）
  - `atr: float`（当前 ATR_14）
  - `price: float`（当前价）
  - `RiskConfig`（Task 2 产物）
  - `daily_stats: DailyStats`：`realized_pnl_today`, `date`
- Produces:
  - `Order` dataclass: `symbol`, `side: Literal['long','short']`, `entry_price`, `stop_price`, `take_profit_price`, `size`, `leverage`
  - `RiskEngine.size(intent, ctx: RiskContext) -> Order | None`
  - `RiskContext` dataclass: `equity`, `atr`, `price`, `daily_realized_pnl`, `initial_capital`, `open_positions_count`

- [ ] **Step 1: 写失败测试 `tests/risk_engine/test_engine.py`**

```python
import math
from rabbit_hunter.config.schema import RiskConfig
from rabbit_hunter.risk_engine.engine import RiskEngine, RiskContext, Order
from rabbit_hunter.strategy_router.router import Intent


def _cfg():
    return RiskConfig(
        risk_per_trade_pct=1.0,
        atr_stop_multiplier=1.5,
        reward_risk_ratio=2.0,
        max_leverage=3,
        daily_max_loss_pct=3.0,
        hold_timeout_bars=48,
    )


def _intent(action="open_long"):
    return Intent(symbol="BTC-USDT-SWAP", action=action, conviction=0.7)


def _ctx(**kw):
    base = dict(equity=10_000.0, atr=100.0, price=50_000.0,
                daily_realized_pnl=0.0, initial_capital=10_000.0, open_positions_count=0)
    base.update(kw)
    return RiskContext(**base)


def test_long_order_sizing():
    e = RiskEngine(_cfg())
    order = e.size(_intent("open_long"), _ctx())
    assert isinstance(order, Order)
    assert order.side == "long"
    assert order.entry_price == 50_000.0
    assert math.isclose(order.stop_price, 50_000.0 - 1.5 * 100.0)
    assert math.isclose(order.take_profit_price, 50_000.0 + 2.0 * 1.5 * 100.0)
    # size = risk / stop_distance = 10000 * 0.01 / 150 ≈ 0.6667
    assert math.isclose(order.size, 100.0 / 150.0, rel_tol=1e-6)
    assert order.leverage <= 3


def test_short_order_sizing():
    e = RiskEngine(_cfg())
    order = e.size(_intent("open_short"), _ctx())
    assert order.side == "short"
    assert math.isclose(order.stop_price, 50_000.0 + 1.5 * 100.0)
    assert math.isclose(order.take_profit_price, 50_000.0 - 2.0 * 1.5 * 100.0)


def test_wait_returns_none():
    e = RiskEngine(_cfg())
    assert e.size(_intent("wait"), _ctx()) is None


def test_daily_circuit_blocks():
    e = RiskEngine(_cfg())
    ctx = _ctx(daily_realized_pnl=-400.0)  # 4% loss on 10000
    assert e.size(_intent("open_long"), ctx) is None


def test_leverage_cap():
    e = RiskEngine(_cfg())
    # ATR 很小 → 名义仓位 = size * price 可能 > equity * max_leverage → 应该被 cap
    order = e.size(_intent("open_long"), _ctx(atr=1.0))
    assert order.leverage <= 3
    assert order.size * order.entry_price <= 3 * 10_000.0 + 1e-6
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/risk_engine -v`
Expected: FAIL

- [ ] **Step 3: 写 `rabbit_hunter/risk_engine/position_sizing.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    stop_price: float
    take_profit_price: float
    size: float
    leverage: float


def compute_order(
    symbol: str,
    side: Literal["long", "short"],
    price: float,
    atr: float,
    equity: float,
    risk_per_trade_pct: float,
    atr_stop_multiplier: float,
    reward_risk_ratio: float,
    max_leverage: float,
) -> Order:
    stop_distance = atr_stop_multiplier * atr
    if side == "long":
        stop = price - stop_distance
        tp = price + reward_risk_ratio * stop_distance
    else:
        stop = price + stop_distance
        tp = price - reward_risk_ratio * stop_distance

    risk_amount = equity * (risk_per_trade_pct / 100.0)
    size = risk_amount / stop_distance if stop_distance > 0 else 0.0

    notional = size * price
    max_notional = equity * max_leverage
    if notional > max_notional:
        size = max_notional / price
        notional = size * price
    leverage = notional / equity if equity > 0 else 0.0

    return Order(
        symbol=symbol, side=side, entry_price=price,
        stop_price=stop, take_profit_price=tp, size=size, leverage=leverage,
    )
```

- [ ] **Step 4: 写 `rabbit_hunter/risk_engine/daily_circuit.py`**

```python
from __future__ import annotations


def daily_loss_tripped(
    daily_realized_pnl: float,
    initial_capital: float,
    daily_max_loss_pct: float,
) -> bool:
    max_loss = initial_capital * (daily_max_loss_pct / 100.0)
    return daily_realized_pnl <= -max_loss
```

- [ ] **Step 5: 写 `rabbit_hunter/risk_engine/engine.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from rabbit_hunter.config.schema import RiskConfig
from rabbit_hunter.strategy_router.router import Intent
from .position_sizing import compute_order, Order
from .daily_circuit import daily_loss_tripped


@dataclass(frozen=True)
class RiskContext:
    equity: float
    atr: float
    price: float
    daily_realized_pnl: float
    initial_capital: float
    open_positions_count: int


class RiskEngine:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def size(self, intent: Intent, ctx: RiskContext) -> Order | None:
        if intent.action not in ("open_long", "open_short"):
            return None
        if daily_loss_tripped(ctx.daily_realized_pnl, ctx.initial_capital, self.cfg.daily_max_loss_pct):
            return None
        if ctx.atr <= 0 or ctx.price <= 0:
            return None
        side = "long" if intent.action == "open_long" else "short"
        return compute_order(
            symbol=intent.symbol,
            side=side,
            price=ctx.price,
            atr=ctx.atr,
            equity=ctx.equity,
            risk_per_trade_pct=self.cfg.risk_per_trade_pct,
            atr_stop_multiplier=self.cfg.atr_stop_multiplier,
            reward_risk_ratio=self.cfg.reward_risk_ratio,
            max_leverage=self.cfg.max_leverage,
        )
```

- [ ] **Step 6: 写 `rabbit_hunter/risk_engine/__init__.py`**

```python
from .engine import RiskEngine, RiskContext
from .position_sizing import Order

__all__ = ["RiskEngine", "RiskContext", "Order"]
```

- [ ] **Step 7: 跑测试确认通过**

Run: `pytest tests/risk_engine -v`
Expected: 5 PASS

- [ ] **Step 8: Commit**

```bash
git add rabbit_hunter/risk_engine tests/risk_engine
git commit -m "feat(risk): ATR sizing + daily loss circuit"
```

---

## Task 14: Execution · BacktestExecutor

**Files:**
- Create: `rabbit_hunter/execution_engine/__init__.py`
- Create: `rabbit_hunter/execution_engine/base.py`
- Create: `rabbit_hunter/execution_engine/backtest_executor.py`
- Create: `rabbit_hunter/execution_engine/paper_executor.py`
- Create: `tests/execution_engine/__init__.py`
- Create: `tests/execution_engine/test_backtest_executor.py`

**Interfaces:**
- Consumes:
  - `Order`（Task 13 产物）
  - `next_bar: dict`（下一根 K 线的 `open/high/low/close/timestamp`）
  - `ExecutionConfig`（Task 2 产物）
- Produces:
  - `Fill` dataclass: `symbol`, `side`, `fill_price`, `size`, `timestamp`, `fees`, `slippage`
  - `BacktestExecutor.submit(order, next_bar, atr) -> Fill`
  - `BacktestExecutor.close_at(symbol, side, size, price, timestamp, atr, is_taker=True) -> Fill`
  - `BacktestExecutor.apply_funding(position_size, price, funding_rate) -> float`
- `PaperExecutor`: 空壳（raise `NotImplementedError`），Phase 1 只保接口

- [ ] **Step 1: 写失败测试 `tests/execution_engine/test_backtest_executor.py`**

```python
import math
from rabbit_hunter.config.schema import ExecutionConfig, FeeConfig
from rabbit_hunter.execution_engine.backtest_executor import BacktestExecutor, Fill
from rabbit_hunter.risk_engine.position_sizing import Order


def _cfg():
    return ExecutionConfig(
        fees=FeeConfig(maker=0.0002, taker=0.0005),
        slippage_atr_multiplier=0.1,
        funding_settlement=True,
    )


def _order(side="long", price=50_000.0, size=0.1):
    stop = price - 150 if side == "long" else price + 150
    tp = price + 300 if side == "long" else price - 300
    return Order(symbol="BTC-USDT-SWAP", side=side, entry_price=price,
                 stop_price=stop, take_profit_price=tp, size=size, leverage=1.0)


def test_long_fill_price_includes_slippage():
    e = BacktestExecutor(_cfg())
    next_bar = {"timestamp": 1, "open": 50_000.0, "high": 50_100.0, "low": 49_900.0, "close": 50_050.0}
    fill = e.submit(_order("long"), next_bar, atr=100.0)
    # 多头买入 → 成交价 = open + slippage_atr_multiplier * atr
    assert math.isclose(fill.fill_price, 50_000.0 + 0.1 * 100.0)
    assert fill.side == "long"
    # taker 费率
    assert math.isclose(fill.fees, 50_010.0 * 0.1 * 0.0005, rel_tol=1e-6)


def test_short_fill_price_slips_down():
    e = BacktestExecutor(_cfg())
    next_bar = {"timestamp": 1, "open": 50_000.0, "high": 50_100.0, "low": 49_900.0, "close": 49_950.0}
    fill = e.submit(_order("short"), next_bar, atr=100.0)
    assert math.isclose(fill.fill_price, 50_000.0 - 0.1 * 100.0)


def test_apply_funding_long_pays_when_positive():
    e = BacktestExecutor(_cfg())
    # 多头持仓，funding 为正 → 多头付钱（负 pnl）
    delta = e.apply_funding(position_size=0.1, price=50_000.0, funding_rate=0.0001)
    assert delta < 0
    assert math.isclose(delta, -0.1 * 50_000.0 * 0.0001)


def test_apply_funding_short_receives_when_positive():
    e = BacktestExecutor(_cfg())
    delta = e.apply_funding(position_size=-0.1, price=50_000.0, funding_rate=0.0001)
    assert delta > 0
    assert math.isclose(delta, 0.1 * 50_000.0 * 0.0001)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/execution_engine -v`
Expected: FAIL

- [ ] **Step 3: 写 `rabbit_hunter/execution_engine/base.py`**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: Literal["long", "short"]
    fill_price: float
    size: float
    timestamp: int
    fees: float
    slippage: float
    reason: str = "entry"


class BaseExecutor(ABC):
    @abstractmethod
    def submit(self, order, next_bar: dict, atr: float) -> Fill: ...
```

- [ ] **Step 4: 写 `rabbit_hunter/execution_engine/backtest_executor.py`**

```python
from __future__ import annotations
from typing import Literal
from rabbit_hunter.config.schema import ExecutionConfig
from rabbit_hunter.risk_engine.position_sizing import Order
from .base import BaseExecutor, Fill


class BacktestExecutor(BaseExecutor):
    def __init__(self, cfg: ExecutionConfig):
        self.cfg = cfg

    def _slippage(self, atr: float) -> float:
        return self.cfg.slippage_atr_multiplier * atr

    def submit(self, order: Order, next_bar: dict, atr: float) -> Fill:
        open_price = float(next_bar["open"])
        slip = self._slippage(atr)
        if order.side == "long":
            fill_price = open_price + slip
        else:
            fill_price = open_price - slip
        notional = fill_price * order.size
        fees = notional * self.cfg.fees.taker
        return Fill(
            symbol=order.symbol,
            side=order.side,
            fill_price=fill_price,
            size=order.size,
            timestamp=int(next_bar["timestamp"]),
            fees=fees,
            slippage=slip,
            reason="entry",
        )

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
        slip = self._slippage(atr)
        # 平多 = 卖 → 减滑点；平空 = 买 → 加滑点
        fill_price = price - slip if side == "long" else price + slip
        rate = self.cfg.fees.taker if is_taker else self.cfg.fees.maker
        fees = fill_price * size * rate
        return Fill(
            symbol=symbol,
            side=side,
            fill_price=fill_price,
            size=size,
            timestamp=timestamp,
            fees=fees,
            slippage=slip,
            reason=reason,
        )

    def apply_funding(self, position_size: float, price: float, funding_rate: float) -> float:
        """position_size > 0 表示多头，< 0 表示空头。返回资金费带来的 pnl delta（多头付/收）。"""
        if not self.cfg.funding_settlement or funding_rate is None:
            return 0.0
        # 惯例：funding 为正 → 多头付空头
        return -position_size * price * funding_rate
```

- [ ] **Step 5: 写 `rabbit_hunter/execution_engine/paper_executor.py`（桩）**

```python
from __future__ import annotations
from .base import BaseExecutor, Fill


class PaperExecutor(BaseExecutor):
    def submit(self, order, next_bar, atr: float) -> Fill:
        raise NotImplementedError("PaperExecutor is a Phase 1b+ deliverable")
```

- [ ] **Step 6: 写 `rabbit_hunter/execution_engine/__init__.py`**

```python
from .base import BaseExecutor, Fill
from .backtest_executor import BacktestExecutor
from .paper_executor import PaperExecutor

__all__ = ["BaseExecutor", "Fill", "BacktestExecutor", "PaperExecutor"]
```

- [ ] **Step 7: 跑测试确认通过**

Run: `pytest tests/execution_engine -v`
Expected: 4 PASS

- [ ] **Step 8: Commit**

```bash
git add rabbit_hunter/execution_engine tests/execution_engine
git commit -m "feat(exec): BacktestExecutor with slippage/fees/funding + PaperExecutor stub"
```

---

## Task 15: Ledger + Backtest main loop

**Files:**
- Create: `rabbit_hunter/backtest/__init__.py`
- Create: `rabbit_hunter/backtest/ledger.py`
- Create: `rabbit_hunter/backtest/engine.py`
- Create: `tests/backtest/__init__.py`
- Create: `tests/backtest/test_ledger.py`
- Create: `tests/backtest/test_engine_small.py`

**Interfaces:**
- Produces:
  - `Position` dataclass: `symbol`, `side`, `entry_time`, `entry_price`, `size`, `stop`, `take_profit`, `entry_snapshot: dict`, `strategy_scores: dict`, `bars_held: int`
  - `Trade` dataclass: entry + exit 的合并记录，含所有列（详见 § 10.3 spec）
  - `Ledger`:
    - `.equity: float`
    - `.open_positions: dict[symbol, Position]`
    - `.closed_trades: list[Trade]`
    - `.record_entry(fill, position_meta)` 打开仓位
    - `.record_exit(fill, exit_reason)` 平仓，追加 Trade
    - `.check_exits(symbol, bar, atr) -> list[Trade]` 遍历该 symbol 未平仓，触发止损/止盈/超时
    - `.apply_funding(fills_delta_per_symbol: dict[str,float])`
    - `.mark_to_market(prices: dict[str,float]) -> float`
  - `BacktestEngine.run(...) -> tuple[Ledger, snapshots_df]`

- [ ] **Step 1: 写失败测试 `tests/backtest/test_ledger.py`**

```python
import math
from rabbit_hunter.backtest.ledger import Ledger, Position
from rabbit_hunter.execution_engine.backtest_executor import BacktestExecutor
from rabbit_hunter.config.schema import ExecutionConfig, FeeConfig
from rabbit_hunter.execution_engine.base import Fill


def _ex():
    return BacktestExecutor(ExecutionConfig(
        fees=FeeConfig(maker=0.0002, taker=0.0005),
        slippage_atr_multiplier=0.1,
        funding_settlement=True,
    ))


def test_open_and_close_long_produces_trade():
    ledger = Ledger(initial_capital=10_000.0)
    entry = Fill("BTC-USDT-SWAP", "long", 50_000.0, 0.01, 1000, fees=25.0, slippage=10.0, reason="entry")
    ledger.record_entry(entry, entry_snapshot={"ema20": 100.0}, strategy_scores={"trend_following": 0.7},
                        stop=49_800.0, take_profit=50_400.0)
    assert "BTC-USDT-SWAP" in ledger.open_positions

    exit_ = Fill("BTC-USDT-SWAP", "long", 50_300.0, 0.01, 2000, fees=25.15, slippage=10.0, reason="take_profit")
    trade = ledger.record_exit(exit_, exit_snapshot={"ema20": 105.0})
    assert trade is not None
    assert trade["side"] == "long"
    assert trade["exit_reason"] == "take_profit"
    assert math.isclose(trade["pnl_raw"], (50_300.0 - 50_000.0) * 0.01)
    assert math.isclose(trade["pnl_after_fees"], trade["pnl_raw"] - trade["fees"])
    assert "BTC-USDT-SWAP" not in ledger.open_positions


def test_stop_loss_triggers_on_long():
    ledger = Ledger(initial_capital=10_000.0)
    entry = Fill("BTC-USDT-SWAP", "long", 50_000.0, 0.01, 1000, fees=25.0, slippage=10.0, reason="entry")
    ledger.record_entry(entry, entry_snapshot={}, strategy_scores={},
                        stop=49_800.0, take_profit=50_400.0)
    ex = _ex()
    bar = {"timestamp": 2000, "open": 50_000.0, "high": 50_050.0, "low": 49_700.0, "close": 49_750.0}
    trades = ledger.check_exits("BTC-USDT-SWAP", bar, atr=50.0, executor=ex, hold_timeout_bars=10, exit_snapshot_fn=lambda: {})
    assert len(trades) == 1
    assert trades[0]["exit_reason"] == "stop_loss"


def test_timeout_exit_at_close():
    ledger = Ledger(initial_capital=10_000.0)
    entry = Fill("BTC-USDT-SWAP", "long", 50_000.0, 0.01, 1000, fees=25.0, slippage=10.0, reason="entry")
    ledger.record_entry(entry, entry_snapshot={}, strategy_scores={},
                        stop=49_000.0, take_profit=51_000.0)
    ex = _ex()
    for i in range(11):
        bar = {"timestamp": 2000 + i, "open": 50_000.0, "high": 50_010.0, "low": 49_990.0, "close": 50_005.0}
        trades = ledger.check_exits("BTC-USDT-SWAP", bar, atr=10.0, executor=ex, hold_timeout_bars=10, exit_snapshot_fn=lambda: {})
    assert any(t["exit_reason"] == "timeout" for t in ledger.closed_trades)
```

- [ ] **Step 2: 写 `rabbit_hunter/backtest/ledger.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal
from rabbit_hunter.execution_engine.base import Fill
from rabbit_hunter.execution_engine.backtest_executor import BacktestExecutor


@dataclass
class Position:
    symbol: str
    side: Literal["long", "short"]
    entry_time: int
    entry_price: float
    size: float
    fees_paid: float
    stop: float
    take_profit: float
    entry_snapshot: dict
    strategy_scores: dict
    bars_held: int = 0
    funding_accum: float = 0.0


@dataclass
class Ledger:
    initial_capital: float
    equity: float = 0.0
    open_positions: dict[str, Position] = field(default_factory=dict)
    closed_trades: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.equity == 0.0:
            self.equity = self.initial_capital

    def record_entry(
        self,
        fill: Fill,
        entry_snapshot: dict,
        strategy_scores: dict,
        stop: float,
        take_profit: float,
    ):
        pos = Position(
            symbol=fill.symbol,
            side=fill.side,
            entry_time=fill.timestamp,
            entry_price=fill.fill_price,
            size=fill.size,
            fees_paid=fill.fees,
            stop=stop,
            take_profit=take_profit,
            entry_snapshot=entry_snapshot,
            strategy_scores=strategy_scores,
        )
        self.open_positions[fill.symbol] = pos
        self.equity -= fill.fees

    def record_exit(self, fill: Fill, exit_snapshot: dict) -> dict:
        pos = self.open_positions.pop(fill.symbol)
        if pos.side == "long":
            pnl_raw = (fill.fill_price - pos.entry_price) * pos.size
        else:
            pnl_raw = (pos.entry_price - fill.fill_price) * pos.size
        fees_total = pos.fees_paid + fill.fees
        pnl_after = pnl_raw + pos.funding_accum - fill.fees  # 入场手续费已从 equity 扣，出场再扣一次
        self.equity += pnl_raw - fill.fees + pos.funding_accum
        trade = {
            "symbol": pos.symbol,
            "side": pos.side,
            "entry_time": pos.entry_time,
            "exit_time": fill.timestamp,
            "entry_price": pos.entry_price,
            "exit_price": fill.fill_price,
            "size": pos.size,
            "pnl_raw": pnl_raw,
            "pnl_after_fees": pnl_raw - fees_total + pos.funding_accum,
            "fees": fees_total,
            "funding": pos.funding_accum,
            "slippage": fill.slippage,
            "hold_bars": pos.bars_held,
            "exit_reason": fill.reason,
            "entry_snapshot": pos.entry_snapshot,
            "exit_snapshot": exit_snapshot,
            "strategy_scores": pos.strategy_scores,
        }
        self.closed_trades.append(trade)
        return trade

    def check_exits(
        self,
        symbol: str,
        bar: dict,
        atr: float,
        executor: BacktestExecutor,
        hold_timeout_bars: int,
        exit_snapshot_fn: Callable[[], dict],
    ) -> list[dict]:
        results: list[dict] = []
        if symbol not in self.open_positions:
            return results
        pos = self.open_positions[symbol]
        pos.bars_held += 1

        high = float(bar["high"]); low = float(bar["low"]); close = float(bar["close"])
        ts = int(bar["timestamp"])

        # 检测触发
        if pos.side == "long":
            hit_stop = low <= pos.stop
            hit_tp = high >= pos.take_profit
            price = pos.stop if hit_stop else (pos.take_profit if hit_tp else close)
        else:
            hit_stop = high >= pos.stop
            hit_tp = low <= pos.take_profit
            price = pos.stop if hit_stop else (pos.take_profit if hit_tp else close)

        reason = None
        if hit_stop and hit_tp:
            reason = "stop_loss"  # 保守：同 bar 内两者都触发 → 假设先触发止损
        elif hit_stop:
            reason = "stop_loss"
        elif hit_tp:
            reason = "take_profit"
        elif pos.bars_held >= hold_timeout_bars:
            reason = "timeout"; price = close

        if reason:
            fill = executor.close_at(symbol, pos.side, pos.size, price, ts, atr, reason)
            trade = self.record_exit(fill, exit_snapshot_fn())
            results.append(trade)
        return results

    def apply_funding(self, symbol: str, price: float, funding_rate: float, executor: BacktestExecutor):
        if symbol not in self.open_positions:
            return
        pos = self.open_positions[symbol]
        signed_size = pos.size if pos.side == "long" else -pos.size
        delta = executor.apply_funding(signed_size, price, funding_rate)
        pos.funding_accum += delta
        self.equity += delta

    def mark_to_market(self, prices: dict[str, float]) -> float:
        eq = self.equity
        for sym, pos in self.open_positions.items():
            if sym not in prices:
                continue
            p = prices[sym]
            if pos.side == "long":
                eq += (p - pos.entry_price) * pos.size
            else:
                eq += (pos.entry_price - p) * pos.size
        return eq
```

- [ ] **Step 3: 写 `rabbit_hunter/backtest/engine.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import pandas as pd
from rabbit_hunter.config.schema import AppConfig
from rabbit_hunter.scoring_engine.base import BaseStrategy
from rabbit_hunter.strategy_router.router import StrategyRouter
from rabbit_hunter.risk_engine.engine import RiskEngine, RiskContext
from rabbit_hunter.execution_engine.backtest_executor import BacktestExecutor
from .ledger import Ledger


@dataclass
class BacktestResult:
    ledger: Ledger
    snapshots: pd.DataFrame  # 每次决策一行
    equity_curve: pd.DataFrame  # timestamp, equity


class BacktestEngine:
    def __init__(
        self,
        app_config: AppConfig,
        strategies: list[BaseStrategy],
    ):
        self.cfg = app_config
        strategy_weights = {
            name: {"weight": entry.weight}
            for name, entry in app_config.strategy_router.enabled_strategies.items()
        }
        self.router = StrategyRouter(
            composer=app_config.strategy_router.composer,
            strategy_weights=strategy_weights,
            strategies=strategies,
        )
        self.risk = RiskEngine(app_config.risk)
        self.executor = BacktestExecutor(app_config.execution)

    def run(
        self,
        features_by_symbol: dict[str, pd.DataFrame],
        open_action_threshold: float = 0.5,
    ) -> BacktestResult:
        ledger = Ledger(initial_capital=self.cfg.backtest.initial_capital)

        # 合并所有 symbol 的 timestamps 并按顺序处理
        all_ts = sorted({int(ts) for df in features_by_symbol.values() for ts in df["timestamp"]})

        # 建索引：{symbol: {ts: row_idx}}
        indexes = {sym: {int(ts): i for i, ts in enumerate(df["timestamp"])} for sym, df in features_by_symbol.items()}

        snapshots: list[dict] = []
        equity_points: list[dict] = []
        daily_realized: dict[str, float] = {}
        last_day: str | None = None

        for ts in all_ts:
            day_str = pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%d")
            if day_str != last_day:
                daily_realized[day_str] = 0.0
                last_day = day_str

            prices_at_ts: dict[str, float] = {}

            for symbol, feats in features_by_symbol.items():
                idx = indexes[symbol].get(ts)
                if idx is None or idx + 1 >= len(feats):
                    continue
                row = feats.iloc[idx].to_dict()
                next_bar = feats.iloc[idx + 1].to_dict()
                price = float(row["close"])
                atr = float(row.get("atr_14") or 0.0)
                prices_at_ts[symbol] = price

                # 结算 funding（每 8 小时）
                dt = pd.to_datetime(ts, unit="ms", utc=True)
                if dt.hour % 8 == 0 and dt.minute == 0:
                    fr = row.get("funding_rate")
                    if fr is not None and pd.notna(fr):
                        ledger.apply_funding(symbol, price, float(fr), self.executor)

                # 检查已有仓位是否触发止损/止盈/超时
                if symbol in ledger.open_positions:
                    exit_snapshot_fn = lambda r=row: r
                    trades = ledger.check_exits(
                        symbol=symbol,
                        bar=next_bar,
                        atr=atr,
                        executor=self.executor,
                        hold_timeout_bars=self.cfg.risk.hold_timeout_bars,
                        exit_snapshot_fn=exit_snapshot_fn,
                    )
                    for t in trades:
                        daily_realized[day_str] += t["pnl_after_fees"]

                # 如果已有该 symbol 仓位，不再开新单
                if symbol in ledger.open_positions:
                    continue

                intent = self.router.route(
                    symbol=symbol,
                    features_row=row,
                    features_history=feats.iloc[max(0, idx - 200): idx + 1],
                    open_action_threshold=open_action_threshold,
                )
                ctx = RiskContext(
                    equity=ledger.equity,
                    atr=atr,
                    price=price,
                    daily_realized_pnl=daily_realized[day_str],
                    initial_capital=self.cfg.backtest.initial_capital,
                    open_positions_count=len(ledger.open_positions),
                )
                order = self.risk.size(intent, ctx)

                snap = {
                    "timestamp": ts,
                    "symbol": symbol,
                    "action": intent.action,
                    "conviction": intent.conviction,
                    "long_score": intent.contributing_strategies,
                    "order_placed": order is not None,
                }
                snapshots.append(snap)

                if order is None:
                    continue
                fill = self.executor.submit(order, next_bar, atr)
                ledger.record_entry(
                    fill=fill,
                    entry_snapshot=row,
                    strategy_scores=intent.contributing_strategies,
                    stop=order.stop_price,
                    take_profit=order.take_profit_price,
                )

            eq_now = ledger.mark_to_market(prices_at_ts)
            equity_points.append({"timestamp": ts, "equity": eq_now})

        snap_df = pd.DataFrame(snapshots)
        eq_df = pd.DataFrame(equity_points)
        return BacktestResult(ledger=ledger, snapshots=snap_df, equity_curve=eq_df)
```

- [ ] **Step 4: 写 `rabbit_hunter/backtest/__init__.py`**

```python
from .engine import BacktestEngine, BacktestResult
from .ledger import Ledger, Position

__all__ = ["BacktestEngine", "BacktestResult", "Ledger", "Position"]
```

- [ ] **Step 5: 写小规模集成测试 `tests/backtest/test_engine_small.py`**

```python
import numpy as np
import pandas as pd
from rabbit_hunter.config.loader import load_config
from rabbit_hunter.config.schema import (
    AppConfig, DataConfig, FeatureEngineConfig, StrategyRouterConfig, StrategyEntry,
    RiskConfig, ExecutionConfig, FeeConfig, BacktestConfig, ReportConfig,
)
from rabbit_hunter.feature_engine.pipeline import build_features
from rabbit_hunter.scoring_engine.strategies.trend_following import TrendFollowing, TFParams
from rabbit_hunter.backtest.engine import BacktestEngine


def _cfg():
    return AppConfig(
        data=DataConfig(exchange="okx", symbols=["BTC-USDT-SWAP"],
                        main_interval="1H", confirm_interval="15m", history_window_days=30),
        feature_engine=FeatureEngineConfig(version="0.1.0"),
        strategy_router=StrategyRouterConfig(composer="weighted_avg",
                                             enabled_strategies={"trend_following": StrategyEntry(weight=1.0, config_file="strategies/trend_following.yaml")}),
        risk=RiskConfig(risk_per_trade_pct=1.0, atr_stop_multiplier=1.5, reward_risk_ratio=2.0,
                        max_leverage=3, daily_max_loss_pct=3.0, hold_timeout_bars=48),
        execution=ExecutionConfig(fees=FeeConfig(maker=0.0002, taker=0.0005),
                                  slippage_atr_multiplier=0.1, funding_settlement=True),
        backtest=BacktestConfig(start="2025-01-01", end="2025-02-01", initial_capital=10_000.0),
        report=ReportConfig(),
    )


def _mk_uptrend_df(n=400, base=100.0):
    ts = [i * 3_600_000 for i in range(n)]
    close = np.linspace(base, base + 100, n)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 100.0),
        "funding_rate": np.full(n, 0.0001),
        "oi": np.linspace(1000, 1500, n),
    })


def test_backtest_produces_trades_and_equity_curve():
    cfg = _cfg()
    tf = TrendFollowing(TFParams(
        ema_fast=20, ema_slow=60, ema_trend=200,
        adx_threshold=25, volume_ratio_threshold=0.5,
        confirm_ema_fast=20, confirm_adx_threshold=15,
    ))
    engine = BacktestEngine(cfg, [tf])
    raw = _mk_uptrend_df()
    feats = build_features(raw)
    result = engine.run({"BTC-USDT-SWAP": feats}, open_action_threshold=0.3)

    assert len(result.equity_curve) > 100
    assert result.equity_curve["equity"].iloc[-1] > 0
    # 明显趋势中至少能触发一次开仓
    assert len(result.snapshots) > 0
```

- [ ] **Step 6: 跑测试确认通过**

Run: `pytest tests/backtest -v`
Expected: 4 PASS

- [ ] **Step 7: Commit**

```bash
git add rabbit_hunter/backtest tests/backtest
git commit -m "feat(backtest): ledger + main loop with funding/exits/equity curve"
```

---

## Task 16: Observability · structlog + Snapshot writer

**Files:**
- Create: `rabbit_hunter/observability/__init__.py`
- Create: `rabbit_hunter/observability/logger.py`
- Create: `rabbit_hunter/observability/snapshot.py`
- Create: `tests/observability/__init__.py`
- Create: `tests/observability/test_snapshot.py`

**Interfaces:**
- Produces:
  - `configure_logger(level="INFO")` — JSON 输出到 stderr
  - `get_logger(name: str) -> structlog.BoundLogger`
  - `SnapshotWriter(root: Path)`
    - `.append(records: list[dict])` — 累积
    - `.flush() -> Path` — 按天分区写 Parquet，返回已写路径列表

- [ ] **Step 1: 写失败测试 `tests/observability/test_snapshot.py`**

```python
from pathlib import Path
import pandas as pd
from rabbit_hunter.observability.snapshot import SnapshotWriter


def test_snapshot_writer_partitions_by_day(tmp_path):
    w = SnapshotWriter(root=tmp_path)
    records = [
        {"timestamp": 1_700_000_000_000, "symbol": "BTC-USDT-SWAP", "action": "wait",
         "conviction": 0.2, "long_score": {}, "order_placed": False},
        {"timestamp": 1_700_086_400_000, "symbol": "BTC-USDT-SWAP", "action": "open_long",
         "conviction": 0.7, "long_score": {"trend_following": 0.7}, "order_placed": True},
    ]
    w.append(records)
    paths = w.flush()
    assert len(paths) == 2  # 两天两文件
    df = pd.concat([pd.read_parquet(p) for p in paths])
    assert len(df) == 2
    assert set(df["action"]) == {"wait", "open_long"}


def test_snapshot_writer_no_records_no_files(tmp_path):
    w = SnapshotWriter(root=tmp_path)
    paths = w.flush()
    assert paths == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/observability -v`
Expected: FAIL

- [ ] **Step 3: 写 `rabbit_hunter/observability/snapshot.py`**

```python
from __future__ import annotations
from pathlib import Path
import pandas as pd


class SnapshotWriter:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._buffer: list[dict] = []

    def append(self, records: list[dict]) -> None:
        self._buffer.extend(records)

    def flush(self) -> list[Path]:
        if not self._buffer:
            return []
        df = pd.DataFrame(self._buffer)
        df["_day"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
        written: list[Path] = []
        for day, grp in df.groupby("_day", sort=True):
            out = self.root / day / "snapshot.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            grp.drop(columns=["_day"]).to_parquet(out, index=False)
            written.append(out)
        self._buffer.clear()
        return written
```

- [ ] **Step 4: 写 `rabbit_hunter/observability/logger.py`**

```python
from __future__ import annotations
import logging
import structlog


def configure_logger(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.EventRenamer("event"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    return structlog.get_logger(name)
```

- [ ] **Step 5: 写 `rabbit_hunter/observability/__init__.py`**

```python
from .snapshot import SnapshotWriter
from .logger import configure_logger, get_logger

__all__ = ["SnapshotWriter", "configure_logger", "get_logger"]
```

- [ ] **Step 6: 跑测试确认通过**

Run: `pytest tests/observability -v`
Expected: 2 PASS

- [ ] **Step 7: Commit**

```bash
git add rabbit_hunter/observability tests/observability
git commit -m "feat(obs): structlog config + snapshot writer with day partition"
```

---

## Task 17: Report Layer · 全套输出

**Files:**
- Create: `rabbit_hunter/backtest/report.py`
- Create: `rabbit_hunter/backtest/templates/report.md.j2`
- Create: `rabbit_hunter/backtest/templates/ai_context.md.j2`
- Modify: `rabbit_hunter/backtest/__init__.py`
- Create: `tests/backtest/test_report.py`

**Interfaces:**
- Produces:
  - `ReportBuilder(config: AppConfig, features_by_symbol: dict[str,pd.DataFrame])`
    - `.build(result: BacktestResult, output_root: Path, git_commit: str = "unknown") -> Path`
    - 返回：报告目录 `output_root/{YYYY-MM-DD-HHMM}/`
    - 内含：`report.md`, `ai_context.md`, `trades.parquet`, `snapshots.parquet`, `charts/*.png`, `config_snapshot.yaml`
  - 内部函数：
    - `compute_stats(trades, equity_curve, initial_capital) -> dict`
    - `find_loss_clusters(trades) -> list[dict]`
    - `compute_baselines(features_by_symbol, initial_capital) -> list[dict]`（Buy-and-Hold per symbol）
    - `regime_conditional_performance(trades) -> pd.DataFrame`
    - `feature_correlation_with_pnl(trades) -> pd.DataFrame`（Spearman top 10）

- [ ] **Step 1: 写 `templates/report.md.j2`**

```jinja
# Rabbit Hunter 回测报告 · {{ run_id }}

## 元数据
- 回测区间: {{ start }} → {{ end }}
- 标的: {{ symbols | join(', ') }}
- 主周期: {{ main_interval }} 确认周期: {{ confirm_interval }}
- 启用策略: {{ strategies | join(', ') }}
- 配置 hash: `{{ config_hash }}`
- Git commit: `{{ git_commit }}`

## 收益概况
| 指标 | 数值 |
|---|---|
| 总收益率 | {{ stats.total_return_pct }}% |
| 年化收益率 | {{ stats.annualized_return_pct }}% |
| 最大回撤 | {{ stats.max_drawdown_pct }}% |
| 夏普比率 | {{ stats.sharpe }} |
| 交易笔数 | {{ stats.trade_count }} |
| 胜率 | {{ stats.win_rate_pct }}% |
| 盈亏比 | {{ stats.profit_factor }} |

![净值曲线](charts/equity_curve.png)

## 分标的表现
| Symbol | 收益率 | 胜率 | 交易笔数 |
|---|---|---|---|
{% for row in per_symbol -%}
| {{ row.symbol }} | {{ row.return_pct }}% | {{ row.win_rate_pct }}% | {{ row.count }} |
{% endfor %}

## 分策略贡献
| 策略 | 平均 long 分 | 触发次数 |
|---|---|---|
{% for row in per_strategy -%}
| {{ row.strategy }} | {{ row.avg_long }} | {{ row.trigger_count }} |
{% endfor %}

## 月度盈亏
![月度盈亏](charts/monthly_pnl.png)

## 亏损最大的 10 笔
| 时间 | Symbol | 方向 | 入场 | 出场 | PnL | 出场原因 |
|---|---|---|---|---|---|---|
{% for t in worst_trades -%}
| {{ t.entry_time }} | {{ t.symbol }} | {{ t.side }} | {{ t.entry_price }} | {{ t.exit_price }} | {{ t.pnl_after_fees }} | {{ t.exit_reason }} |
{% endfor %}

## 决策快照汇总
- 快照总数: {{ snapshot_count }}
- 快照文件: `snapshots.parquet`

## 配置全文
```yaml
{{ config_yaml }}
```
```

- [ ] **Step 2: 写 `templates/ai_context.md.j2`**

```jinja
# AI Review Context - {{ run_id }}

## Data Provenance
- 数据源: OKX REST 公共行情
- 时间窗口: {{ start }} → {{ end }}
- Feature Engine 版本: {{ feature_engine_version }}
- 手续费模型: maker={{ fees.maker }}, taker={{ fees.taker }}
- 滑点模型: {{ slippage_atr_multiplier }} × ATR_14

## Baseline Comparisons
| Baseline | 收益率 | 夏普 | 最大回撤 |
|---|---|---|---|
{% for b in baselines -%}
| {{ b.name }} | {{ b.total_return_pct }}% | {{ b.sharpe }} | {{ b.max_drawdown_pct }}% |
{% endfor %}
| 本策略 | {{ stats.total_return_pct }}% | {{ stats.sharpe }} | {{ stats.max_drawdown_pct }}% |

## Failure Mode Clusters
按 regime / session / day_of_week 分组，胜率 ≤ 40% 且交易数 ≥ 20 的桶：

{% if clusters -%}
| 维度 | 交易数 | 胜率 | 累计 PnL |
|---|---|---|---|
{% for c in clusters -%}
| {{ c.dim }} | {{ c.trades }} | {{ c.winrate_pct }}% | {{ c.total_pnl }} |
{% endfor %}
{% else -%}
（未发现符合阈值的集群）
{% endif %}

## Regime-Conditional Performance
| Regime | 交易数 | 收益率 | 胜率 | 平均 PnL |
|---|---|---|---|---|
{% for r in regime_perf -%}
| {{ r.regime }} | {{ r.count }} | {{ r.return_pct }}% | {{ r.win_rate_pct }}% | {{ r.avg_pnl }} |
{% endfor %}

## Feature Correlation with PnL (Spearman, top 10)
| 特征 | 相关性 |
|---|---|
{% for f in feature_corr -%}
| {{ f.feature }} | {{ f.rho }} |
{% endfor %}
```

- [ ] **Step 3: 写失败测试 `tests/backtest/test_report.py`**

```python
from pathlib import Path
import numpy as np
import pandas as pd
from rabbit_hunter.backtest.report import (
    compute_stats,
    find_loss_clusters,
    ReportBuilder,
)
from rabbit_hunter.backtest.engine import BacktestResult
from rabbit_hunter.backtest.ledger import Ledger
from rabbit_hunter.config.schema import (
    AppConfig, DataConfig, FeatureEngineConfig, StrategyRouterConfig, StrategyEntry,
    RiskConfig, ExecutionConfig, FeeConfig, BacktestConfig, ReportConfig,
)


def _cfg():
    return AppConfig(
        data=DataConfig(exchange="okx", symbols=["BTC-USDT-SWAP"],
                        main_interval="1H", confirm_interval="15m", history_window_days=30),
        feature_engine=FeatureEngineConfig(version="0.1.0"),
        strategy_router=StrategyRouterConfig(composer="weighted_avg",
                                             enabled_strategies={"trend_following": StrategyEntry(weight=1.0, config_file="strategies/trend_following.yaml")}),
        risk=RiskConfig(risk_per_trade_pct=1.0, atr_stop_multiplier=1.5, reward_risk_ratio=2.0,
                        max_leverage=3, daily_max_loss_pct=3.0, hold_timeout_bars=48),
        execution=ExecutionConfig(fees=FeeConfig(maker=0.0002, taker=0.0005),
                                  slippage_atr_multiplier=0.1, funding_settlement=True),
        backtest=BacktestConfig(start="2025-01-01", end="2025-02-01", initial_capital=10_000.0),
        report=ReportConfig(),
    )


def _mk_trades(n=30):
    return [
        {
            "symbol": "BTC-USDT-SWAP", "side": "long" if i % 2 == 0 else "short",
            "entry_time": i * 3_600_000, "exit_time": (i + 1) * 3_600_000,
            "entry_price": 50_000.0, "exit_price": 50_100.0 if i % 3 else 49_900.0,
            "size": 0.01, "pnl_raw": 1.0 if i % 3 else -1.0,
            "pnl_after_fees": 0.9 if i % 3 else -1.1,
            "fees": 0.1, "funding": 0.0, "slippage": 5.0, "hold_bars": 5,
            "exit_reason": "take_profit" if i % 3 else "stop_loss",
            "entry_snapshot": {"regime": "trending", "adx": 30.0, "rsi_14": 55},
            "exit_snapshot": {"regime": "trending", "adx": 28.0, "rsi_14": 60},
            "strategy_scores": {"trend_following": 0.7},
        }
        for i in range(n)
    ]


def test_compute_stats_returns_expected_keys():
    trades = _mk_trades()
    eq = pd.DataFrame({"timestamp": range(30), "equity": np.linspace(10000, 10100, 30)})
    stats = compute_stats(trades, eq, initial_capital=10_000.0)
    for k in ("total_return_pct", "sharpe", "max_drawdown_pct", "trade_count", "win_rate_pct", "profit_factor"):
        assert k in stats


def test_find_loss_clusters_finds_something_with_many_losses():
    losing_trades = [
        {**t, "pnl_after_fees": -1.0,
         "entry_snapshot": {"regime": "ranging", "adx": 15, "rsi_14": 75},
         "exit_snapshot": {"regime": "ranging", "adx": 15, "rsi_14": 75}}
        for t in _mk_trades(30)
    ]
    df = pd.DataFrame([{
        "regime": t["entry_snapshot"]["regime"],
        "session": "asia",
        "day_of_week": 0,
        "pnl_after_fees": t["pnl_after_fees"],
    } for t in losing_trades])
    clusters = find_loss_clusters(df, min_trades=10, max_winrate=0.5)
    assert len(clusters) > 0


def test_report_builder_writes_all_files(tmp_path):
    result = BacktestResult(
        ledger=Ledger(initial_capital=10_000.0, equity=10_050.0, closed_trades=_mk_trades()),
        snapshots=pd.DataFrame([{"timestamp": 0, "symbol": "BTC-USDT-SWAP",
                                 "action": "wait", "conviction": 0.1, "order_placed": False}]),
        equity_curve=pd.DataFrame({"timestamp": [0, 1, 2], "equity": [10_000.0, 10_020.0, 10_050.0]}),
    )
    fake_feats = pd.DataFrame({
        "timestamp": [0, 3_600_000, 7_200_000],
        "open": [50_000.0, 50_100.0, 50_050.0],
        "high": [50_050.0, 50_150.0, 50_100.0],
        "low": [49_950.0, 50_050.0, 50_000.0],
        "close": [50_000.0, 50_100.0, 50_050.0],
    })
    builder = ReportBuilder(_cfg(), {"BTC-USDT-SWAP": fake_feats})
    out_dir = builder.build(result, output_root=tmp_path, git_commit="test123")

    assert (out_dir / "report.md").exists()
    assert (out_dir / "ai_context.md").exists()
    assert (out_dir / "trades.parquet").exists()
    assert (out_dir / "snapshots.parquet").exists()
    assert (out_dir / "config_snapshot.yaml").exists()
    assert (out_dir / "charts" / "equity_curve.png").exists()
    assert (out_dir / "charts" / "monthly_pnl.png").exists()
```

- [ ] **Step 4: 写 `rabbit_hunter/backtest/report.py`**

```python
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
    if "regime_t0" not in trades_df.columns:
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
    if "pnl_after_fees" not in trades_df.columns:
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
```

- [ ] **Step 5: 在 `rabbit_hunter/backtest/__init__.py` 追加导出**

```python
from .report import ReportBuilder, compute_stats, find_loss_clusters
```

- [ ] **Step 6: 跑测试确认通过**

Run: `pytest tests/backtest/test_report.py -v`
Expected: 3 PASS

- [ ] **Step 7: Commit**

```bash
git add rabbit_hunter/backtest tests/backtest
git commit -m "feat(report): AI-learnable markdown + parquet + charts"
```

---

## Task 18: CLI（typer）

**Files:**
- Create: `rabbit_hunter/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Produces:
  - `rabbit fetch` — 拉配置里定义的 symbols 的 K 线 + funding + OI
  - `rabbit data quality` — 检查已下载数据的质量并输出 md 报告
  - `rabbit features build` — 计算并缓存全部 symbol × interval 的特征
  - `rabbit backtest` — 端到端跑回测，产出报告目录

CLI 层只做参数解析 + 组装依赖 + 调子系统，不含业务逻辑。

- [ ] **Step 1: 写失败测试 `tests/test_cli.py`**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL

- [ ] **Step 3: 写 `rabbit_hunter/cli.py`**

```python
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import typer
import pandas as pd
from rabbit_hunter.config.loader import load_config
from rabbit_hunter.observability.logger import configure_logger, get_logger

app = typer.Typer(help="Rabbit Hunter V5.1 Phase 1a CLI")
data_app = typer.Typer(help="Data engine commands")
app.add_typer(data_app, name="data")
features_app = typer.Typer(help="Feature engine commands")
app.add_typer(features_app, name="features")


def _iso_to_ms(s: str) -> int:
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


@app.command()
def fetch(
    config: Path = typer.Option(Path("configs/default.yaml")),
    data_root: Path = typer.Option(Path("data")),
):
    """拉配置里所有 symbols × intervals 的 K 线 + funding + OI。"""
    configure_logger()
    log = get_logger("cli.fetch")
    cfg = load_config(config)
    from rabbit_hunter.data_engine.okx_fetcher import (
        fetch_ohlcv, fetch_funding_rate_history, fetch_open_interest_history,
    )
    from rabbit_hunter.data_engine.quality import check_ohlcv
    from rabbit_hunter.data_engine.storage import write_ohlcv

    end_ms = _iso_to_ms(cfg.backtest.end)
    start_ms = _iso_to_ms(cfg.backtest.start)

    for symbol in cfg.data.symbols:
        for interval in (cfg.data.main_interval, cfg.data.confirm_interval):
            log.info("fetch_start", symbol=symbol, interval=interval)
            df = fetch_ohlcv(symbol, interval, start_ms, end_ms)
            qr = check_ohlcv(df, interval)
            paths = write_ohlcv(qr.clean_df, data_root, symbol, interval)
            log.info("fetch_done", symbol=symbol, interval=interval,
                     rows=len(qr.clean_df), issues=len(qr.issues), files=[str(p) for p in paths])

        # funding + OI 每个 symbol 只拉一次（1H 时序）
        fr = fetch_funding_rate_history(symbol, start_ms, end_ms)
        oi = fetch_open_interest_history(symbol, start_ms, end_ms)
        (data_root / "raw" / "okx" / symbol).mkdir(parents=True, exist_ok=True)
        fr.to_parquet(data_root / "raw" / "okx" / symbol / "funding.parquet", index=False)
        oi.to_parquet(data_root / "raw" / "okx" / symbol / "oi.parquet", index=False)
    typer.echo("fetch done")


@data_app.command("quality")
def data_quality(
    config: Path = typer.Option(Path("configs/default.yaml")),
    data_root: Path = typer.Option(Path("data")),
    report_out: Path = typer.Option(Path("data/quality_report.md")),
):
    """扫描已下载数据的质量并输出 md 报告。"""
    configure_logger()
    cfg = load_config(config)
    from rabbit_hunter.data_engine.storage import read_ohlcv
    from rabbit_hunter.data_engine.quality import check_ohlcv

    end_ms = _iso_to_ms(cfg.backtest.end)
    start_ms = _iso_to_ms(cfg.backtest.start)
    lines = ["# Data Quality Report", ""]
    for symbol in cfg.data.symbols:
        for interval in (cfg.data.main_interval, cfg.data.confirm_interval):
            df = read_ohlcv(data_root, symbol, interval, start_ms, end_ms)
            qr = check_ohlcv(df, interval)
            lines.append(f"## {symbol} @ {interval}")
            lines.append(f"- rows: {len(qr.clean_df)}")
            lines.append(f"- issues: {len(qr.issues)}")
            for i in qr.issues[:20]:
                lines.append(f"  - {i}")
            lines.append("")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines), encoding="utf-8")
    typer.echo(f"quality report written to {report_out}")


@features_app.command("build")
def features_build(
    config: Path = typer.Option(Path("configs/default.yaml")),
    data_root: Path = typer.Option(Path("data")),
):
    """预计算并缓存所有 symbol × interval 的特征。"""
    configure_logger()
    cfg = load_config(config)
    from rabbit_hunter.data_engine.storage import read_ohlcv
    from rabbit_hunter.feature_engine.pipeline import load_or_compute_features

    end_ms = _iso_to_ms(cfg.backtest.end)
    start_ms = _iso_to_ms(cfg.backtest.start)
    for symbol in cfg.data.symbols:
        def _raw(sym=symbol):
            df = read_ohlcv(data_root, sym, cfg.data.main_interval, start_ms, end_ms)
            fr_path = data_root / "raw" / "okx" / sym / "funding.parquet"
            oi_path = data_root / "raw" / "okx" / sym / "oi.parquet"
            if fr_path.exists():
                fr = pd.read_parquet(fr_path)
                df = pd.merge_asof(df.sort_values("timestamp"), fr.sort_values("timestamp"),
                                   on="timestamp", direction="backward")
            if oi_path.exists():
                oi = pd.read_parquet(oi_path)
                df = pd.merge_asof(df.sort_values("timestamp"), oi.sort_values("timestamp"),
                                   on="timestamp", direction="backward")
            return df

        def _confirm(sym=symbol):
            return read_ohlcv(data_root, sym, cfg.data.confirm_interval, start_ms, end_ms)

        feats = load_or_compute_features(
            root=data_root, symbol=symbol, interval=cfg.data.main_interval,
            engine_version=cfg.feature_engine.version,
            fetch_raw=_raw, fetch_confirm=_confirm,
        )
        typer.echo(f"features for {symbol} @ {cfg.data.main_interval}: {len(feats)} rows")


@app.command()
def backtest(
    config: Path = typer.Option(Path("configs/default.yaml")),
    data_root: Path = typer.Option(Path("data")),
    report_root: Path = typer.Option(Path("reports")),
    snapshot_root: Path = typer.Option(Path("snapshots")),
    start: str = typer.Option(None),
    end: str = typer.Option(None),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """端到端跑回测。"""
    configure_logger()
    log = get_logger("cli.backtest")
    cfg = load_config(config)
    if start:
        cfg.backtest.start = start
    if end:
        cfg.backtest.end = end

    if dry_run:
        typer.echo(f"dry-run: config loaded, symbols={cfg.data.symbols}, "
                   f"strategies={list(cfg.strategy_router.enabled_strategies.keys())}")
        raise typer.Exit(code=0)

    end_ms = _iso_to_ms(cfg.backtest.end)
    start_ms = _iso_to_ms(cfg.backtest.start)

    from rabbit_hunter.data_engine.storage import read_ohlcv
    from rabbit_hunter.feature_engine.pipeline import load_or_compute_features
    from rabbit_hunter.scoring_engine.strategies.trend_following import TrendFollowing, TFParams
    from rabbit_hunter.backtest.engine import BacktestEngine
    from rabbit_hunter.backtest.report import ReportBuilder
    from rabbit_hunter.observability.snapshot import SnapshotWriter
    import yaml as _yaml

    features_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in cfg.data.symbols:
        def _raw(sym=symbol):
            df = read_ohlcv(data_root, sym, cfg.data.main_interval, start_ms, end_ms)
            fr_path = data_root / "raw" / "okx" / sym / "funding.parquet"
            oi_path = data_root / "raw" / "okx" / sym / "oi.parquet"
            if fr_path.exists():
                fr = pd.read_parquet(fr_path)
                df = pd.merge_asof(df.sort_values("timestamp"), fr.sort_values("timestamp"),
                                   on="timestamp", direction="backward")
            if oi_path.exists():
                oi = pd.read_parquet(oi_path)
                df = pd.merge_asof(df.sort_values("timestamp"), oi.sort_values("timestamp"),
                                   on="timestamp", direction="backward")
            return df

        def _confirm(sym=symbol):
            return read_ohlcv(data_root, sym, cfg.data.confirm_interval, start_ms, end_ms)

        feats = load_or_compute_features(
            root=data_root, symbol=symbol, interval=cfg.data.main_interval,
            engine_version=cfg.feature_engine.version,
            fetch_raw=_raw, fetch_confirm=_confirm,
        )
        features_by_symbol[symbol] = feats

    tf_cfg_path = Path("configs") / cfg.strategy_router.enabled_strategies["trend_following"].config_file
    tf_yaml = _yaml.safe_load(tf_cfg_path.read_text(encoding="utf-8"))
    tf = TrendFollowing(TFParams(**tf_yaml["params"]))

    engine = BacktestEngine(cfg, [tf])
    result = engine.run(features_by_symbol)

    # 写快照
    sw = SnapshotWriter(root=snapshot_root)
    sw.append(result.snapshots.to_dict(orient="records"))
    sw.flush()

    # 生成报告
    builder = ReportBuilder(cfg, features_by_symbol)
    out_dir = builder.build(result, output_root=report_root, git_commit=_git_commit())
    log.info("backtest_done", report=str(out_dir),
             trades=len(result.ledger.closed_trades),
             final_equity=result.ledger.equity)
    typer.echo(f"report: {out_dir}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_cli.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add rabbit_hunter/cli.py tests/test_cli.py
git commit -m "feat(cli): fetch/data/features/backtest commands"
```

---

## Task 19: End-to-end integration test + smoke run

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_e2e_backtest.py`
- Create: `tests/integration/fixtures/gen_synthetic_ohlcv.py`

**Interfaces:**
- 合成一份 3 个月的 1H + 15m 数据（BTC + ETH），跑完整 pipeline，验证：
  1. `reports/YYYY-MM-DD-HHMM/` 目录被创建
  2. 六个交付文件都存在且非空：`report.md`, `ai_context.md`, `trades.parquet`, `snapshots.parquet`, `config_snapshot.yaml`, `charts/equity_curve.png`
  3. `report.md` 至少含"# Rabbit Hunter" 头
  4. `ai_context.md` 至少含 "Baseline Comparisons" 段

- [ ] **Step 1: 写合成数据 helper `tests/integration/fixtures/gen_synthetic_ohlcv.py`**

```python
import numpy as np
import pandas as pd


def gen_synthetic(n_bars: int = 24 * 90, base: float = 50_000.0, seed: int = 42) -> pd.DataFrame:
    """3 个月 1H 数据，含随机波动 + 缓慢上涨趋势。"""
    rng = np.random.default_rng(seed)
    ts = [i * 3_600_000 for i in range(n_bars)]
    trend = np.linspace(0, base * 0.5, n_bars)
    noise = rng.normal(0, base * 0.005, n_bars).cumsum()
    close = base + trend + noise
    close = np.clip(close, 1.0, None)
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0, base * 0.002, n_bars)
    low = np.minimum(open_, close) - rng.uniform(0, base * 0.002, n_bars)
    low = np.clip(low, 1.0, None)
    return pd.DataFrame({
        "timestamp": ts,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.uniform(100, 1000, n_bars),
        "funding_rate": rng.uniform(-0.0001, 0.0002, n_bars),
        "oi": np.linspace(1000, 1500, n_bars),
    })
```

- [ ] **Step 2: 写端到端集成测试 `tests/integration/test_e2e_backtest.py`**

```python
from pathlib import Path
import pandas as pd
from rabbit_hunter.config.loader import load_config
from rabbit_hunter.feature_engine.pipeline import build_features
from rabbit_hunter.scoring_engine.strategies.trend_following import TrendFollowing, TFParams
from rabbit_hunter.backtest.engine import BacktestEngine
from rabbit_hunter.backtest.report import ReportBuilder
from rabbit_hunter.observability.snapshot import SnapshotWriter
from .fixtures.gen_synthetic_ohlcv import gen_synthetic
import yaml


def test_full_pipeline_smoke(tmp_path):
    cfg = load_config(Path("configs/default.yaml"))
    cfg.backtest.start = "2025-01-01"
    cfg.backtest.end = "2025-04-01"

    features_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in cfg.data.symbols:
        raw = gen_synthetic(seed=42 if symbol == "BTC-USDT-SWAP" else 123)
        feats = build_features(raw, confirm=None, engine_version=cfg.feature_engine.version)
        features_by_symbol[symbol] = feats

    tf_cfg = yaml.safe_load(Path("configs/strategies/trend_following.yaml").read_text(encoding="utf-8"))
    tf = TrendFollowing(TFParams(**tf_cfg["params"]))

    engine = BacktestEngine(cfg, [tf])
    result = engine.run(features_by_symbol, open_action_threshold=0.3)

    sw = SnapshotWriter(root=tmp_path / "snapshots")
    sw.append(result.snapshots.to_dict(orient="records"))
    sw.flush()

    builder = ReportBuilder(cfg, features_by_symbol)
    out_dir = builder.build(result, output_root=tmp_path / "reports", git_commit="test")

    for f in ("report.md", "ai_context.md", "trades.parquet", "snapshots.parquet",
              "config_snapshot.yaml", "charts/equity_curve.png"):
        p = out_dir / f
        assert p.exists() and p.stat().st_size > 0, f"missing or empty: {f}"

    report_md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "# Rabbit Hunter" in report_md
    ai_md = (out_dir / "ai_context.md").read_text(encoding="utf-8")
    assert "Baseline Comparisons" in ai_md
```

- [ ] **Step 3: 跑测试确认通过**

Run: `pytest tests/integration -v`
Expected: 1 PASS，耗时 ~30-60s

- [ ] **Step 4: Docker 内跑一次全测试**

```bash
make build
make test
```
Expected: 全部测试 PASS

- [ ] **Step 5: 全流程 smoke run（真实 OKX 数据，缩短窗口）**

先手工缩短 config：
```bash
python -c "
import yaml
p = 'configs/default.yaml'
d = yaml.safe_load(open(p))
d['backtest']['start'] = '2026-06-01'
d['backtest']['end'] = '2026-07-01'
d['data']['history_window_days'] = 30
open(p, 'w').write(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
"
```

跑：
```bash
make fetch
make features
make backtest
```

验证：
```bash
ls -la reports/ | tail -5   # 应该看到新目录
ls -la reports/$(ls -t reports | head -1)   # 应该看到 report.md 等
head -30 reports/$(ls -t reports | head -1)/report.md
```

跑完把 config 还原到 2 年：
```bash
python -c "
import yaml
p = 'configs/default.yaml'
d = yaml.safe_load(open(p))
d['backtest']['start'] = '2024-07-05'
d['backtest']['end'] = '2026-07-05'
d['data']['history_window_days'] = 730
open(p, 'w').write(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
"
```

- [ ] **Step 6: 完整 2 年回测（Phase 1a 出口标准）**

```bash
make fetch      # 首次，约 5-10 分钟
make features   # 首次，约 30s
make backtest   # 完整回测，约 30-60s
```

验证 `reports/{YYYY-MM-DD-HHMM}/` 目录含全套六件套。

- [ ] **Step 7: Commit + push**

```bash
git add tests/integration
git commit -m "test(integration): full pipeline smoke on synthetic data"
git push -u origin phase-1a
```

---

## Task 20: Merge to `main` + tag `v0.1a`

**Files:**
- Modify: `README.md`（追加 Phase 1a 完成清单）

- [ ] **Step 1: 追加 README Phase 1a 完成清单**

在 `README.md` 末尾追加：

```markdown
## Phase 1a 完成清单

- [x] 数据采集：OKX 2 年 BTC/ETH 1H + 15m + funding + OI
- [x] 特征引擎：EMA/ADX/RSI/BB/ATR + 价格行为 + Regime 自动打标
- [x] 打分引擎：BaseStrategy 接口 + 硬约束 + trend_following v0.1.0
- [x] Strategy Router：weighted_avg composer
- [x] Risk Engine：ATR sizing + 单日熔断
- [x] BacktestExecutor：手续费 + 滑点 + 资金费
- [x] 报告层：Markdown + Parquet + PNG，AI 可学习格式
- [x] CLI + Docker + Makefile
- [x] 三层测试：单元 + 集成 + e2e

下一步：Phase 1b（加均值回归策略 + 多策略合成验证）
```

- [ ] **Step 2: 从 `phase-1a` 分支创建 PR，本地合并到 `main`**

```bash
git checkout main
git merge --no-ff phase-1a -m "merge: Phase 1a end-to-end backtest engine + trend_following"
git tag -a v0.1a -m "Phase 1a: end-to-end backtest with trend_following strategy"
```

- [ ] **Step 3: Push main 与 tag**

```bash
git push origin main
git push origin v0.1a
```

- [ ] **Step 4: 归档：把最新一次真实回测报告链接进 README**

在 README 末尾追加：

```markdown
## 最近一次回测报告

见 `reports/` 目录最新时间戳子目录。示例：`reports/2026-07-05-2100/report.md`
```

Commit：

```bash
git add README.md
git commit -m "docs: link latest backtest report"
git push origin main
```

Phase 1a 完成。1b/1c 开始前请重新走一次 brainstorming → writing-plans 流程（新策略、新合成模式可能需要重新对齐）。

---

## Self-Review Notes

对照 spec `docs/superpowers/specs/2026-07-05-rabbit-hunter-phase-1-design.md` 的每一节：

| Spec 章节 | 对应 Task | 覆盖度 |
|---|---|---|
| § 2 整体架构 + Docker | Task 1 | ✅ |
| § 3 Data Engine | Task 3/4/5 | ✅ |
| § 4 Feature Engine | Task 6/7/8/9 | ✅ |
| § 5 Scoring Engine | Task 10/11 | ✅（仅 trend_following；1b/1c 补 mean_reversion/price_action） |
| § 6 Strategy Router | Task 12 | ✅（仅 weighted_avg；其他 composer 显式 NotImplementedError） |
| § 7 Risk Engine（Phase 1 精简版） | Task 13 | ✅ |
| § 8 Execution Engine | Task 14 | ✅ |
| § 9 Backtest 主循环 | Task 15 | ✅ |
| § 10 Report Layer | Task 17 | ✅（含 trades.parquet / ai_context.md / 失败集群 / regime 表 / feature 相关性） |
| § 11 Config 管理 | Task 2 | ✅ |
| § 12 Testing 策略 | 所有 Task 均 TDD；Task 9 有 baseline 快照锁死 | ✅ |
| § 13 Observability | Task 16 | ✅ |
| § 14 CLI | Task 18 | ✅ |
| § 15 错误处理 | 分散在各 Task（BacktestExecutor/Ledger/RiskEngine 均已包含防御性判空/拒单） | 部分（生产层 try/except 保护主循环建议在 e2e 验证时补） |
| § 16 开发工作流 | Task 20 | ✅ |
| § 17 显式排除项 | Task 12/14 显式 NotImplementedError 覆盖 | ✅ |

**兼容性检查**：
- `Order` 类型在 Task 13 定义，Task 14/15 直接消费 ✅
- `Intent` 类型在 Task 12 定义，Task 15 消费 ✅
- `Fill` 类型在 Task 14 定义，Task 15 消费 ✅
- `Position/Ledger` 在 Task 15 定义，Task 17 报告构建消费 ✅
- `AppConfig` 在 Task 2 定义，被所有下游 Task 消费 ✅
- Feature 列名（`ema20`, `adx`, `rsi_14`, `atr_14`, `regime`, `funding_rate` 等）在 Task 6/7/8/9 生产，在 Task 11 策略打分中消费，在 Task 17 报告 flatten 时消费 ✅

**已知未覆盖**（后续 phase）：
- `PaperExecutor` 只留桩，`LiveExecutor` 不做（1b/1c/Phase 3）
- `composer ∈ {unanimous, regime_switch, max_score}` 只在 Router 里显式抛 NotImplementedError（1b/1c）
- 组合层风控 / 相关性矩阵 / 独立熔断进程（Phase 3）
- 可训练打分模型（Phase 2）
- AI Review Agent 接 LLM（Phase 4）—— Phase 1a 只准备数据

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-05-rabbit-hunter-phase-1a.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
