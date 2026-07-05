# Rabbit Hunter V5.1 · Phase 1 设计稿

- 日期：2026-07-05
- 版本：Phase 1 落地设计 v1
- 依据：`Rabbit_Hunter_V5.1_架构设计稿.docx`（原始架构文档）
- 远程仓库：`https://github.com/xiaomimi123/rabbit-Hunter-5.0-.git`（注：远程仓库名为 5.0，本设计基于 5.1 架构稿；后续可以决定是否新建 5.1 仓库或沿用此仓库）

## 目录

1. [决策快照](#1-决策快照)
2. [整体架构](#2-整体架构)
3. [Data Engine](#3-data-engine)
4. [Feature Engine](#4-feature-engine)
5. [Scoring Engine](#5-scoring-engine)
6. [Strategy Router](#6-strategy-router)
7. [Risk Engine（Phase 1 精简版）](#7-risk-enginephase-1-精简版)
8. [Execution Engine](#8-execution-engine)
9. [Backtest 主循环](#9-backtest-主循环)
10. [Report Layer（AI 可学习版）](#10-report-layerai-可学习版)
11. [Config 管理](#11-config-管理)
12. [Testing 策略](#12-testing-策略)
13. [Observability](#13-observability)
14. [CLI](#14-cli)
15. [错误处理](#15-错误处理)
16. [开发工作流：Phase 1a → 1b → 1c](#16-开发工作流phase-1a--1b--1c)
17. [显式排除项（Phase 1 不做）](#17-显式排除项phase-1-不做)

---

## 1. 决策快照

| 决策项 | 结果 | 备注 |
|---|---|---|
| 覆盖范围 | Phase 1 端到端跑通 | 架构稿 § 6 建议的第一阶段 |
| Phase 1 内部拆分 | 1a 趋势跟随 → 1b 加均值回归 → 1c 加价格行为学 | 用户主动要求三个策略风格，用分阶段降低复杂度 |
| 技术栈 | Python 3.11+ | 量化生态成熟，回测/AI 集成方便 |
| 交易所 | OKX 永续合约 | 用户指定 |
| MVP 出口 | 回测跑通、能出报告（无需 API Key） | 架构稿 § 6 Phase 1 定位 |
| 交易标的 | BTC-USDT-SWAP + ETH-USDT-SWAP | 数据干净、验证多标的架构 |
| 主 K 线周期 | 1H | 用户指定 |
| 确认周期 | 15m | 跨周期确认 |
| 回测窗口 | 2 年 | 覆盖 2024 牛/2025 震荡两种风格 |
| 架构模式 | 单进程模块化 Python 包 + Parquet/DuckDB | 无消息队列，回测/实盘同源 |
| 运行环境 | Docker 单容器 | 用户指定 |
| 报告输出 | Markdown + PNG + Parquet + ai_context.md | AI 可学习版本 |

---

## 2. 整体架构

### 2.1 两种运行模式（共用同一份 Feature/Scoring/Strategy/Risk 代码）

```
┌───────────────────────────────────────────────────────────────┐
│  模式一：BACKTEST（Phase 1 MVP 唯一交付）                       │
│  历史 K 线（Parquet）→ Feature → Scoring → Strategy →           │
│  Risk → BacktestExecutor（模拟成交）→ 交易账本 → 报告            │
├───────────────────────────────────────────────────────────────┤
│  模式二：PAPER（Phase 1 只留桩，代码接口相同）                  │
│  OKX WebSocket K 线 → 同上四层 → PaperExecutor（不真下单）      │
└───────────────────────────────────────────────────────────────┘
```

关键约束：Feature/Scoring/Strategy/Risk 四层完全不感知回测/纸面/实盘的区别。区别只在最上游（数据源）和最下游（Executor 类型）。这是架构稿 § 4.2 "回测和实盘必须调用完全相同的 Feature Engine 代码"的技术落点。

### 2.2 目录结构

```
rabbit-hunter/
├── rabbit_hunter/                    # 主包
│   ├── data_engine/
│   │   ├── okx_fetcher.py            # OKX REST 拉 K 线 + funding + OI
│   │   ├── quality.py                # 跳空/缺失/时间戳异常检测
│   │   └── storage.py                # Parquet 读写 + DuckDB 视图
│   ├── feature_engine/
│   │   ├── indicators.py             # EMA/ADX/RSI/BB/ATR
│   │   ├── price_action.py           # K 线形态/S/R/市场结构
│   │   ├── regime.py                 # 自动打行情标签
│   │   └── pipeline.py               # K 线 → 特征 DataFrame
│   ├── scoring_engine/
│   │   ├── base.py                   # BaseStrategy 抽象类
│   │   ├── rules_hard.py             # 硬约束（流动性/熔断）
│   │   └── strategies/               # ★ 策略插件目录
│   │       ├── trend_following.py    # 1a 上线
│   │       ├── mean_reversion.py     # 1b 上线
│   │       └── price_action.py       # 1c 上线
│   ├── strategy_router/
│   │   └── router.py                 # 多策略合成
│   ├── risk_engine/
│   │   ├── position_sizing.py        # ATR 止损 + 固定风险仓位
│   │   └── daily_circuit.py          # 单日熔断
│   ├── execution_engine/
│   │   ├── base.py                   # BaseExecutor 接口
│   │   ├── backtest_executor.py      # 滑点/手续费/资金费
│   │   └── paper_executor.py         # 桩，Phase 1 不实现
│   ├── backtest/
│   │   ├── engine.py                 # 主循环
│   │   ├── ledger.py                 # 交易账本
│   │   └── report.py                 # AI 可学习报告
│   ├── observability/
│   │   ├── logger.py                 # structlog
│   │   └── snapshot.py               # 决策快照落 Parquet
│   ├── config/
│   │   ├── schema.py                 # Pydantic 校验
│   │   └── loader.py                 # YAML → dataclass
│   └── cli.py                        # typer 入口
├── configs/
│   ├── default.yaml
│   └── strategies/
│       ├── trend_following.yaml
│       ├── mean_reversion.yaml
│       └── price_action.yaml
├── data/                             # .gitignore
│   ├── raw/
│   └── features/
├── snapshots/                        # .gitignore
├── reports/                          # .gitignore
├── tests/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── pyproject.toml
├── README.md
├── Makefile
└── docs/superpowers/specs/           # 本设计稿在此
```

### 2.3 依赖

| 类别 | 库 | 用途 |
|---|---|---|
| 数据 | `pandas`, `pyarrow`, `duckdb` | K 线 / 特征 / 快照存取 |
| 指标 | `pandas-ta` | EMA/RSI/ADX/BB/ATR（纯 Python，Docker 友好） |
| 交易所 | `ccxt` | OKX 公共历史行情（无 Key） |
| 配置 | `pydantic`, `pyyaml` | 类型校验 |
| 日志 | `structlog` | 结构化日志 |
| CLI | `typer` | 命令行 |
| 报告 | `jinja2`, `matplotlib`, `scipy` | MD 模板 + PNG 图 + 相关性算 |
| 测试 | `pytest`, `pytest-cov` | 单元 + 集成 + 回归 |

### 2.4 Docker

- **单容器**：`rabbit` 服务，Python 3.11 slim 基底，装完依赖即可运行
- DuckDB 无进程依赖，直连 Parquet 文件；不需要独立数据库容器
- 挂载：`data/` `snapshots/` `reports/` `configs/` 到宿主
- 命令：`make fetch` / `make backtest` / `make test`

---

## 3. Data Engine

### 3.1 组件

```
OKX 公共 REST ─→ okx_fetcher ─→ quality ─→ storage (Parquet 分区)
                                     │
                                     └→ data/quality_report.md
```

| 文件 | 职责 |
|---|---|
| `okx_fetcher.py` | ccxt 拉 K 线 + funding rate 历史 + OI 历史；断点续传；无 API Key |
| `quality.py` | 跳空 / 时间戳错乱 / 价格为 0 / NaN 检测；异常 K 线丢弃并写异常日志 |
| `storage.py` | Parquet 分区读写 + DuckDB 视图 |

### 3.2 存储 Schema

- 路径：`data/raw/okx/{symbol}/{interval}/year={YYYY}/month={MM}.parquet`
- 列：`timestamp` (UTC ISO8601), `open`, `high`, `low`, `close`, `volume`, `quote_volume`, `funding_rate`, `oi`
- 分区：按年/月，方便按窗口读

---

## 4. Feature Engine

### 4.1 模块

| 文件 | 内容 |
|---|---|
| `indicators.py` | pandas-ta 封装：EMA(20/60/200)、ADX、RSI、BB、ATR、ATR% |
| `price_action.py` | K 线形态、S/R、市场结构 |
| `regime.py` | 自动打行情标签 `regime ∈ {trending, ranging, high_vol, low_vol}` |
| `pipeline.py` | 调度器：K 线 → 特征 DataFrame（含跨周期 1H↔15m 对齐） |

### 4.2 完整特征列表（Phase 1a 就全部计算，无论当前只用哪个策略）

```
基础: timestamp, symbol, interval, open, high, low, close, volume

趋势跟随: ema20, ema60, ema200, ema20_slope, adx, di_plus, di_minus

均值回归: rsi_14, bb_upper, bb_middle, bb_lower, bb_width, bb_pct, zscore_20

价格行为学:
  pattern_engulfing_bull, pattern_engulfing_bear, pattern_pinbar,
  pattern_inside_bar, pattern_doji,
  swing_high_last, swing_low_last,
  structure_regime, bos_flag, choch_flag

上下文: atr_14, atr_pct, volume_ratio_20, funding_rate, oi_change_pct

跨周期: ema20_1h_on_15m, adx_1h_on_15m

自动打标: regime, session, day_of_week
```

理由：架构稿 § 3.5 要求 AI Review 拿到"每笔交易的完整特征快照"。Phase 1a 就把 PA 特征算好意味着 Phase 1c 上策略时不用回改 Feature Engine，AI Review 也一直有完整上下文。

### 4.3 特征缓存

- 首跑写 `data/features/{symbol}/{interval}/*.parquet`
- 复跑校验 Feature Engine 语义版本 + 关键参数 hash 是否变化，一致则直接读缓存
- BTC+ETH 2 年缓存约 30MB

### 4.4 防未来函数

- 无状态、逐 tick：`compute(bars_up_to_t) -> features_at_t`
- 单元测试锁死历史特征值：任何改动如果影响 baseline 快照 CSV 立即红灯（对应架构稿 § 4.1 "模型版本管理"）

---

## 5. Scoring Engine

### 5.1 两层结构

```
输入: features DataFrame
   │
   ▼
硬约束层 rules_hard.py（非黑即白）
   • 流动性过滤（24h 成交额 < 阈值）
   • 极端波动熔断（ATR% > N 倍近期均值）
   • 数据质量拒绝
   │ 通过
   ▼
策略层 strategies/*.py（评分 0~1）
   • 每策略独立文件，实现 BaseStrategy
```

### 5.2 BaseStrategy 接口（锁定，未来加策略不改）

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import pandas as pd

@dataclass(frozen=True)
class ScoreOutput:
    long: float                    # 0.0 ~ 1.0
    short: float                   # 0.0 ~ 1.0
    components: dict[str, float]   # 可解释性："EMA 交叉贡献 0.4"
    metadata: dict[str, Any]       # 命中形态名、关键位价格，进 trades.parquet

class BaseStrategy(ABC):
    name: str                      # 唯一标识，config 用它开关
    version: str                   # 语义版本

    @abstractmethod
    def score(
        self,
        features_row: dict,
        features_history: pd.DataFrame,
    ) -> ScoreOutput:
        """
        features_row: 当前 tick 全部特征
        features_history: 最近 N 根 K 线（策略自定义 lookback）
        """
```

### 5.3 加新策略步骤

1. 在 `strategies/` 新建 `my_new_strategy.py`，继承 `BaseStrategy`
2. 实现 `score()`（一般 20-80 行）
3. 在 `configs/strategies/` 新建 `my_new_strategy.yaml`
4. `configs/default.yaml` 加一行 `enabled: true`

上层完全不用改。

---

## 6. Strategy Router

### 6.1 职责

1. 启动时扫描 `strategies/` + 读 config，加载所有 `enabled: true` 的策略
2. 每 tick 把 features 喂给所有启用策略，并行拿多份 ScoreOutput
3. 按合成模式产出下单意图 `Intent`

### 6.2 合成模式

| 模式 | 逻辑 | 适合 |
|---|---|---|
| `weighted_avg`（默认） | `final_long = Σ w_i × strategy_i.long` | 通用，1a 单策略时天然退化 |
| `unanimous` | 全部策略 > 阈值才开仓 | 保守，减少交易 |
| `regime_switch` | 按 regime 特征切策略 | 1b/1c 多策略后可用 |
| `max_score` | 取分数最高的策略 | 简单 |

### 6.3 Intent 输出

```python
@dataclass(frozen=True)
class Intent:
    symbol: str
    action: Literal['open_long', 'open_short', 'close', 'wait']
    conviction: float                     # 0~1，风控用来决定仓位
    contributing_strategies: dict         # {'trend_following': 0.6, ...}
    features_snapshot: dict               # 完整特征快照，写入 trades.parquet
```

---

## 7. Risk Engine（Phase 1 精简版）

### 7.1 单笔风控

```
Intent
   │
   ▼
position_sizing.py
   • ATR 止损距离 = k × ATR_14
   • 止盈距离 = 止损 × R:R
   • 仓位 = (equity × risk_per_trade_pct) / stop_distance
   • 最大杠杆 cap
   │
   ▼
daily_circuit.py
   • 单日亏损 > X% → 当日禁开新单
   │
   ▼
Order（含止损/止盈价、仓位大小）
```

### 7.2 Phase 1 不做

- ❌ 组合相关性矩阵（推 Phase 3）
- ❌ Beta 集中度（推 Phase 3）
- ❌ 独立熔断进程（推 Phase 3）

对应架构稿 § 6 Phase 3 定义。

---

## 8. Execution Engine

### 8.1 接口

```python
class BaseExecutor(ABC):
    @abstractmethod
    def submit(self, order: Order) -> Fill: ...
    @abstractmethod
    def close(self, position: Position) -> Fill: ...
```

### 8.2 Phase 1 实现

| 实现 | Phase 1 状态 |
|---|---|
| `BacktestExecutor` | ✅ 完整；下一根 K 线开盘价成交；含滑点/手续费/资金费 |
| `PaperExecutor` | 🔶 空壳桩，仅确认接口 |
| `LiveExecutor` | ❌ Phase 3 才做 |

### 8.3 成本模型（架构稿 § 4.2 强制）

```yaml
execution:
  fees:
    maker: 0.02%
    taker: 0.05%
  slippage_model: "atr_based"
  slippage_atr_multiplier: 0.1     # slippage = 0.1 × ATR_14
  funding_settlement: true          # 每 8 小时结算
```

---

## 9. Backtest 主循环

```python
def run(config):
    features_all = feature_pipeline.load_or_compute(...)
    ledger = Ledger()
    executor = BacktestExecutor(config.execution)
    router = StrategyRouter(config)
    risk = RiskEngine(config.risk)

    for timestamp in timestamps:
        for symbol in symbols:
            features_row = features_all.loc[(symbol, timestamp)]
            history = features_all.loc[symbol].loc[:timestamp].tail(200)

            if not rules_hard.pass_(features_row):
                continue

            intent = router.route(features_row, history)
            order = risk.size(intent, ledger.equity, ledger.open_positions)
            if order is None:
                continue

            fill = executor.submit(order, next_bar)
            ledger.record(fill)
            snapshot.append(timestamp, symbol, features_row, intent, order, fill)

        ledger.check_exits(features_all, timestamp)

    return ledger, snapshot
```

---

## 10. Report Layer（AI 可学习版）

### 10.1 输出目录结构

```
reports/2026-07-05-1930/
├── report.md              # 人看的摘要
├── charts/*.png           # 净值/回撤/月度/散点
├── trades.parquet         # ★ AI 主食：每笔交易 30+ 列
├── snapshots.parquet      # ★ AI 主食：每次决策全量快照
├── ai_context.md          # ★ AI 主食：baseline/失败集群/regime 表
└── config_snapshot.yaml   # 本次回测完整配置
```

### 10.2 report.md 结构（人看）

```markdown
# Rabbit Hunter 回测报告 · 2026-07-05 19:30

## 元数据
- 回测区间 / 标的 / 主周期 / 启用策略 / 配置 hash / git commit

## 收益概况
| 总收益率 / 年化 / 最大回撤 / 夏普 / 交易笔数 / 胜率 / 盈亏比 |

![净值曲线](charts/equity_curve.png)

## 分标的表现
（每 symbol 一行）

## 分策略贡献
（Phase 1a 只有一个策略；1b+ 展开）

## 月度盈亏
![月度盈亏](charts/monthly_pnl.png)

## 亏损最大的 10 笔
（AI Review 快速定位关键 loser）

## 决策快照汇总
（快照总数、平均延迟、索引路径）

## 配置全文
（内嵌 default.yaml，AI 一站式理解上下文）
```

### 10.3 trades.parquet Schema

```
基本: symbol, side, entry_time, exit_time, entry_price, exit_price, size,
      pnl_raw, pnl_after_fees, fees, funding, slippage, hold_bars

入场快照 (t0): ema20_t0, ema60_t0, ema200_t0, adx_t0, rsi_t0, atr_t0,
             atr_pct_t0, bb_width_t0, volume_ratio_t0, funding_rate_t0,
             oi_change_t0, ...

出场快照 (t_exit): 同上

信号: entry_signal_name, score_long_t0, score_short_t0,
     score_components (JSON: {'ema_cross': 0.4, ...})

出场原因: exit_reason ∈ {take_profit, stop_loss, signal_flip, timeout}

行情标签: regime, session, day_of_week
```

### 10.4 ai_context.md 结构（AI 专用事实清单）

```markdown
# AI Review Context - 2026-07-05 backtest

## Data Provenance
- 数据源 / 时间窗口 / 质量报告链接

## Baseline Comparisons
| Baseline | 收益率 | 夏普 | 最大回撤 |
| Buy-and-Hold BTC | ... |
| Buy-and-Hold ETH | ... |
| 本策略 | ... |

## Failure Mode Clusters
（脚本自动算出胜率 < 40% 且交易数 > 20 的桶）

## Regime-Conditional Performance
| Regime | 交易数 | 收益率 | 胜率 | 夏普 |

## Feature Correlation with PnL
（Spearman 相关性 top 10 正/负相关特征）
```

### 10.5 失败集群算法

```python
def find_loss_clusters(trades, min_trades=20, max_winrate=0.4):
    dims = ['regime', 'session', 'day_of_week']
    clusters = []
    for dim_combo in itertools.chain(dims, itertools.combinations(dims, 2)):
        grouped = trades.groupby(list(dim_combo)).agg(
            trades=('pnl', 'count'),
            winrate=('pnl', lambda x: (x > 0).mean()),
            total_pnl=('pnl', 'sum'),
        )
        clusters.extend(grouped[
            (grouped['trades'] >= min_trades) &
            (grouped['winrate'] <= max_winrate)
        ].to_records())
    return clusters
```

放在 `report.py` 里，纯 Python 预算，可测试可复现。AI Review Agent 拿到 `ai_context.md` 就已经看到集群，不用现算。

---

## 11. Config 管理

### 11.1 分层

- `configs/default.yaml` —— 全局
- `configs/strategies/*.yaml` —— 每策略独立

### 11.2 default.yaml 示例

```yaml
data:
  exchange: okx
  symbols: [BTC-USDT-SWAP, ETH-USDT-SWAP]
  main_interval: 1H
  confirm_interval: 15m
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
  fees: {maker: 0.02%, taker: 0.05%}
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

### 11.3 校验

Pydantic 加载时全字段类型检查，错立即报错。

---

## 12. Testing 策略

### 12.1 三层

| 层 | 时长 | 覆盖 |
|---|---|---|
| 单元测试 | `<1s` | 每个纯函数（指标计算、风控公式、报告聚合） |
| 集成测试 | `~10s` | 引擎组合（Feature+Scoring 恒等输入恒等输出） |
| 回归测试 | `~1min` | 小样本回测（锁一段真实历史，胜率/收益必须在 ε 内） |

### 12.2 关键锁死测试

- **回测↔实盘代码路径一致性**：同一段 K 线经 Feature Engine，回测调用 vs. PaperExecutor 调用产出必须**逐字节相等**（不是数值近似）
- **特征基线快照**：`tests/baselines/features_v0_1_0.csv`，任何影响历史特征值的改动必红灯（对应架构稿 § 4.1）

---

## 13. Observability

### 13.1 结构化日志（structlog）

```json
{"ts": "2026-07-05T19:30:00Z", "level": "info", "event": "score_computed",
 "symbol": "BTC-USDT-SWAP", "strategy": "trend_following",
 "long": 0.72, "short": 0.05, "elapsed_ms": 3.4}
```

### 13.2 决策快照

- 每次打分落 Parquet
- 内容：timestamp, symbol, features_all_columns, strategy_outputs, intent, order, fill
- 存储：`snapshots/YYYY-MM-DD/*.parquet`，按天分区
- 用途：回测报告可查、AI Review 训练、生产故障复现（架构稿 § 4.3）

---

## 14. CLI

```bash
rabbit fetch --symbols BTC-USDT-SWAP,ETH-USDT-SWAP --interval 1H --days 730
rabbit data quality --report
rabbit features build
rabbit backtest --config configs/default.yaml
rabbit backtest --start 2025-01-01 --end 2025-06-30
rabbit backtest --dry-run
```

Docker 里对应：`make fetch` / `make backtest` / `make test`。

---

## 15. 错误处理

| 层 | 出错时行为 |
|---|---|
| 数据层 | 跳过 tick + 结构化告警，不 crash |
| 策略层 | 单策略 exception → 该 tick 返回 wait，落日志，其他策略继续 |
| 风控层 | 任何异常 = 拒单（Fail Safe） |
| 回测框架自身 | 立即 crash + 完整堆栈 |

---

## 16. 开发工作流：Phase 1a → 1b → 1c

### 16.1 三个里程碑

| 里程碑 | 交付 | 完成标准 |
|---|---|---|
| **1a** | 引擎骨架 + trend_following 策略 | 能 `rabbit backtest` 跑完 2 年 BTC/ETH，出 `reports/YYYY-MM-DD/` 完整目录 |
| **1b** | 加 mean_reversion 策略 + 多策略合成 | 同上，且 `strategy_router.composer=weighted_avg` 两个策略并跑 |
| **1c** | 加 price_action 策略 | 三个策略并跑，`ai_context.md` 里三个策略的贡献都可查 |

### 16.2 Git 分支

- `main`：只有稳定版
- `phase-1a` / `phase-1b` / `phase-1c`：三个里程碑分支
- 每完成一小步 merge 回 main + tag `v0.1a` / `v0.1b` / `v0.1c`

### 16.3 Merge 前必须过

- `make test`（全部三层测试）
- `make backtest`（回归回测；与前一版差异在 ε 内，或有明确解释）

---

## 17. 显式排除项（Phase 1 不做）

对应架构稿 § 6 后续 Phase：

- ❌ AI Review Agent（Phase 4）——本设计只准备"AI 能吃的数据"，不接 LLM
- ❌ 可训练打分模型（Phase 2）——Phase 1 只有规则打分
- ❌ 组合风险层（Phase 3）
- ❌ 心跳看门狗（Phase 3）
- ❌ 多交易所抽象（Phase 5）——OKX 硬编码
- ❌ WebSocket 实时数据（Phase 1 只做回测；PaperExecutor 只是接口桩）
- ❌ 影子模式 / 样本外回测框架（Phase 4）

---

## 附：与架构稿的映射

| 架构稿章节 | 本设计对应 | Phase 1 覆盖度 |
|---|---|---|
| § 3.1 Data Engine | § 3 Data Engine | 部分覆盖：多源校验推 Phase 2；异常检测、Parquet 落地已做 |
| § 3.2 Scoring Engine | § 4 Feature + § 5 Scoring | 规则版打分覆盖；可训练模型 Phase 2 |
| § 3.3 Strategy & Portfolio Risk | § 6 Router + § 7 Risk | 单笔风控覆盖；组合层 Phase 3 |
| § 3.4 Execution Engine | § 8 Execution | 回测执行器覆盖；看门狗/对账 Phase 3 |
| § 3.5 AI Review Agent | § 10 Report Layer | Phase 1 准备好 AI 能吃的数据，不接 LLM |
| § 4.1 模型版本管理 | § 12 特征基线快照 | 部分覆盖：strategy version 字段 + baseline CSV 锁死 |
| § 4.2 研究/生产环境隔离 | § 2.1 两种模式共用代码 | 完全覆盖 |
| § 4.3 可观测性与审计 | § 13 Observability | 完全覆盖：结构化日志 + 决策快照 |
