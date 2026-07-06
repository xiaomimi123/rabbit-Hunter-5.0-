# 日内短打 · 高杠杆 · 移动止损 profile 设计

- 日期：2026-07-06
- 版本：v0.2.0-scalp（相对 Phase 1a 是策略画像变体，不推翻底层引擎）
- 用户诉求：真实本金 $1000，允许高杠杆，重点在移动止盈止损，日内高频（~3 单/天），持仓 ≤ 8 小时

---

## 1. 资金 & 杠杆模型

| 参数 | 原 (v0.1a-tuned) | 新 (v0.2.0-scalp) | 含义 |
|---|---|---|---|
| `initial_capital` | 10000 | **1000** | 直接建模真实本金 $1000 |
| `risk_per_trade_pct` | 1.0% | **3.0%** | 单笔最多亏 3% × $1000 = **$30** |
| `max_leverage` | 3× | **20×** | 名义仓位上限 = 20 × $1000 = $20,000 |
| `daily_max_loss_pct` | 3.0% | **10.0%** | 单日累亏 > $100 熔断 → 当日禁开新单 |

**每笔实际杠杆推导**：
- risk_amount = $30
- stop_distance = 1.0 × ATR14（下面会说为什么这么紧）
- size = $30 / stop_distance
- notional = size × price
- leverage = notional / $1000

具体数字例：BTC $60,000, ATR14 = $400 → stop_dist = $400 → size = 0.075 BTC → notional = $4,500 → leverage = **4.5×**

BTC 低波动时段（ATR = $200）→ leverage = **9×**，可能会触到 20× cap 的极端情况：ATR = $50 → leverage = 60×，被 cap 到 20×。

---

## 2. 时间框架 & 频率

| 参数 | 原 | 新 |
|---|---|---|
| `main_interval` | `1H` | **`15m`** |
| `confirm_interval` | `15m` | **`1H`**（反过来用，用高周期做趋势确认） |
| `history_window_days` | 730 | 730（不变） |
| `hold_timeout_bars` | 96（96 × 1H = 96 小时） | **32**（32 × 15m = **8 小时**）|

**频率预期**：
- 15m 主 K → 每天 96 根决策 bar（而不是 24 根）→ 4× 决策机会
- 但 trend_following 本身不是高频策略，即使放宽也很难到 3 单/天
- 预估：**每天 1-2 单**，2 年累计 **500-1000 笔**
- 若你确实要 3 单/天，得等 Phase 1b 加均值回归策略（本 spec 不做，标注为已知限制）

---

## 3. 止盈止损结构

### 3.1 静态止损（安全网，入场时挂上）
```
stop_distance = atr_stop_multiplier × ATR14   # atr_stop_multiplier = 1.0
多头: static_stop = entry_price - stop_distance
空头: static_stop = entry_price + stop_distance
```
比原来 2.5× 收紧了。理由：既然要短打，就不能给它太多"缓冲空间"。

### 3.2 静态止盈（保底 TP）
```
tp_distance = reward_risk_ratio × stop_distance   # reward_risk_ratio = 2.0
多头: static_tp = entry_price + tp_distance   # 2R 上方
空头: static_tp = entry_price - tp_distance   # 2R 下方
```
比原来 2.5R 收紧到 2R。理由：短打持仓，不能等太远的止盈。

### 3.3 移动止损（trailing SL，达到 1R 后激活）

**新增数据结构**（`Position` 加字段）：
```python
@dataclass
class Position:
    ...
    initial_stop: float         # 入场时的原始静态止损，永久保留（安全网）
    max_favorable_price: float  # 多头看 bar.high 最大值，空头看 bar.low 最小值
    trailing_active: bool       # 是否已进入 trailing 模式
```

**每根 K 的更新逻辑**（写在 `Ledger.check_exits` 里）：

```python
# Step 1: 更新最有利价（水位线）
if pos.side == "long":
    pos.max_favorable_price = max(pos.max_favorable_price, bar.high)
else:
    pos.max_favorable_price = min(pos.max_favorable_price, bar.low)

# Step 2: 检查是否达到激活门槛（1R 利润）
if not pos.trailing_active:
    initial_r = abs(pos.entry_price - pos.initial_stop)
    profit_r = abs(pos.max_favorable_price - pos.entry_price)
    if profit_r >= trailing_activation_r × initial_r:   # trailing_activation_r = 1.0
        pos.trailing_active = True

# Step 3: 激活后每根 K 计算新的 trailing stop
if pos.trailing_active:
    if pos.side == "long":
        new_trail = pos.max_favorable_price - trailing_atr_multiplier × ATR14
        pos.stop = max(pos.stop, new_trail)   # 只能上移，不能下调
    else:
        new_trail = pos.max_favorable_price + trailing_atr_multiplier × ATR14
        pos.stop = min(pos.stop, new_trail)   # 只能下移，不能上调
```

**参数**：
- `trailing_activation_r = 1.0` —— 达到 1R 利润（回本再赚一半"止损距离"）才激活
- `trailing_atr_multiplier = 1.0` —— 追踪 1 × ATR14 距离

**行为示例**（多头 $60,000 入场，止损 $59,600，止盈 $60,800，1R = $400）：
```
t=0    入场   entry=$60,000, stop=$59,600, tp=$60,800, HW=$60,000, trail=OFF
t=1    价 $60,200, HW=$60,200, profit=0.5R, trail=OFF
t=2    价 $60,400, HW=$60,400, profit=1.0R → 激活 trail
                                        → trail_stop = $60,400 - $400 = $60,000
                                        → pos.stop = max($59,600, $60,000) = $60,000（上移了）
t=3    价 $60,700, HW=$60,700, trail_stop = $60,700 - $400 = $60,300
                                → pos.stop = max($60,000, $60,300) = $60,300（继续上移）
t=4    价 $60,500（回调）  HW 不变 = $60,700, trail_stop = $60,300 不变（不能回退）
t=5    价 $60,290 → 击穿 $60,300 stop → 平仓，赚 = $60,300 - $60,000 = $300
```
这笔按静态 TP 走本来能拿到 $800，但价格没直接冲到 TP，中途回调触发 trailing 出场，锁到 $300。

**代价**：某些趋势特别强的时候，价格从 $60,000 一路推到 $61,500 静态 TP 是 $60,800，本会拿到 $800 —— 但如果那次没有回调直接冲，静态 TP 会先触发，不会被 trailing 影响。所以 trailing 不会伤害"直冲 TP"的赢家。

---

## 4. 策略参数

**trend_following v0.1.3 参数放宽**（不改代码，只改 yaml）：
| 参数 | 原 | 新 | 影响 |
|---|---|---|---|
| `adx_threshold` | 35 | **20** | 允许中等强度趋势入场 |
| `volume_ratio_threshold` | 1.5 | **1.2** | 放松成交量确认 |
| `confirm_adx_threshold` | 25 | **20** | 15m 主 K 时，1H 的 ADX 门槛降回 20 |
| `funding_weight` | 0.20 | **0.15** | 略降 funding 权重（15m 上 funding 变动更少） |

---

## 5. 需要的代码改动

### 5.1 `config/schema.py` — RiskConfig 加字段
```python
class RiskConfig(BaseModel):
    ...  # 现有字段
    trailing_enabled: bool = True
    trailing_activation_r: float = Field(gt=0, default=1.0)
    trailing_atr_multiplier: float = Field(gt=0, default=1.0)
```

### 5.2 `backtest/ledger.py` — Position + check_exits
- `Position` 加 `initial_stop`, `max_favorable_price`, `trailing_active`
- `record_entry` 初始化新字段
- `check_exits` 每 bar 三步（上面 § 3.3）

### 5.3 `backtest/engine.py` — 把 trailing_config 从 RiskConfig 传下去
`check_exits` 需要 trailing 参数。当前签名：
```python
ledger.check_exits(symbol, bar, atr, executor, hold_timeout_bars, exit_snapshot_fn)
```
改成：
```python
ledger.check_exits(symbol, bar, atr, executor, hold_timeout_bars, exit_snapshot_fn, trailing_config)
```
其中 `trailing_config` 是个小 dataclass 或 tuple `(enabled, activation_r, atr_mult)`。

### 5.4 单元测试（写 2 个）
- `test_trailing_activates_at_1r_profit`
- `test_trailing_never_moves_stop_backward`

### 5.5 `open_action_threshold` 是否要暴露成 config？
- 当前 `BacktestEngine.run(open_action_threshold=0.5)` 是硬编码
- v0.2.0-scalp 想放宽入场门槛到 0.4 → 需要暴露
- 加到 `RiskConfig` 里？还是 `StrategyRouterConfig`？语义上更适合 router
- **建议加到 `StrategyRouterConfig`**：`open_action_threshold: float = 0.5`

---

## 6. 完整新 config（`configs/default.yaml`）

```yaml
data:
  exchange: okx
  symbols: [BTC-USDT-SWAP, ETH-USDT-SWAP]
  main_interval: "15m"                    # 变
  confirm_interval: "1H"                  # 变（反过来）
  history_window_days: 730

feature_engine:
  version: "0.1.0"
  cache_enabled: true

strategy_router:
  composer: weighted_avg
  open_action_threshold: 0.40             # 新字段（原 0.5 硬编码，放松到 0.4）
  enabled_strategies:
    trend_following:
      weight: 1.0
      config_file: strategies/trend_following.yaml

risk:
  risk_per_trade_pct: 3.0                 # 变
  atr_stop_multiplier: 1.0                # 变（从 2.5 收紧）
  reward_risk_ratio: 2.0                  # 变（从 2.5 收紧）
  max_leverage: 20                        # 变（从 3）
  daily_max_loss_pct: 10.0                # 变（从 3）
  hold_timeout_bars: 32                   # 变（32 × 15m = 8h）
  trailing_enabled: true                  # 新
  trailing_activation_r: 1.0              # 新
  trailing_atr_multiplier: 1.0            # 新

execution:
  fees: {maker: 0.0002, taker: 0.0005}
  slippage_atr_multiplier: 0.1
  funding_settlement: true

backtest:
  start: "2024-07-05"
  end: "2026-07-05"
  initial_capital: 1000                   # 变（从 10000）

report:
  output_dir: reports
  ai_context: true

hard_rules:
  enabled: true
  min_quote_volume_24h: 1000000.0
  atr_pct_max_multiplier: 5.0
  atr_pct_baseline_window: 500
```

---

## 7. 预期结果范围（我的直觉，回测前）

| 指标 | 悲观情况 | 中位情况 | 乐观情况 |
|---|---|---|---|
| 交易数（2 年） | 300 | **700-1000** | 1500 |
| 一天平均 | 0.4 | **1.0-1.4** | 2.0 |
| 收益率 | -50% | **+80% ~ +200%** | +500% |
| Sharpe | 0.5 | **0.9-1.2** | 1.5 |
| 最大回撤 | 40% | **50-70%** | 30% |
| 胜率 | 35% | **38-42%** | 45% |

**关键风险点**：15m + tight stop (1×ATR) 天然会被 wick 大量扫单，胜率大概率下降。移动止损能挽回一部分（锁盈了不吐回去），但整体上高频短打的胜率就是天然低于低频趋势跟随。

**如果没到 3 单/天**：这几乎是必然。真高频只能靠加均值回归策略（Phase 1b）解决。目前先看单一 trend_following 在 15m 上表现是不是可接受。

---

## 8. 交付路径

1. 你 review 这份 spec，确认参数
2. 我先写 trailing 代码 + 单元测试（约 2 subagent 轮次）
3. 改 config schema + config yaml + engine 传参
4. 删 feature cache → 重建 features on 15m（3-5 分钟）
5. 跑回测（1-2 分钟）
6. 出报告，跟 v0.1a-tuned 并列对比

**回滚路径**：所有改动都在 `phase-1b-scalp` 分支上做，v0.1a-tuned 保持不动。如果实测很糟，直接切回 main / v0.1a-tuned tag。
