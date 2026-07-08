# Shadow 验证协议 · v0.1.10

**启动日**: 2026-07-08
**目的**: 在投入真实资金前，用影子模式验证 v0.1.10 冠军配置的实盘行为
**部署加固**: shadow_24h_run + rabbit-frontend 均已设 `restart: unless-stopped`
（Mac 重启 / Docker 更新后自动恢复，ledger 状态落盘持久化）

## 被验证的配置指纹

| 项 | 值 |
|---|---|
| 策略 | trend_following v0.1.3 (extreme momentum gate) + ml_scoring (LightGBM) |
| regime_rules | range: 禁多留空 |
| stop_mode | atr (2.5×)，结构止损已测 REJECT |
| funding_weight | 0.0（Binance 不可达的临时措施） |
| 回测基线 | Sharpe 5.86 / MaxDD -5.2% / 125 笔/2年 / 胜率 56.8% |
| drift 基线 | baselines/v0.1.10.json |

## 期望管理（重要）

回测频率 125 笔/2 年 ≈ **全组合每周 1.2 笔**。因此：

**几周内能验证的**：
1. 机器完整性 —— 运行时间、数据新鲜度、零 error 心跳
2. 特征分布一致性 —— features_log vs 基线（每天 216 个数据点，统计充足）
3. 逐笔规则合规 —— 每笔实际开单的入场快照必须满足开单规则
   （extreme momentum + regime rules + 硬规则），人工复核
4. 执行真实性 —— 成交价 vs 盘口价差、资金费结算正确性

**几周内不能验证的**（数学上不可能）：
- 胜率 / Sharpe / PF —— 需要 30-50 笔 ≈ **6-10 个月**
- 聚类漂移 —— drift 检测 min_live_trades=20，同样需要月级

结论：验证分两阶段。**阶段一（2-4 周）验机器**，通过后可考虑 2-3 倍
杠杆的小额真实资金；**阶段二（持续）验 edge**，胜率显著偏离基线时
drift 告警触发复审。

## 每周检查清单

```bash
# 1. 心跳与健康（应为 healthy，分钟级延迟）
docker exec rabbit-frontend rabbit shadow watchdog --state-dir shadows
curl -s http://localhost:8080/api/shadow/state | python3 -m json.tool

# 2. 数据源（应 9/9 健康）
curl -s http://localhost:8080/api/data/health | python3 -c "import json,sys; print(json.load(sys.stdin)['summary'])"

# 3. 特征分布漂移（每周一次，ALERT 则立即复审）
docker exec rabbit-frontend rabbit shadow feature-drift \
  --baseline-path baselines/features_v0.1.3.json

# 4. 统一摘要（可读版，包含全部分析）
docker exec rabbit-frontend rabbit shadow digest \
  --trade-baseline baselines/v0.1.10.json \
  --feature-baseline baselines/features_v0.1.3.json

# 5. 有新交易时：逐笔复核入场快照
curl -s http://localhost:8080/api/shadow/trades | python3 -m json.tool
```

或直接看前端: http://localhost:8080 （实况页 = 心跳带 + 通道 + 日志）

## 阶段一通过标准（2-4 周后评估）

| # | 标准 | 门槛 |
|---|---|---|
| 1 | 运行时间 | 无超过 6h 的静默中断（watchdog 无 DOWN） |
| 2 | 数据完整 | data/health 缺口 = 0 |
| 3 | 特征漂移 | feature-drift 无 ALERT（或 ALERT 有可解释的市场原因） |
| 4 | 规则合规 | 每笔交易入场快照 100% 满足开单条件 |
| 5 | 执行真实 | 成交价与盘口偏差 < 0.1% |
| 6 | 无违规单 | 零笔 range+long（regime rule 生效验证） |

全部通过 → 可开启小额真实资金（$1k @ 2-3 倍有效杠杆起步）。
任一不通过 → 修复后重新计时。

## 关键日志事件速查

| 事件 | 含义 |
|---|---|
| `tick_done` | 处理了新 bar（每小时应出现） |
| `shadow_entry` | ★开单了 —— 触发逐笔复核 |
| `shadow_exit` | 平仓（止损/止盈/超时） |
| `metrics_alert` | 告警（回撤/滞后/连续错误） |
| `archive_write_failed` | 归档失败（数据管道问题） |
