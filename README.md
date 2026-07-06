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

## Phase 1a 完成清单

- [x] 数据采集：OKX 2 年 BTC/ETH 1H + 15m + funding + OI
- [x] 特征引擎：EMA/ADX/RSI/BB/ATR + 价格行为 + Regime 自动打标 + `quote_volume_24h` + `atr_pct_baseline`
- [x] 打分引擎：BaseStrategy 接口 + 硬约束 wired-in + trend_following v0.1.0
- [x] Strategy Router：weighted_avg composer
- [x] Risk Engine：ATR sizing + 单日熔断
- [x] BacktestExecutor：手续费 + 滑点 + 资金费（funding 精确单次入账）
- [x] Ledger + BacktestEngine：主循环 + 止损/止盈/超时 + 跨日 daily_realized 归零
- [x] 报告层：Markdown + Parquet + PNG，AI 可学习格式（trades/snapshots/ai_context/config_snapshot/charts）
- [x] CLI + Docker + Makefile
- [x] 三层测试：单元 + 集成 + e2e（61 tests）
- [x] 15m 确认层：真实取自 15m 指标（不是 1H 自举）

下一步：Phase 1b（加均值回归策略 + 多策略合成验证）

## 最近一次回测报告

见 `reports/` 目录最新时间戳子目录。示例路径：`reports/2026-07-06-0800/report.md`。
