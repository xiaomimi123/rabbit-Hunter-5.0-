# 实验归档：Kronos-mini K 线基础模型评测

**日期**: 2026-07-08
**结论**: **REJECT** — 不接入生产
**耗时**: ~4 小时（含环境搭建）

## 动机

操盘员提问：能否用 K 线基础模型（[Kronos](https://github.com/shiyu-coder/Kronos)，
MIT 协议，45+ 交易所数据预训练的 decoder-only Transformer）提升市场结构判断，
从而减轻震荡期连亏（2025Q2-Q4 累计 -$800 的历史回撤主因）。

## 环境

- Kronos-mini（4.1M 参数）+ Kronos-Tokenizer-2k
- torch 2.13 CPU / einops / safetensors
- HF 被墙：`huggingface_hub` 1.x 走 hf-mirror.com 的 snapshot_download 失败，
  **手动 urllib 拉 config.json + model.safetensors 可行**（各 ~16MB）
- 推理速度：0.6-1.0s/次（400 根 lookback → 24 根预测，CPU）

## 实验一：全年 walk-forward 方向/波动率预测（735 点）

BTC-USDT-SWAP 1H，每 12 小时一个评测点，每点预测未来 24 根。

### 方向准确率 — 死刑

| 尺度 | Kronos | 动量基线 | 基础上涨率 |
|---|---|---|---|
| 1 步 | 49.4% | 47.8% | 50.6% |
| 6 步 | 50.6% | 45.0% | 49.5% |
| 24 步 | 51.4% | 47.6% | 51.6% |

全部在 ±3.6pp 置信区间内 = 与掷硬币无差异。
高置信子集（|预测收益| top20%）准确率 48-49% —— 置信度不含信息。

### 波动率预测 — 单独不敌朴素基线

| 模型 | corr(预测, 实际 24h 波动率) |
|---|---|
| Kronos | 0.419 |
| 朴素持续性（明天=今天） | **0.431** |

联合回归 R²: 朴素 0.186 → +Kronos 0.250（+0.064）—— 存在与持续性正交的
微弱信息，触发实验二。

## 实验二：kronos_pred_vol_t0 作为 ML 特征的 A/B（裁决实验）

- 数据：2,996 笔回测交易（2024-07 → 2026-07，10 symbols），
  逐笔在入场前 400 根上跑 Kronos 推理（严格无未来函数），2,982 笔成功
- 设计：同一样本、同一 70/30 walk-forward 切分、同一 LightGBM 超参
  - 控制组：现役 18 特征
  - 处理组：18 + `kronos_pred_vol_t0`
- 裁决门槛：Δ test AUC ≥ +0.01（与 retrain promote 政策一致）

### 结果

| 组 | train AUC | test AUC |
|---|---|---|
| 控制 | 0.6787 | **0.5170** |
| 处理 | 0.6940 | **0.5097** |

**Δ test AUC = -0.0073 → REJECT**

train↑ + test↓ 的组合是噪音特征的教科书签名：模型 in-sample 拟合了该特征，
out-of-sample 反受其害。

## 为什么失败（事后解释）

1. 实验一的 +0.064 R² 波动率增量没有转化为交易结果预测力，因为现役特征里
   `atr_pct_t0` / `atr_pct_baseline_t0` 已覆盖波动率 regime —— Kronos 提供
   的是"已有信息的噪音副本"
2. Kronos-mini 只有 4.1M 参数，预训练分布 vs 加密 1H 永续可能错配
3. zero-shot 点预测丢掉了模型输出分布的形状信息（未测 sample_count>1 的
   分布统计量）

## 什么情况值得重开此实验

- Kronos-base/large（102M/499M）或后继模型开源且推理成本可接受
- 换任务框架：不做价格预测，改为在标注的 regime 数据上探测其 embedding
  的可分性（linear probe）
- 采样 sample_count=20 取预测分布的分位距/偏度作为特征（分布形状可能比
  点估计含更多 regime 信息）

## 可复现性

- 评测脚本：`vendor/kronos/rh_eval.py`（全年 walk-forward）、
  `vendor/kronos/rh_augment.py`（逐笔特征增强）、
  `vendor/kronos/rh_ab_train.py`（A/B 裁决）— vendor/ 不入库，
  重建方法见 git log eb84be2 的 commit message
- 结果数据：`vendor/kronos/rh_eval_results.parquet`（735 点）、
  `vendor/kronos/trades_kronos.parquet`（2,982 笔增强交易）

## 结论对系统方法论的印证

本实验从"听起来很合理的想法"到"数据裁决"用了半天，走完了完整的
promote/reject 管线。对照组设计（朴素基线、同样本 A/B、AUC 门槛）
拦截了三个如果直接接入就会发生的错误：
1. 小样本方向准确率 55% 的错觉（20 点冒烟 → 735 点归于 49%）
2. 波动率 corr 0.419 的错觉（输给朴素基线 0.431）
3. 回归 R² 增量 +0.064 的错觉（在真实交易分类任务上 -0.0073）
