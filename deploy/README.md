# Chronos-2 部署与测试工具

本目录包含 Chronos-2 模型的部署、测试和性能评估工具：

| 脚本 | 用途 |
|------|------|
| `deploy_to_sagemaker.py` | 将微调模型部署为 SageMaker real-time endpoint |
| `test_sagemaker_endpoint.py` | 对已部署的 endpoint 进行精度和延迟测试 |
| `benchmark.py` | 本地推理性能和精度的综合评估 |

---

## deploy_to_sagemaker.py — SageMaker 部署

将微调后的 Chronos-2 模型部署为 SageMaker real-time endpoint。

### 工作流程

1. 通过 JumpStart 获取 Chronos-2 的容器镜像和基础模型 artifacts（含推理代码）
2. 下载基础 artifacts，用微调 checkpoint 替换模型权重
3. 打包为 tar.gz 上传到 S3
4. 创建 SageMaker Model 并部署为 real-time endpoint
5. 自动 smoke test 验证 endpoint 可用

### 使用方法

```bash
# 部署（使用默认参数）
python deploy/deploy_to_sagemaker.py

# 自定义参数
python deploy/deploy_to_sagemaker.py \
    --model_path output/chronos2-336-48-covariant-full-100k-w-covariates-covariate_dropout-4-gpus/finetuned-ckpt \
    --endpoint_name chronos2-finetuned-zendure \
    --instance_type ml.g5.2xlarge \
    --region us-west-2

# 删除 endpoint（停止计费）
python deploy/deploy_to_sagemaker.py --delete --endpoint_name chronos2-finetuned-zendure
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `output/chronos2-...-4-gpus/finetuned-ckpt` | 微调模型本地路径 |
| `--endpoint_name` | `chronos2-finetuned-zendure` | SageMaker endpoint 名称 |
| `--instance_type` | `ml.g5.2xlarge` | 部署实例类型 |
| `--initial_instance_count` | `1` | 初始实例数 |
| `--region` | 自动检测 | AWS 区域 |
| `--s3_bucket` | SageMaker 默认 bucket | 存放模型 artifacts 的 S3 bucket |
| `--s3_prefix` | `chronos2-finetuned-zendure` | S3 路径前缀 |
| `--role` | 自动检测 | SageMaker 执行角色 ARN |
| `--delete` | `False` | 删除 endpoint 而非部署 |

> **注意**：endpoint 运行期间会持续产生费用，不用时请及时删除。

---

## test_sagemaker_endpoint.py — Endpoint 测试

对已部署的 SageMaker endpoint 进行精度（MAE）和延迟测试，数据加载和评估方法与 `benchmark.py` 对齐。

### 测试内容

1. **Smoke Test** — 用合成数据验证 endpoint 可用
2. **MAE 评估** — 滑动窗口方式评估预测精度（批量请求 endpoint）
3. **延迟测试** — 单样本延迟统计 + 批量吞吐量

### 使用方法

```bash
# 基础测试（无协变量）
python deploy/test_sagemaker_endpoint.py \
    --endpoint_name chronos2-finetuned-zendure

# 带协变量的完整测试
python deploy/test_sagemaker_endpoint.py \
    --endpoint_name chronos2-finetuned-zendure \
    --use_covariates \
    --num_samples 50 \
    --n_windows 10
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--endpoint_name` | `chronos2-finetuned-zendure` | 要测试的 endpoint 名称 |
| `--data_dir` | `data/` | 数据目录 |
| `--split_csv` | `zero-shot/data_split.csv` | 数据划分文件 |
| `--context_length` | `336` | 上下文长度 |
| `--prediction_length` | `48` | 预测长度（延迟测试用） |
| `--use_covariates` | `False` | 是否使用协变量（temperature, irradiance） |
| `--num_samples` | `20` | test series 数量上限 |
| `--n_windows` | `5` | 每个 series 的滑动窗口数 |
| `--stride_days` | `1` | 窗口滑动步长（天） |
| `--forecast_days` | `1` | 每个窗口的预测天数 |
| `--batch_size` | `20` | 每次 endpoint 请求包含的时间序列数 |
| `--quantile_levels` | `0.1,0.5,0.9` | 请求的分位数 |
| `--region` | 自动检测 | AWS 区域 |
| `--output` | `deploy/endpoint_test_results.json` | 结果输出路径 |

### 请求数据格式

每条时间序列以 dict 形式发送，结构取决于是否使用协变量：

```python
# 无协变量
{"target": [1.2, 3.4, 5.6, ...]}   # context_length 个 float

# 有协变量
{
    "target": [1.2, 3.4, 5.6, ...],           # context_length 个值 (336)
    "past_covariates": {
        "temperature": [25.1, 26.3, ...],      # context_length 个值
        "irradiance": [800.0, 850.2, ...],     # context_length 个值
    },
    "future_covariates": {
        "temperature": [27.0, 28.1, ...],      # prediction_length 个值 (48)
        "irradiance": [900.0, 920.5, ...],     # prediction_length 个值
    },
}
```

发送时多条时间序列放入 `inputs` 列表，一次请求批量处理：

```python
payload = {
    "inputs": [entry_1, entry_2, ...],   # batch_size 条时间序列
    "parameters": {"prediction_length": 48, "quantile_levels": [0.1, 0.5, 0.9]}
}
```

### Batch 机制

| 场景 | batch 方式 | 说明 |
|------|-----------|------|
| MAE 评估 | 按 `--batch_size`（默认 20）分批 | 所有滑动窗口样本分批发送，加速评估 |
| 单样本延迟 | 逐条发送 | 测量每次请求的 round-trip 时间 |
| 批量延迟 | 固定 10/50/100 | 测量不同 batch size 下的吞吐量 |

> **注意**：SageMaker endpoint payload 有大小限制（默认 6MB），单次请求不宜发送过多样本。

### 输出格式

```json
{
  "config": { "endpoint_name": "...", "use_covariates": true, "..." : "..." },
  "mae": {
    "num_windows": 20310,
    "overall_mae": 153.53,
    "std_mae": 201.53,
    "median_mae": 90.07,
    "per_step_mae": [97.22, 119.88, "...48 values..."]
  },
  "latency": {
    "single_sample": { "count": 2050, "mean_ms": 36.83, "p50_ms": 36.87, "p99_ms": 39.39 },
    "batch": {
      "10":  { "batch_size": 10, "throughput_samples_per_sec": 150.3, "per_sample_ms": 6.7 },
      "50":  { "batch_size": 50, "throughput_samples_per_sec": 280.5, "per_sample_ms": 3.6 },
      "100": { "batch_size": 100, "throughput_samples_per_sec": 350.1, "per_sample_ms": 2.9 }
    }
  }
}
```

---

## benchmark.py — 本地推理 Benchmark

对 Chronos-2 模型进行本地推理性能和预测精度的综合评估，包含以下测试项：

1. **Zero-Shot MAE** — 滑动窗口评估预测精度（与 `zero-shot/` 方法对齐）
2. **Single Sample Latency** — 单样本推理延迟统计
3. **Batch Throughput** — 不同 batch size 下的吞吐量
4. **Concurrency** — 多线程并发请求性能

## 快速开始

```bash
# 基础用法（zero-shot 模型）
python deploy/benchmark.py \
    --model_path amazon/chronos-2 \
    --data_dir data/ \
    --split_csv zero-shot/data_split.csv \
    --use_covariates

# 微调模型
python deploy/benchmark.py \
    --model_path output/chronos2-336-48-covariant-full-100k/finetuned-ckpt \
    --data_dir data/ \
    --split_csv zero-shot/data_split.csv \
    --use_covariates
```

## 参数说明

### 模型与数据

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `amazon/chronos-2` | 模型路径或 HuggingFace model ID |
| `--data_dir` | `data/` | 数据目录 |
| `--split_csv` | `zero-shot/data_split.csv` | 数据划分文件，从中读取 test split 的文件列表 |
| `--use_covariates` | `False` | 是否使用协变量（temperature, irradiance） |
| `--dtype` | `bfloat16` | 模型精度，可选 `bfloat16` / `float32` |

### 样本控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_samples` | `50` | 使用的 test series 数量上限（截断文件列表） |

> **注意**：`num_samples` 限制的是 **series（文件）数量**，不是总评估样本数。
> 实际 test 文件数取决于 `data_split.csv` 中 test split 的条目数。
> 如果 test 只有 60 个文件，设 5000 也只能用 60 个。

### MAE 评估参数（滑动窗口）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--forecast_days` | `1` | 每个窗口的预测天数（1天 = 48 步，30min 间隔） |
| `--n_windows` | `10` | 每个 series 的滑动窗口数 |
| `--stride_days` | `1` | 窗口之间的滑动步长（天） |

MAE 总评估样本数 = `有效 series 数 x n_windows`

### 延迟/吞吐参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--context_length` | `336` | 上下文长度（数据点数） |
| `--prediction_length` | `48` | 预测长度（用于延迟/吞吐测试） |
| `--warmup_runs` | `5` | 正式计时前的预热次数（排除 CUDA 初始化开销） |
| `--batch_sizes` | `1,4,8,16,32,64` | 测试的 batch size 列表 |
| `--concurrency_levels` | `1,2,4,8` | 测试的并发线程数列表 |

### 输出

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output` | `deploy/benchmark_results.json` | 结果输出路径 |

## 评估方法详解

### Zero-Shot MAE（滑动窗口）

为了与 `zero-shot/chronos_zero_shot.py` 的评估结果对齐，MAE 计算采用相同的方法：

```
Series: ──────────────────────────────────────────────────►
                                        time

Window 1:  [...context_length...]|[forecast_days]|
Window 2:       [...context_length...]|[forecast_days]|
Window 3:            [...context_length...]|[forecast_days]|
                          ◄─stride_days─►
```

**数据预处理**（与 zero-shot 一致）：
- Resample 到 30min 等间隔频率
- 线性插值填补缺失值
- 边界用 bfill/ffill 处理

**预测方式**：
- 使用 `pipeline.predict()` 获取分位数预测
- 取 median (0.5 分位数) 作为点预测
- 计算与 ground truth 的 MAE

**输出指标**：
- `overall_mae` — 所有窗口的平均 MAE
- `std_mae` / `min_mae` / `max_mae` / `median_mae` — MAE 分布统计
- `per_step_mae` — 每个预测步的平均 MAE（可观察误差随预测步长的增长趋势）

### Latency Benchmark

单样本延迟测试流程：
1. **Warmup** — 先跑 N 次预热，消除 CUDA kernel 编译、内存分配等一次性开销
2. **Timed runs** — 对每个样本单独计时，统计 mean / p50 / p90 / p99 / min / max

### Batch Throughput

对每个 batch size：
1. Warmup 1 次（同时检测 OOM）
2. 将所有样本按 batch size 分组，逐批推理并计时
3. 计算 `throughput = total_samples / total_time`

### Concurrency

模拟多个并发请求打到同一个模型的场景：
- 使用线程池发送单样本请求
- 统计 QPS（每秒处理请求数）和每个请求的延迟

> GPU 模型通常是串行执行的，并发不会提升吞吐，只会增加排队延迟。
> 真正提升并发能力需要：batch 聚合 或 多实例水平扩展。

## 输出格式

结果保存为 JSON，结构如下：

```json
{
  "config": {
    "model_path": "amazon/chronos-2",
    "context_length": 336,
    "prediction_length": 48,
    "use_covariates": true,
    "dtype": "bfloat16",
    "device": "cuda",
    "gpu_name": "NVIDIA L40S",
    "num_samples": 50,
    "mae_n_windows": 10,
    "mae_stride_days": 1.0,
    "mae_forecast_days": 1.0,
    "mae_total_windows": 490
  },
  "model_load": {
    "time_sec": 0.35,
    "gpu_mb": { "allocated_mb": 455.8, "peak_mb": 456.0 }
  },
  "zero_shot_mae": {
    "num_windows": 490,
    "overall_mae": 153.51,
    "std_mae": 200.12,
    "median_mae": 98.34,
    "per_step_mae": [116.3, 148.3, "...48 values..."]
  },
  "single_latency": {
    "count": 50,
    "mean_ms": 22.49,
    "p50_ms": 22.49,
    "p90_ms": 22.54,
    "p99_ms": 22.59
  },
  "batch_throughput": {
    "1":  { "throughput_samples_per_sec": 44.45, "..." : "..." },
    "32": { "throughput_samples_per_sec": 1119.91, "..." : "..." }
  },
  "concurrency": {
    "1": { "qps": 43.85, "mean_ms": 22.76, "..." : "..." },
    "8": { "qps": 40.94, "mean_ms": 193.31, "..." : "..." }
  }
}
```

## 已有测试结果参考（NVIDIA L40S）

| 指标 | 值 |
|------|-----|
| 模型加载时间 | 0.35s |
| GPU 显存占用 | ~456 MB |
| 单样本推理延迟 | 22.5 ms (mean) |
| 最佳批处理吞吐 | 1120 samples/s (batch_size=32) |
| 并发 QPS | ~44 (并发数无显著增益) |
| Zero-shot MAE | 192.09 (2050 series, 单窗口) |

## 与 zero-shot 评估的关系

| 维度 | `zero-shot/chronos_zero_shot.py` | `deploy/benchmark.py` |
|------|----------------------------------|----------------------|
| 目的 | 完整的精度评估 | 部署性能 + 快速精度验证 |
| 预测 API | `pipeline.predict_df()` (DataFrame) | `pipeline.predict()` (Tensor) |
| 滑动窗口 | 按天滑动，时间戳对齐 | 按数据点索引滑动 |
| 缺失值处理 | resample + 线性插值 | resample + 线性插值（已对齐） |
| 输出指标 | MAE, RMSE, sMAPE, MASE, R2 等 | MAE + per-step MAE |
| 额外功能 | 分位数区间、可视化、per-series 指标 | 延迟、吞吐、并发、GPU 显存 |

如需完整的精度评估（包含所有指标），请使用 `zero-shot/chronos_zero_shot.py`。
`benchmark.py` 和 `test_sagemaker_endpoint.py` 的 MAE 用于快速验证模型在部署环境下的预测质量是否正常。

---

## 典型工作流

```bash
# 1. 部署微调模型到 SageMaker
python deploy/deploy_to_sagemaker.py \
    --model_path output/chronos2-336-48-covariant-full-100k-w-covariates-covariate_dropout-4-gpus/finetuned-ckpt

# 2. 测试 endpoint 精度和延迟
python deploy/test_sagemaker_endpoint.py \
    --endpoint_name chronos2-finetuned-zendure \
    --use_covariates \
    --num_samples 50

# 3. 不用时删除 endpoint
python deploy/deploy_to_sagemaker.py --delete --endpoint_name chronos2-finetuned-zendure
```
