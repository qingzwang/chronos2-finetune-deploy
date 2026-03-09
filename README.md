# Chronos-2 Fine-tune & Deploy

基于 [Amazon Chronos-2](https://github.com/amazon-science/chronos-forecasting) 的光伏发电功率预测方案，涵盖数据划分、Zero-shot 评估、模型微调、性能测试和 SageMaker 部署的完整流程。

## 项目结构

```
.
├── data/                   # 光伏数据（13,765 个 CSV 文件）
├── zero-shot/              # 数据划分 & Zero-shot 评估
├── finetune/               # 模型微调训练
├── deploy/                 # 推理性能测试 & SageMaker 部署
├── output/                 # 训练输出（checkpoint、日志）
└── chronos-forecasting/    # Chronos 源码（依赖）
```

## 数据格式

每个 CSV 文件代表一个光伏站点，30 分钟采样间隔：

```csv
date,power,temperature,irradiance
2024-08-16 00:00:00,0.0,0.0,0.0
2024-08-16 00:30:00,0.0,19.5,0.0
2024-08-16 01:00:00,15.3,20.1,120.5
```

| 列 | 说明 |
|---|---|
| `power` | 发电功率（目标变量） |
| `temperature` | 温度（协变量） |
| `irradiance` | 辐照度（协变量） |

数据集划分：11,700 train / 2,065 test（85%/15%）。

---

## 快速开始

### 1. 环境准备

```bash
pip install -U "sagemaker<3"
pip install chronos-forecasting[training] torch accelerate
```

### 2. 数据划分

```bash
python zero-shot/split_data.py
```

生成 `zero-shot/data_split.csv`，随机划分 train/test（seed=42）。

### 3. Zero-shot 评估（基线）

使用预训练 Chronos-2 直接预测，不做任何训练：

```bash
python zero-shot/chronos_zero_shot.py \
    --model amazon/chronos-2 \
    --data-dir data/ \
    --split-csv zero-shot/data_split.csv \
    --use-covariates \
    --forecast-days 1 \
    --n-windows 10
```

输出预测结果和多维度评估指标（MAE、RMSE、sMAPE、MASE、R²）。

### 4. 微调训练

```bash
bash finetune/run_training.sh \
    --model_type chronos2 \
    --num_gpus 4 \
    --batch_size 32 \
    --num_steps 100000 \
    --learning_rate 1e-5 \
    --context_length 336 \
    --prediction_length 48 \
    --use_covariates true
```

或使用 YAML 配置：

```bash
bash finetune/run_training.sh --config finetune/configs/chronos2.yaml
```

训练输出保存在 `output/` 目录。

### 5. 本地推理测试

```bash
python deploy/benchmark.py \
    --model_path output/chronos2-336-48-covariant-full-100k-w-covariates-covariate_dropout-4-gpus/finetuned-ckpt \
    --use_covariates
```

测试内容：Zero-shot MAE、单样本延迟、批处理吞吐量、并发性能。

### 6. 部署到 SageMaker

```bash
# 部署
python deploy/deploy_to_sagemaker.py \
    --model_path output/chronos2-336-48-covariant-full-100k-w-covariates-covariate_dropout-4-gpus/finetuned-ckpt \
    --endpoint_name chronos2-finetuned \
    --instance_type ml.g5.2xlarge \
    --region us-east-1

# 测试 endpoint
python deploy/test_sagemaker_endpoint.py \
    --endpoint_name chronos2-finetuned \
    --region us-east-1 \
    --use_covariates \
    --num_samples 50

# 用完删除（停止计费）
python deploy/deploy_to_sagemaker.py --delete --endpoint_name chronos2-finetuned
```

---

## 模块详情

### zero-shot/ — 数据划分与 Zero-shot 评估

| 文件 | 说明 |
|------|------|
| `split_data.py` | 数据集 train/test 划分 |
| `chronos_zero_shot.py` | Zero-shot 推理（滑动窗口评估） |
| `evaluate_metrics.py` | 指标计算（MAE、RMSE、sMAPE、MASE、R²） |
| `data_split.csv` | 划分结果文件 |

评估采用滑动窗口方式，支持协变量输入：

```
Series: ──────────────────────────────────────────────────►
Window 1:  [...context...]|[forecast]|
Window 2:       [...context...]|[forecast]|
                     ◄─stride─►
```

详见 [zero-shot/README.md](zero-shot/README.md)。

### finetune/ — 模型微调

| 文件 | 说明 |
|------|------|
| `train.py` | 训练主脚本 |
| `config.py` | 训练配置（dataclass） |
| `run_training.sh` | 多 GPU 训练启动脚本 |
| `metrics.py` | 训练指标 |
| `data/dataset.py` | 数据加载与预处理 |
| `data/preprocessing.py` | 数据增强（Jitter、Scaling、Magnitude Warping） |
| `trainers/` | 模型训练器（chronos1、chronos2、chronos_bolt） |
| `configs/` | YAML 配置模板 |

支持三种模型：

| 模型 | 协变量 | 分布式训练 | 微调方式 |
|------|--------|-----------|---------|
| Chronos-2 | temperature, irradiance | accelerate（多 GPU） | Full / LoRA |
| Chronos-Bolt | 不支持 | accelerate | Full / LoRA |
| Chronos-1 | 不支持 | torchrun | Full |

关键训练参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `context_length` | 512 | 输入上下文长度（数据点数） |
| `prediction_length` | 48 | 预测长度（48 步 = 24 小时） |
| `num_steps` | 10,000 | 训练步数 |
| `learning_rate` | 1e-5 | 学习率 |
| `use_covariates` | false | 是否使用协变量（仅 Chronos-2） |

详见 [finetune/README.md](finetune/README.md)。

### deploy/ — 推理测试与 SageMaker 部署

| 文件 | 说明 |
|------|------|
| `benchmark.py` | 本地推理性能 Benchmark（延迟、吞吐、MAE） |
| `deploy_to_sagemaker.py` | 部署微调模型到 SageMaker real-time endpoint |
| `test_sagemaker_endpoint.py` | Endpoint 精度和延迟测试 |

部署流程：下载 JumpStart 基础 artifacts → 替换为微调权重 → 打包上传 S3 → 创建 endpoint。

详见 [deploy/README.md](deploy/README.md)。

---

## 性能参考

### 预测精度（MAE）

| 模型 | 协变量 | MAE | 评估方式 |
|------|--------|-----|---------|
| Chronos-2（预训练） | 有 | ~192 | 滑动窗口，2065 series |
| Chronos-2（微调） | 有 | ~149 | 滑动窗口，488 windows |

### 推理性能（ml.g5.2xlarge）

| 指标 | 本地 GPU | SageMaker Endpoint |
|------|---------|-------------------|
| 单样本延迟 | ~22 ms | ~37 ms |
| 最佳吞吐量 | ~900 samples/s | ~125 samples/s |
| SageMaker overhead | — | ~6 ms |

> SageMaker 吞吐量受 JSON 序列化开销限制。需要更高吞吐可水平扩展（多实例）。

---

## 完整工作流

```
数据准备                  评估基线                 微调训练
split_data.py ──────► chronos_zero_shot.py ──────► run_training.sh
     │                      │                          │
     ▼                      ▼                          ▼
data_split.csv         基线 MAE/RMSE            output/finetuned-ckpt/
                                                       │
                    ┌──────────────────────────────────┘
                    │
              本地验证                        部署上线
          benchmark.py ──────────► deploy_to_sagemaker.py
                │                          │
                ▼                          ▼
         精度 + 延迟确认           SageMaker Endpoint
                                           │
                                           ▼
                                test_sagemaker_endpoint.py
                                           │
                                           ▼
                                    线上精度 + 延迟确认
```
