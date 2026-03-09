# Chronos 模型微调模块

本模块提供 Chronos 时序预测模型的微调训练代码，支持 Chronos-1、Chronos-2 和 Chronos-Bolt 三种模型。

## 目录结构

```
finetune/
├── __init__.py                  # 模块入口
├── config.py                    # 配置类定义
├── metrics.py                   # 评估指标 (MAE, MSE, RMSE, sMAPE)
├── train.py                     # 主训练入口
├── run_training.sh              # 启动脚本
├── analyze_data.py              # 数据质量分析脚本
├── configs/                     # YAML 配置文件
│   ├── chronos1.yaml
│   ├── chronos2.yaml
│   └── chronos_bolt.yaml
├── data/                        # 数据加载模块
│   ├── __init__.py
│   ├── dataset.py               # 数据集类
│   └── preprocessing.py         # 数据预处理
└── trainers/                    # 训练器
    ├── __init__.py
    ├── chronos1_trainer.py      # Chronos-1 训练器
    ├── chronos2_trainer.py      # Chronos-2 训练器
    └── chronos_bolt_trainer.py  # Chronos-Bolt 训练器
```

## 模型对比

| 特性 | Chronos-1 | Chronos-2 | Chronos-Bolt |
|------|-----------|-----------|--------------|
| 架构 | T5 Seq2Seq | Encoder-Decoder | Encoder-Decoder |
| 损失函数 | Cross-Entropy | Quantile Loss | Quantile Loss |
| 协变量支持 | ❌ | ✅ | ❌ |
| LoRA 微调 | ❌ | ✅ | ❌ |
| 学习率调度 | Linear | Cosine Annealing | Linear |
| 分布式训练 | torchrun | Accelerate | Accelerate |
| HuggingFace ID | `amazon/chronos-t5-*` | `amazon/chronos-2` | `amazon/chronos-bolt-*` |

## 数据格式

### 输入数据 (CSV)

每个 CSV 文件包含一个时序样本：

```csv
date,power,temperature,irradiance
2025-09-28 00:00:00,0.0,0.0,0.0
2025-09-28 00:30:00,0.0,0.0,0.0
2025-09-28 01:00:00,0.0,6.04,0.0
...
```

| 列名 | 说明 |
|------|------|
| `date` | 时间戳 (30分钟间隔) |
| `power` | 目标变量 (功率) |
| `temperature` | 协变量 (温度，仅 Chronos-2 使用) |
| `irradiance` | 协变量 (辐照度，仅 Chronos-2 使用) |

### 数据划分文件 (data_split.csv)

```csv
filename,split
231870.csv,train
204220.csv,train
...
258949.csv,test
```

### Chronos-2 协变量处理

Chronos-2 训练时，协变量数据格式：

```python
{
    "target": tensor([power...]),              # 完整序列
    "past_covariates": {
        "temperature": tensor([temp...]),      # 完整序列 (包含未来值)
        "irradiance": tensor([irr...]),        # 完整序列 (包含未来值)
    },
    "future_covariates": {
        "temperature": None,                   # 标记为已知未来协变量
        "irradiance": None,                    # 值设为 None，训练时自动切片
    },
}
```

协变量在模型内部处理流程：
1. 与目标变量拼接成多变量序列
2. 各变量独立做 Instance Normalization
3. 通过 Transformer 的 Group Attention 学习变量间关系
4. 训练时随机切片，future_covariates 从 past_covariates 对应位置提取

## 配置参数

### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_type` | str | `chronos2` | 模型类型: `chronos1`, `chronos2`, `chronos_bolt` |
| `model_id` | str | - | HuggingFace 模型 ID |
| `finetune_mode` | str | `full` | 微调模式: `full` (全参数), `lora` (仅 Chronos-2) |

### 数据参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_dir` | str | `/home/ubuntu/.../data/` | CSV 数据目录 |
| `split_csv` | str | `/home/ubuntu/.../data_split.csv` | 数据划分文件 |
| `context_length` | int | `512` | 历史上下文长度 |
| `prediction_length` | int | `48` | 预测长度 (48 = 24小时 @ 30分钟间隔) |
| `use_covariates` | bool | `False` | 是否使用协变量 (仅 Chronos-2) |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `batch_size` | int | `32` | 每个 GPU 的批次大小 |
| `learning_rate` | float | `1e-5` | 学习率 |
| `num_steps` | int | `10000` | 训练步数 |
| `val_ratio` | float | `0.1` | 验证集比例 (0 = 不使用验证集) |
| `gradient_accumulation_steps` | int | `1` | 梯度累积步数 |
| `warmup_steps` | int | `0` | 预热步数 |
| `weight_decay` | float | `0.0` | 权重衰减 |
| `max_grad_norm` | float | `1.0` | 梯度裁剪阈值 |

### LoRA 参数 (仅 Chronos-2)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lora_rank` | int | `8` | LoRA 秩 |
| `lora_alpha` | int | `32` | LoRA alpha 缩放参数 |
| `lora_dropout` | float | `0.1` | LoRA dropout |

### 输出参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_dir` | str | `./output/` | 模型保存目录 |
| `logging_steps` | int | `100` | 日志打印间隔 |
| `save_steps` | int | `1000` | 模型保存间隔 |
| `eval_steps` | int | `500` | 验证间隔 |

### 精度参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `bf16` | bool | `True` | 使用 BFloat16 (推荐 Ampere GPU) |
| `fp16` | bool | `False` | 使用 Float16 |

### 数据增强参数 (仅 Chronos-2)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `augmentation` | bool | `False` | 是否启用数据增强 |
| `aug_jitter_sigma` | float | `0.03` | Jittering 噪声强度 (相对于序列标准差) |
| `aug_scaling_sigma` | float | `0.1` | Scaling 缩放变化范围 |
| `aug_mag_warp_sigma` | float | `0.2` | Magnitude Warping 强度 (围绕 1.0 的变化幅度) |
| `aug_mag_warp_knots` | int | `4` | Magnitude Warping 控制点数量 (越多曲线变化越频繁) |
| `aug_prob` | float | `0.5` | 每种增强的应用概率 |

## 数据增强

数据增强仅在训练时应用，且只对 **target (功率)** 进行增强，不对协变量 (温度、辐照度) 增强。

### 增强方式与作用范围

| 增强方式 | 作用于 Context | 作用于 Target | 说明 |
|----------|----------------|---------------|------|
| **Jitter** | ✅ | ❌ | 只对历史输入加噪声 |
| **Scaling** | ✅ | ✅ | 同一因子同时缩放，保持相对关系 |
| **Magnitude Warping** | ✅ | ✅ | 同一曲线平滑过渡，无跳变 |

### Jittering (添加噪声)

向 **context（历史输入）** 添加高斯噪声，模拟测量误差。**不对 target 加噪声**，保持标签准确性。

```python
# 计算原序列标准差
std_val = std(context)

# 生成噪声：标准差 = aug_jitter_sigma * 原序列标准差
noise = randn() * aug_jitter_sigma * std_val

# 只添加到 context
context = context + noise
# target 保持不变
```

**示例** (aug_jitter_sigma=0.03):
```
原始 context:  [100, 150, 200, 180, 120]
噪声:          [-1.2, 0.8, 2.1, -0.5, 1.5]
增强后 context: [98.8, 150.8, 202.1, 179.5, 121.5]
target:        保持原值
```

### Scaling (随机缩放)

对 **context 和 target 同时** 乘以相同的随机因子，模拟不同容量/效率的设备：

```python
# 从正态分布采样缩放因子
scale_factor = normal(1.0, aug_scaling_sigma)

# 限制范围在 [0.8, 1.2]
scale_factor = clip(scale_factor, 0.8, 1.2)

# 同一因子同时应用于 context 和 target
context = context * scale_factor
target = target * scale_factor
```

**示例** (aug_scaling_sigma=0.1, scale_factor=1.08):
```
原始 context:  [100, 150, 200, 180, 120]  →  增强后: [108, 162, 216, 194, 130]
原始 target:   [90, 60]                   →  增强后: [97, 65]
相对关系保持不变
```

### Magnitude Warping (幅度变形)

使用**三次样条插值**生成平滑的 warping 曲线，对 **context 和 target 同时** 应用，模拟渐变的天气变化：

```python
# 生成 knots 个控制点，值在 1.0 附近随机波动
knot_values = normal(1.0, aug_mag_warp_sigma, size=knots+2)
knot_values = clip(knot_values, 0.5, 1.5)

# 用三次样条插值生成平滑曲线
warp_curve = CubicSpline(knot_positions, knot_values)

# 曲线覆盖整个 context + target 长度
# 同一曲线的不同部分分别应用于 context 和 target
context = context * warp_curve[:context_len]
target = target * warp_curve[context_len:]
```

**示例** (aug_mag_warp_sigma=0.2, knots=4):
```
warping 曲线:
factor: 1.1──╮    ╭──1.05──╮    ╭──0.95
             ╰────╯        ╰────╯
时间:   [........context........][...target...]
                 平滑过渡，无跳变

效果: 序列的不同部分被不同程度地放大/缩小，但变化是平滑的
```

### 增强效果示意

```
原始功率曲线（context + target）:
      context (历史)       target (预测目标)
    ╭───────────╮         ╭───╮
   ╱             ╲       ╱     ╲
  ╱               ╲     ╱       ╲
──╯                 ╰───╯         ╰──

Jittering 后:                    Scaling 后 (1.1x):
    ╭~─~─~─~─~─╮         ╭───╮       ╭───────────╮         ╭───╮
   ╱ ~       ~  ╲       ╱     ╲     ╱             ╲       ╱     ╲
  ╱~            ~╲     ╱       ╲   ╱               ╲     ╱       ╲
──╯              ~╰───╯         ╰──────╯                 ╰───╯         ╰───
  context 有波动    target 不变      整体抬高，context 和 target 同比例

Magnitude Warping 后:
    ╭─────────────╮      ╭──╮
   ╱               ╲    ╱    ╲
  ╱    (放大)       ╲__╱(缩小)╲___
──╯                              ╰──
      曲线渐变，平滑过渡
```

### 设计要点

| 设计点 | 原因 |
|--------|------|
| Jitter 只对 context | 保持 target 标签准确，避免学习噪声 |
| Scaling 对两者都加 | 保持 context-target 相对关系，模拟不同设备容量 |
| Mag Warp 对两者都加 | 平滑曲线避免边界跳变，模拟天气渐变 |
| 噪声强度相对于 std | 不同设备功率量级不同 (1kW vs 100kW)，用相对值更通用 |
| 限制 0.8-1.2 / 0.5-1.5 | 防止极端变换导致不合理数据 |
| 只增强 target 序列 | 协变量 (温度、辐照度) 是真实测量/预报值，不应改变 |
| 结果 clamp(min=0) | 太阳能功率不能为负 |
| 只在训练时增强 | 验证和测试需要评估真实数据上的表现 |

### 推荐参数

| 场景 | jitter_sigma | scaling_sigma | mag_warp_sigma | knots | prob |
|------|--------------|---------------|----------------|-------|------|
| 保守（默认） | 0.03 | 0.1 | 0.2 | 4 | 0.5 |
| 激进 | 0.05 | 0.15 | 0.3 | 6 | 0.7 |
| 只 scaling | 0.0 | 0.1 | 0.0 | 4 | 0.8 |
| 只 jittering | 0.03 | 0.0 | 0.0 | 4 | 0.8 |
| 只 mag warp | 0.0 | 0.0 | 0.2 | 4 | 0.8 |

### 配置示例

```yaml
# configs/chronos2.yaml
augmentation: true
aug_jitter_sigma: 0.03       # Jitter 强度
aug_scaling_sigma: 0.1       # Scaling 强度
aug_mag_warp_sigma: 0.2      # Magnitude Warping 强度
aug_mag_warp_knots: 4        # Warping 曲线控制点数量
aug_prob: 0.5                # 每种增强的应用概率
```

## 学习率调度

### Chronos-2: Cosine Annealing + Warmup

```
学习率曲线 (warmup_steps=500, num_steps=10000):

lr_max ─────╮    ╭─────────╮
            │   ╱           ╲
            │  ╱             ╲
            │ ╱               ╲
         0 ─┴─────────────────────→ steps
            0   500          10000
            ↑
          warmup
```

- **Warmup 阶段**: 学习率从 0 线性增加到 `learning_rate`
- **Cosine 阶段**: 学习率按余弦曲线衰减

```
warmup:  lr = lr_max * (step / warmup_steps)
cosine:  lr = lr_max * 0.5 * (1 + cos(π * progress))
```

### Chronos-1 / Chronos-Bolt: Linear Decay

学习率从 `learning_rate` 线性衰减到 0。

## 使用方法

### 方式一：使用 YAML 配置文件 (推荐)

```bash
# 使用预定义配置
python finetune/train.py --config finetune/configs/chronos2.yaml

# 配置文件 + 命令行覆盖 (命令行参数优先)
python finetune/train.py \
    --config finetune/configs/chronos2.yaml \
    --num_steps 20000 \
    --warmup_steps 1000
```

### 方式二：使用启动脚本

```bash
# Chronos-2 全参数微调 (带协变量)
./finetune/run_training.sh \
    --model_type chronos2 \
    --num_gpus 4 \
    --use_covariates \
    --batch_size 32 \
    --num_steps 10000 \
    --val_ratio 0.1

# Chronos-2 LoRA 微调
./finetune/run_training.sh \
    --model_type chronos2 \
    --finetune_mode lora \
    --use_covariates \
    --learning_rate 1e-4

# Chronos-1 微调
./finetune/run_training.sh \
    --model_type chronos1 \
    --num_gpus 4

# Chronos-Bolt 微调
./finetune/run_training.sh \
    --model_type chronos_bolt \
    --num_gpus 4

# 不使用验证集 (全部数据用于训练)
./finetune/run_training.sh \
    --model_type chronos2 \
    --val_ratio 0
```

### 方式三：直接调用 Python

```bash
# Chronos-2 (使用 Accelerate)
accelerate launch --multi_gpu --num_processes=4 finetune/train.py \
    --model_type chronos2 \
    --model_id amazon/chronos-2 \
    --use_covariates \
    --batch_size 32 \
    --num_steps 10000 \
    --warmup_steps 500

# Chronos-1 (使用 torchrun)
torchrun --nproc_per_node=4 finetune/train.py \
    --model_type chronos1 \
    --model_id amazon/chronos-t5-small

# Chronos-Bolt (使用 Accelerate)
accelerate launch --multi_gpu --num_processes=4 finetune/train.py \
    --model_type chronos_bolt \
    --model_id amazon/chronos-bolt-small
```

## 评估指标

### Chronos-2 训练日志

训练过程中显示 loss 和学习率：

```
{'loss': 1.9536, 'grad_norm': 12.17, 'learning_rate': 9.901e-06, 'epoch': 0.01}
{'eval_loss': 2.0146, 'eval_runtime': 0.31, 'epoch': 0.01}
{'loss': 1.7690, 'grad_norm': 5.71, 'learning_rate': 9.801e-06, 'epoch': 0.02}
```

- `loss`: 训练 Quantile Loss
- `eval_loss`: 验证 Quantile Loss
- `grad_norm`: 梯度范数
- `learning_rate`: 当前学习率

### 训练结束后评估

训练完成后自动计算完整指标：

```
Evaluation results:
  mae: 12.3456
  mse: 234.5678
  rmse: 15.3158
  smape: 18.45
```

| 指标 | 公式 | 说明 |
|------|------|------|
| **MAE** | `mean(\|pred - target\|)` | 平均绝对误差 |
| **MSE** | `mean((pred - target)²)` | 均方误差 |
| **RMSE** | `sqrt(MSE)` | 均方根误差 |
| **sMAPE** | `200 * mean(\|pred - target\| / (\|pred\| + \|target\|))` | 对称平均绝对百分比误差 |

> 注：使用 sMAPE 而非 MAPE，因为 sMAPE 能正确处理目标值为 0 的情况。

### Chronos-Bolt 训练日志

```
Step 1000/10000 | Loss: 0.0234 | LR: 9.00e-06
  Validation: loss: 0.0256 | mae: 12.34 | mse: 234.56 | rmse: 15.31 | smape: 18.45
```

## 数据处理流程

### 总体流程

```
data_split.csv (11,700 train files)
         │
         ▼
┌─────────────────────────────────────┐
│  按 val_ratio 随机划分               │
│  (使用 seed 保证可复现)              │
└─────────────────────────────────────┘
         │                    │
         ▼                    ▼
   Train Files           Val Files
   (1 - val_ratio)       (val_ratio)
         │                    │
         ▼                    ▼
┌─────────────────────────────────────┐
│  转换为模型输入格式                  │
│  - target: power 序列               │
│  - past_covariates: 温度、辐照度     │
│  - future_covariates: 标记已知协变量 │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  训练时随机切片                      │
│  context_length + prediction_length │
└─────────────────────────────────────┘
         │
         ▼
    Training Batches
```

### Chronos-2 详细数据流

Chronos-2 使用内置的 `Pipeline.fit()` 方法训练，数据经过以下转换：

#### 步骤 1: CSV → Dict 格式 (`_convert_files_to_inputs`)

```python
# 输入: CSV 文件
# date, power, temperature, irradiance
# 2025-09-28 00:00:00, 0.0, 15.2, 0.0
# ...

# 输出: Dict 格式 (带协变量)
{
    "target": tensor([0.0, 0.0, ...]),           # shape: (5032,) 完整 power 序列
    "past_covariates": {
        "temperature": tensor([15.2, ...]),     # shape: (5032,) 完整序列
        "irradiance": tensor([0.0, ...]),       # shape: (5032,) 完整序列
    },
    "future_covariates": {
        "temperature": None,                    # None = 标记为已知未来协变量
        "irradiance": None,                     # 训练时自动从 past_covariates 切片
    },
}

# 输出: Dict 格式 (无协变量)
tensor([0.0, 0.0, ...])  # shape: (5032,) 仅 power 序列
```

#### 步骤 2: Dict → Chronos2Dataset (`Chronos2Dataset.convert_inputs`)

```python
# Chronos2Dataset._prepare_tasks() 处理:
# 1. future_covariates 的 None 值填充为 np.full(prediction_length, np.nan)
# 2. validate_and_prepare_single_dict_task() 转换为 tensor 格式

# 转换后的内部格式:
task_context_tensor        # shape: (n_targets + n_covariates, seq_len)
                           # = (1 + 2, 5032) = (3, 5032)
                           # 第 0 行: power (target)
                           # 第 1 行: irradiance
                           # 第 2 行: temperature

task_future_covariates     # shape: (3, prediction_length)
                           # 第 0 行: NaN (target 未来未知)
                           # 第 1 行: irradiance 未来值
                           # 第 2 行: temperature 未来值
```

#### 步骤 3: 训练时随机切片 (`_construct_slice`)

```
原始序列 (5032 时间步):
┌──────────────────────────────────────────────────────┐
│  0    ...   slice_idx-context_len  ...  slice_idx  ...  slice_idx+pred_len  ...  5031  │
└──────────────────────────────────────────────────────┘
                    │                        │                    │
                    └────── context ─────────┘                    │
                                             └──── future_target ─┘

# 训练模式: 随机选择 slice_idx
slice_idx = random.randint(min_past, seq_len - prediction_length)

# 切片结果:
context         # shape: (3, context_length)     历史数据
future_target   # shape: (3, prediction_length)  未来目标 (用于计算 loss)
future_covariates # shape: (3, prediction_length) 已知未来协变量
```

#### 步骤 4: 构建 Batch (`_build_batch`)

```python
# 多个 task 拼接成一个 batch
batch = {
    "context": tensor,           # shape: (batch_size, context_length)
    "future_target": tensor,     # shape: (batch_size, prediction_length)
    "future_covariates": tensor, # shape: (batch_size, prediction_length)
    "group_ids": tensor,         # shape: (batch_size,) 标识哪些行属于同一个 task
    "num_output_patches": int,   # 输出 patch 数量
}
```

#### 完整流程图

```
CSV 文件 (power, temperature, irradiance)
    │
    ▼ load_data_from_csv()
DataFrame
    │
    ▼ _convert_files_to_inputs()
List[Dict] 格式
    │  {
    │    "target": tensor (5032,),
    │    "past_covariates": {"temp": tensor, "irr": tensor},
    │    "future_covariates": {"temp": None, "irr": None}
    │  }
    │
    ▼ Chronos2Dataset.convert_inputs()
Chronos2Dataset (IterableDataset)
    │  内部存储: List[Tuple[task_context, task_future, n_targets, ...]]
    │
    ▼ __iter__() 训练迭代
每个 step 随机切片
    │  slice_idx = random(min_past, seq_len - pred_len)
    │  context = task_context[:, slice_idx-ctx_len : slice_idx]
    │  future_target = task_context[:, slice_idx : slice_idx+pred_len]
    │
    ▼ _build_batch()
Batch Dict
    │  context:           (batch_size, context_length)
    │  future_target:     (batch_size, prediction_length)
    │  future_covariates: (batch_size, prediction_length)
    │  group_ids:         (batch_size,)
    │
    ▼ HuggingFace Trainer
Model Forward → Quantile Loss → Backward
```

#### 关键点说明

1. **`future_covariates = None` 的作用**
   - 不是传递未来值，而是**标记**哪些协变量是"已知未来"的
   - 训练时，Chronos2Dataset 会自动从 `past_covariates` 的完整序列中切片出对应位置的未来值

2. **为什么 `past_covariates` 包含未来值**
   - 对于训练数据，我们有完整的历史记录（包括"未来"时间点）
   - `past_covariates` 存储完整序列，训练时随机切片，切片点之后的数据就是"未来"

3. **group_ids 的作用**
   - 标识哪些行（target + covariates）属于同一个预测任务
   - 模型通过 Group Attention 学习同一任务内变量间的关系

### 万级文件处理详解

以 10,000 个 CSV 文件为例，详细说明每一步的处理过程。

#### 步骤 1: 加载所有文件 (`_convert_files_to_inputs`)

```python
# 输入: 10,000 个 CSV 文件路径
file_paths = ["001.csv", "002.csv", ..., "10000.csv"]

# 输出: List[Dict]，长度 = 10,000
inputs = [
    # 文件 001.csv
    {
        "target": tensor([p0, p1, ..., p5031]),           # shape: (5032,)
        "past_covariates": {
            "temperature": tensor([t0, t1, ..., t5031]), # shape: (5032,)
            "irradiance": tensor([i0, i1, ..., i5031]),  # shape: (5032,)
        },
        "future_covariates": {
            "temperature": None,
            "irradiance": None,
        },
    },
    # 文件 002.csv
    {...},
    # ... 共 10,000 个 dict
]
```

#### 步骤 2: 预处理 (`Chronos2Dataset._prepare_tasks`)

```python
# 对每个 dict 调用 validate_and_prepare_single_dict_task()
# 转换成内部 tensor 格式

tasks = []  # 长度 = 10,000

for idx, raw_task in enumerate(inputs):  # 遍历 10,000 个文件
    task = (
        task_context_tensor,      # shape: (3, 5032) = (target + 2协变量, 序列长度)
        task_future_tensor,       # shape: (3, 48)   = (3, prediction_length)
        task_n_targets,           # = 1 (只有 power 是 target)
        task_n_covariates,        # = 2 (temperature, irradiance)
        task_n_future_covariates, # = 2 (两个都是已知未来协变量)
    )
    tasks.append(task)

# 最终: tasks 是长度为 10,000 的列表，全部加载到内存
```

#### 步骤 3: 训练迭代 (`__iter__` - 无限循环)

```python
def _generate_train_batches(self):
    while True:  # 无限循环，直到达到 num_steps
        current_batch_size = 0
        task_indices = []

        # 随机选择 task，直到凑够 batch_size
        while current_batch_size < self.batch_size:  # batch_size=32
            task_idx = np.random.randint(len(self.tasks))  # 从 0~9999 随机选
            task_indices.append(task_idx)
            # 每个 task 贡献 3 行 (1 target + 2 covariates)
            current_batch_size += 3

        # 例如: task_indices = [4521, 892, 7234, 156, ...]
        # 11 个 task × 3 行/task = 33 行 >= batch_size(32)

        yield self._build_batch(task_indices)
```

#### 步骤 4: 随机切片 (`_construct_slice`)

```python
def _construct_slice(self, task_idx):
    # 取出第 task_idx 个文件的数据
    task_context_tensor = self.tasks[task_idx][0]  # shape: (3, 5032)

    # 训练模式: 随机选择切片起点
    # slice_idx 范围: [min_past, seq_len - prediction_length] = [48, 4984]
    slice_idx = np.random.randint(48, 4984)  # 例如 slice_idx = 2000

    # 切片 context (历史) - 往前取 context_length 个点
    context = task_context_tensor[:, 2000-336 : 2000]  # shape: (3, 336)

    # 切片 future_target (用于计算 loss)
    future_target = task_context_tensor[:, 2000 : 2048]  # shape: (3, 48)

    return context, future_target, future_covariates, n_targets=1
```

**图示：**
```
task_context_tensor: shape (3, 5032)
┌─────────────────────────────────────────────────────────────────────────┐
│ power:       [p0, ..., p1664, ..., p1999, p2000, ..., p2047, ..., p5031]│
│ irradiance:  [i0, ..., i1664, ..., i1999, i2000, ..., i2047, ..., i5031]│
│ temperature: [t0, ..., t1664, ..., t1999, t2000, ..., t2047, ..., t5031]│
└─────────────────────────────────────────────────────────────────────────┘
                      │                   │              │
                slice_idx-336        slice_idx     slice_idx+48
                      │                   │              │
                      └──── context ──────┘              │
                          (336 points)                   │
                                          └─ future_target ─┘
                                              (48 points)
```

#### min_past 与 context_length 的区别

这两个参数容易混淆，但作用完全不同：

| 参数 | 含义 | 默认值 | 作用 |
|------|------|--------|------|
| `min_past` | 切片点之前**至少**需要的历史点数 | 48 | 决定可切片位置数量 |
| `context_length` | 送入模型的**最大**历史长度 | 336 | 决定模型看到的历史长度 |

**切片代码逻辑** (`chronos2/dataset.py:485-499`):
```python
# 1. 确定切片位置 (受 min_past 约束)
slice_idx = np.random.randint(self.min_past, full_length - self.prediction_length + 1)

# 2. 提取 context (受 context_length 约束)
if slice_idx >= self.context_length:
    # 历史足够长 → 取最近 context_length 个点
    context = data[:, slice_idx - self.context_length : slice_idx]
else:
    # 历史不够长 → 取全部可用历史
    context = data[:, :slice_idx]
```

**不同 slice_idx 的效果：**
```
序列长度 = 5000, min_past = 48, context_length = 336

slice_idx = 48   → context 长度 = 48   (最短，等于 min_past)
slice_idx = 100  → context 长度 = 100  (不足 context_length)
slice_idx = 336  → context 长度 = 336  (刚好达到 context_length)
slice_idx = 2000 → context 长度 = 336  (超过则截断)
slice_idx = 4952 → context 长度 = 336  (最大切片位置)
```

**图示：**
```
序列: [0, 1, 2, ..., 47, 48, ..., 335, 336, ..., 4951, 4952, ..., 4999]
                     │                │
                 min_past=48    context_length=336
                     │                │
      ←───────────────┼────────────────┼─────────────────────────────→
      切片位置范围: [48 ························· 4952]
                     │                │
                     │                └─ 超过此位置，context 被截断为 336
                     │
                     └─ 最小切片位置，context 长度 = 48
```

**为什么需要两个参数？**
- `min_past` 较小 (48) → 允许更多切片位置 → 增加数据多样性
- `context_length` 较大 (336) → 模型能看到更长历史 → 捕捉长期模式

#### 步骤 5: 组装 Batch (`_build_batch`)

```python
def _build_batch(self, task_indices):
    # task_indices = [4521, 892, 7234, ..., 5678]  # 11 个随机选的 task

    for group_id, task_idx in enumerate(task_indices):
        # 对每个 task 做随机切片
        context, future_target, future_covariates, _ = self._construct_slice(task_idx)
        # context shape: (3, 336)
        # group_ids: [0,0,0], [1,1,1], [2,2,2], ...

    # 拼接所有 task: 11 个 task × 3 行/task = 33 行
    return {
        "context": ...,           # shape: (33, 336)
        "future_target": ...,     # shape: (33, 48)
        "future_covariates": ..., # shape: (33, 48)
        "group_ids": ...,         # [0,0,0, 1,1,1, 2,2,2, ..., 10,10,10]
    }
```

**Batch 结构图示：**
```
context: shape (33, 336)
┌────────────────────────────────────────────┐
│ task 4521 - power       │ 336 时间步       │  group_id = 0
│ task 4521 - irradiance  │ 336 时间步       │  group_id = 0
│ task 4521 - temperature │ 336 时间步       │  group_id = 0
├────────────────────────────────────────────┤
│ task 892  - power       │ 336 时间步       │  group_id = 1
│ task 892  - irradiance  │ 336 时间步       │  group_id = 1
│ task 892  - temperature │ 336 时间步       │  group_id = 1
├────────────────────────────────────────────┤
│ ...                     │ ...              │  ...
├────────────────────────────────────────────┤
│ task 5678 - power       │ 336 时间步       │  group_id = 10
│ task 5678 - irradiance  │ 336 时间步       │  group_id = 10
│ task 5678 - temperature │ 336 时间步       │  group_id = 10
└────────────────────────────────────────────┘
     共 33 行 (11 tasks × 3 variates)
```

#### 步骤 6: 送入模型训练

```python
# HuggingFace Trainer 内部循环
for step in range(num_steps):  # 例如 10,000 步
    batch = next(train_dataset)

    # 模型前向传播
    predictions = model(
        context=batch["context"],           # (33, 336)
        group_ids=batch["group_ids"],       # (33,) 用于 Group Attention
        future_covariates=batch["future_covariates"],
    )
    # predictions: (33, 9, 48) = (batch, num_quantiles, pred_len)

    # 计算 Quantile Loss
    loss = quantile_loss(predictions, batch["future_target"])

    # 反向传播
    loss.backward()
    optimizer.step()
```

#### 完整训练流程图

```
10,000 个 CSV 文件
        │
        ▼ _convert_files_to_inputs()
List[Dict] 长度 10,000
        │  每个 dict: {target: (5032,), past_covariates: {...}, ...}
        │
        ▼ Chronos2Dataset._prepare_tasks()
List[Tuple] 长度 10,000
        │  每个 tuple: (context_tensor(3,5032), future_tensor(3,48), 1, 2, 2)
        │
        ▼ 全部加载到内存
        │
        │  ╔═══════════════════════════════════════════════════════════╗
        │  ║  训练循环 (无限迭代，直到 num_steps)                        ║
        │  ╠═══════════════════════════════════════════════════════════╣
        │  ║                                                           ║
        │  ║  Step 1: 随机选 ~11 个 task (凑够 batch_size=32)           ║
        │  ║          task_indices = [4521, 892, 7234, ...]            ║
        │  ║                          │                                ║
        │  ║                          ▼                                ║
        │  ║  Step 2: 每个 task 随机切片 (每次位置都不同!)               ║
        │  ║          slice_idx = random(48, 4984)                     ║
        │  ║          context = data[:, idx-336:idx]                   ║
        │  ║          future = data[:, idx:idx+48]                     ║
        │  ║                          │                                ║
        │  ║                          ▼                                ║
        │  ║  Step 3: 拼接成 batch                                     ║
        │  ║          context: (33, 336)                               ║
        │  ║          future_target: (33, 48)                          ║
        │  ║          group_ids: [0,0,0, 1,1,1, ..., 10,10,10]         ║
        │  ║                          │                                ║
        │  ║                          ▼                                ║
        │  ║  Step 4: 模型前向 + Quantile Loss + 反向传播               ║
        │  ║                                                           ║
        │  ╚═══════════════════════════════════════════════════════════╝
        │
        ▼ 重复 num_steps 次 (例如 10,000 次)
训练完成
```

#### 关键数字汇总

| 阶段 | 数量 | 说明 |
|------|------|------|
| CSV 文件数 | 10,000 | 原始数据文件 |
| tasks 列表长度 | 10,000 | 每个文件 → 一个 task |
| 每个 task 的 variates | 3 | 1 target + 2 covariates |
| 每个 task 的序列长度 | ~5,032 | 约 105 天的数据 |
| 每个 batch 的 tasks 数 | ~11 | 随机选择，直到总行数 ≥ 32 |
| 每个 batch 的行数 | ~33 | 11 tasks × 3 variates |
| context_length | 336 | 7 天历史数据 |
| prediction_length | 48 | 预测 24 小时 |
| 切片位置 | 随机 | 每个 task 每次迭代切片位置不同 |
| 总训练步数 | num_steps | 例如 10,000 步 |

#### 数据增强效果

由于每次迭代对每个 task 随机切片，同一个文件在不同 step 会产生不同的训练样本：

```
文件 001.csv (5032 时间步):
  Step 100:  切片位置 2000 → context[1664:2000], future[2000:2048]
  Step 500:  切片位置 3500 → context[3164:3500], future[3500:3548]
  Step 1000: 切片位置 1200 → context[864:1200],  future[1200:1248]
  ...
```

这相当于数据增强，10,000 个文件可以产生大量不同的训练样本。

### 训练数据量计算

以下示例计算模型在训练过程中实际看到的数据量。

#### 示例参数

```
num_steps = 100,000                 # 优化器更新次数
batch_size = 32                     # 每次前向传播的行数
gradient_accumulation_steps = 4    # 梯度累积步数
文件数 = 10,000                     # CSV 文件数量
```

#### 计算公式

```
1. 前向传播次数
   = num_steps × gradient_accumulation_steps
   = 100,000 × 4
   = 400,000 次

2. 每次前向传播的数据量
   - batch_size = 32 是"行数"（包含 target + covariates）
   - 每个 task 有 3 行（1 power + 1 temperature + 1 irradiance）
   - 每个 batch 约有 11 个 task（因为 11 × 3 = 33 ≥ 32）

3. 总数据量
   - 总行数 = 400,000 × 33 ≈ 13,200,000 行
   - 总 task 切片数 = 400,000 × 11 ≈ 4,400,000 个切片
```

#### 结果汇总

| 指标 | 数值 | 说明 |
|------|------|------|
| 优化器更新次数 | 100,000 | = num_steps |
| 前向传播次数 | 400,000 | = num_steps × grad_accum |
| 每 batch 行数 | ~33 | ≈ 11 tasks × 3 variates |
| 总切片数 | ~4,400,000 | = 400,000 × 11 |
| 每文件平均采样次数 | ~440 | = 4,400,000 / 10,000 |

#### 图示

```
                    num_steps = 100,000 (优化器更新)
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    grad_accum=4        grad_accum=4        grad_accum=4
          │                   │                   │
     ┌────┴────┐         ┌────┴────┐         ┌────┴────┐
     │ │ │ │   │         │ │ │ │   │         │ │ │ │   │
    4次前向   ...        4次前向   ...        4次前向   ...
     │                   │                   │
  每次~11个task       每次~11个task       每次~11个task
     │                   │                   │
    ~44个task           ~44个task           ~44个task
     切片/step           切片/step           切片/step

总切片数 = 100,000 × 4 × 11 ≈ 4,400,000
```

#### 与数据集大小的对比

```
数据集:
  - 10,000 个文件
  - 每个文件 ~5,000 个时间步
  - 可切片位置: ~4,900 个 (5000 - min_past - pred_len)
    - min_past = 48: 切片点之前至少需要 48 个历史点
    - pred_len = 48: 切片点之后需要 48 个未来点用于计算 loss

理论最大切片数 = 10,000 × 4,900 = 49,000,000
实际采样切片数 = 4,400,000
覆盖率 ≈ 4,400,000 / 49,000,000 ≈ 9%
```

#### 每文件采样示例

```
文件 001.csv 被采样 ~440 次，每次切片位置随机:
  采样 #1:   切片位置 2000
  采样 #2:   切片位置 3500
  采样 #3:   切片位置 1200
  ...
  采样 #440: 切片位置 4100
```

#### 快速计算公式

```python
# 计算模型看到的总切片数
total_slices = num_steps * gradient_accumulation_steps * (batch_size // 3)

# 计算每个文件平均被采样次数
samples_per_file = total_slices / num_files

# 示例
total_slices = 100000 * 4 * 11 = 4,400,000
samples_per_file = 4400000 / 10000 = 440
```

## 数据质量分析

使用 `analyze_data.py` 分析数据质量：

```bash
# 基本用法 (默认 48 个连续 0 = 24小时)
python finetune/analyze_data.py

# 设置连续 0 阈值为 96 (48小时)
python finetune/analyze_data.py --min_zeros 96

# 只分析训练集
python finetune/analyze_data.py --split train

# 输出详细结果到 CSV
python finetune/analyze_data.py --output data_quality.csv
```

输出示例：
```
【缺失值统计】
  power 缺失:     1,234 (0.00%)
  temperature 缺失: 5,678 (0.01%)

【连续 0 片段统计】(阈值: >= 48 个连续 0)
  总片段数:       8,234
  含片段的文件数: 6,543 (55.9%)
```

## 依赖安装

```bash
pip install torch transformers accelerate peft pandas numpy pyyaml
```

## 常见问题

### 1. 如何选择模型？

- **Chronos-2**: 推荐首选，支持协变量 (温度、辐照度)，效果最好
- **Chronos-Bolt**: 推理速度快，适合资源受限场景
- **Chronos-1**: 经典 T5 架构，兼容性好

### 2. 全参数微调 vs LoRA？

| 方式 | 优点 | 缺点 |
|------|------|------|
| **Full** | 效果更好 | 显存占用大，训练慢 |
| **LoRA** | 显存小，训练快 | 效果略差 |

推荐：显存充足用 `full`，显存紧张用 `lora`

### 3. 如何调整 batch_size？

- 单卡 A100 80GB: `batch_size=64`
- 单卡 A100 40GB: `batch_size=32`
- 单卡 V100 32GB: `batch_size=16`

显存不足时可增加 `gradient_accumulation_steps`：
```bash
--batch_size 16 --gradient_accumulation_steps 2  # 等效 batch_size=32
```

### 4. 训练多少步合适？

- 11,700 个文件约产生 ~100万个样本
- 推荐 `num_steps=10000~50000`
- 观察验证集 loss，不再下降时可停止

### 5. warmup_steps 设置多少？

推荐 `warmup_steps` 为总步数的 5%-10%：
- `num_steps=10000` → `warmup_steps=500~1000`
- `num_steps=50000` → `warmup_steps=2500~5000`

### 6. Quantile Loss 数值正常范围？

Chronos-2 的 loss 通常在 1.5~3.0 范围内：
- 初始 loss ~2.0 是正常的
- 训练后应下降到 1.5~1.8
- loss 不应出现 NaN 或剧烈波动

## 代码示例

### 加载微调后的模型进行预测

```python
import torch
from chronos import ChronosPipeline  # Chronos-1/Bolt
from chronos.chronos2 import Chronos2Pipeline  # Chronos-2

# Chronos-2
pipeline = Chronos2Pipeline.from_pretrained(
    "./output/chronos2/final_model",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

# 预测 (无协变量)
context = torch.tensor([...])  # shape: (history_length,)
forecast = pipeline.predict(
    context=context.unsqueeze(0),
    prediction_length=48,
)
median_prediction = forecast.median(dim=1).values  # shape: (1, 48)

# 预测 (带协变量)
forecast = pipeline.predict(
    context=context.unsqueeze(0),
    prediction_length=48,
    future_covariates={
        "temperature": torch.tensor([...]),  # shape: (48,)
        "irradiance": torch.tensor([...]),   # shape: (48,)
    },
)
```

### 自定义评估

```python
from finetune.metrics import compute_metrics

predictions = model_output  # numpy array
targets = ground_truth      # numpy array

metrics = compute_metrics(predictions, targets)
print(f"MAE: {metrics['mae']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"sMAPE: {metrics['smape']:.2f}%")
```
