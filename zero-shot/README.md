# Chronos Zero-Shot Forecasting

基于 Chronos 模型的零样本时序预测脚本，用于光伏发电功率预测。

## 支持的模型

| 模型 | 模型名称 | 支持协变量 | 说明 |
|------|----------|-----------|------|
| **Chronos2** | `amazon/chronos-2` | ✅ 是 | 最新模型，支持温度、辐照度等协变量 |
| **Chronos-Bolt** | `amazon/chronos-bolt-{tiny,mini,small,base}` | ❌ 否 | 快速推理模型，直接输出分位数 |
| **Chronos1** | `amazon/chronos-t5-{tiny,mini,small,base,large}` | ❌ 否 | 原始模型，基于采样生成预测 |

## 安装依赖

```bash
pip install torch transformers pandas numpy matplotlib tqdm
pip install chronos-forecasting  # 或从本地安装
```

## 使用方法

### 查看帮助

```bash
python chronos_zero_shot.py --help
```

### 基本用法

```bash
# 使用默认参数（Chronos2 + 协变量）
python chronos_zero_shot.py

# 使用 Chronos-Bolt 模型
python chronos_zero_shot.py --model amazon/chronos-bolt-small

# 使用 Chronos1 模型
python chronos_zero_shot.py --model amazon/chronos-t5-base
```

### 完整示例

```bash
# Chronos2 带协变量
python chronos_zero_shot.py \
    --model amazon/chronos-2 \
    --data-dir /path/to/data \
    --output-dir /path/to/output \
    --context-days 7 \
    --forecast-days 1 \
    --n-windows 20 \
    --batch-size 100

# Chronos-Bolt（自动禁用协变量）
python chronos_zero_shot.py \
    --model amazon/chronos-bolt-base \
    --context-days 3 \
    --forecast-days 1 \
    --stride-days 1 \
    --n-windows 10 \
    --gpu 0

# Chronos1 大模型
python chronos_zero_shot.py \
    --model amazon/chronos-t5-large \
    --context-days 7 \
    --forecast-days 1 \
    --batch-size 50 \
    --device cuda

# Chronos2 禁用协变量（仅使用功率数据）
python chronos_zero_shot.py \
    --model amazon/chronos-2 \
    --no-covariates
```

## 参数说明

### 数据路径

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | `/home/ubuntu/efs/to_customer/zendure/data` | 包含 CSV 数据文件的目录 |
| `--output-dir` | `/home/ubuntu/efs/to_customer/zendure/zero-shot` | 输出结果保存目录 |
| `--split-csv` | `OUTPUT_DIR/data_split.csv` | 数据划分文件路径 |

### 模型设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `amazon/chronos-2` | 模型名称（见支持的模型列表） |
| `--use-covariates` | `True` | 使用协变量（温度、辐照度），仅 Chronos2 有效 |
| `--no-covariates` | `False` | 禁用协变量，强制使用单变量模式 |
| `--device` | `cuda` | 运行设备（`cuda` 或 `cpu`） |
| `--gpu` | `0` | GPU 设备 ID |

### 预测设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--context-days` | `1` | 历史上下文天数（用于预测的历史数据长度） |
| `--forecast-days` | `1` | 预测天数（每个窗口预测的未来天数） |

### 滑动窗口设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stride-days` | `1` | 滑动窗口步长（天），1 表示每天一个窗口 |
| `--n-windows` | `10` | 每个序列的窗口数量，0 表示尽可能多 |
| `--min-context-days` | 同 `context-days` | 最小上下文天数要求 |

### 其他设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | `100` | 批处理大小（同时处理的序列数） |
| `--seasonality` | `48` | 季节性周期（48 = 30分钟数据的24小时） |
| `--freq` | `30min` | 数据频率 |

## 输出文件

脚本会在 `--output-dir` 目录下生成以下文件：

| 文件 | 说明 |
|------|------|
| `predictions_all_windows.csv` | 所有窗口的预测结果 |
| `metrics_per_series.csv` | 每个序列的评估指标 |
| `metrics_overall.csv` | 整体评估指标 |
| `forecast_*.png` | 预测可视化图（示例序列） |

## 数据格式

### 输入数据

每个 CSV 文件应包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `date` | datetime | 时间戳 |
| `power` | float | 功率值（预测目标） |
| `temperature` | float | 温度（协变量，可选） |
| `irradiance` | float | 辐照度（协变量，可选） |

### data_split.csv

数据划分文件格式：

```csv
filename,split
file1.csv,train
file2.csv,test
...
```

## 注意事项

1. **协变量支持**：只有 Chronos2 支持协变量（温度、辐照度）。使用 Chronos-Bolt 或 Chronos1 时会自动切换到单变量模式。

2. **内存使用**：大模型（如 `chronos-t5-large`）需要更多 GPU 内存，可以通过减小 `--batch-size` 来降低内存使用。

3. **预测长度**：Chronos-Bolt 有固定的预测长度限制，超出时会自动分块预测。

4. **数据频率**：确保 `--freq` 参数与实际数据频率一致，默认为 30 分钟间隔。
