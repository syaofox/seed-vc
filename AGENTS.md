# Seed-VC 开发指南

本文档为 AI 代理提供代码开发规范指导。

## 项目概述

Seed-VC 是零样本语音转换模型，支持零样本语音转换、实时语音转换、歌声转换及自定义数据微调。

**技术栈**: Python 3.10+, PyTorch, Gradio, Hydra

## 环境配置

```bash
# 使用 uv 管理虚拟环境（推荐）
uv sync
pip install -r requirements.txt
```

- Python 3.10+
- GPU（实时转换推荐 NVIDIA RTX 3060+）
- 实时 GUI 需要 Tkinter 支持

## 常用命令

### 推理

```bash
# V1 模型
python inference.py --source <源音频> --target <参考音频> --output <输出目录> \
    --diffusion-steps 25 --fp16 True

# V2 模型
python inference_v2.py --source <源音频> --target <参考音频> --output <输出目录> \
    --diffusion-steps 25 --cfm-checkpoint-path <cfm模型> --ar-checkpoint-path <ar模型>
```

### 训练

```bash
# V1 模型微调
python train.py --config <配置文件> --dataset-dir <数据集> \
    --run-name <名称> --batch-size 2 --max-steps 1000

# V2 模型微调（支持多卡）
uv run accelerate launch train_v2.py --dataset-dir <数据集> \
    --run-name <名称> --batch-size 2 --max-steps 1000 --train-cfm --train-ar
```

### Web UI

```bash
python app_vc.py --checkpoint <模型> --config <配置> --fp16 True
python app_vc_v2.py --cfm-checkpoint-path <cfm> --ar-checkpoint-path <ar>
python app.py --enable-v1 --enable-v2
python real-time-gui.py --checkpoint-path <模型> --config-path <配置>
```

### Lint / 测试

```bash
# 运行 ruff 检查
uv run ruff check .

# 自动修复
uv run ruff check . --fix

# 代码格式化
uv run ruff format .

# 运行测试（如有）
uv run pytest tests/ -v
uv run pytest tests/test_file.py::test_function -v
```

## 代码规范

### 导入排序

遵循标准 Python 导入顺序，使用 ruff 自动管理：

```python
# 标准库
import os
import sys
import json
from typing import Optional, Tuple, Dict, List

# 第三方库
import torch
import numpy as np
import librosa
import yaml

# 本地模块
from modules.commons import str2bool
from optimizers import build_optimizer
```

### 命名规范

- 类名：PascalCase（如 `AttrDict`, `Trainer`）
- 函数/变量：snake_case（如 `get_padding`, `max_steps`）
- 常量：全大写 snake_case（如 `MAX_SR`）

### 类型注解

为公共函数添加类型注解：

```python
def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)

def build_model(config: Dict, stage: str) -> Optional[torch.nn.Module]:
    ...
```

### 错误处理

捕获具体异常，避免空捕获：

```python
# 推荐
try:
    config = yaml.safe_load(open(config_path))
except FileNotFoundError:
    raise FileNotFoundError(f"Config file not found: {config_path}")

# 避免
try:
    config = yaml.safe_load(open(config_path))
except:
    pass
```

### 格式化

- 最大行长度：120 字符
- 使用 ruff 格式化
- ruff 配置（pyproject.toml）：忽略 E402

## 注意事项

- HuggingFace 访问：如遇网络问题，使用镜像 `HF_ENDPOINT=https://hf-mirror.com`
- 模型下载：首次运行自动下载，默认缓存 `./checkpoints/hf_cache`
- 实时转换延迟 ≈ Block Time × 2 + Extra context (right) + 设备延迟(~100ms)
- Mac 运行 `real-time-gui.py` 报错 `_tkinter` 缺失：需安装支持 Tkinter 的 Python

## 预训练模型

| 版本 | 模型 | 用途 | 采样率 |
|------|------|------|--------|
| v1.0 | seed-uvit-tat-xlsr-tiny | 实时 VC | 22050 |
| v1.0 | seed-uvit-whisper-small-wavenet | 离线 VC | 22050 |
| v1.0 | seed-uvit-whisper-base | SVC | 44100 |
| v2.0 | hubert-bsqvae-small | VC/口音转换 | 22050 |

配置文件位于 `configs/presets/` 目录。
