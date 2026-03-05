# Seed-VC 开发指南

本文档为 AI 代理提供代码开发规范指导。

## 项目概述

Seed-VC 是零样本语音转换模型，支持零样本语音转换、实时语音转换、歌声转换及自定义数据微调。

**技术栈**: Python 3.10+, PyTorch, Gradio, Hydra
**音频采样率**: 22050Hz (v1/v2) 或 44100Hz (SVC)

## 环境配置

```bash
# 使用 uv 管理虚拟环境（推荐）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

- Python 3.10+
- GPU（实时转换推荐 NVIDIA RTX 3060+）
- 实时 GUI 需要 Tkinter 支持

## 常用命令

### 推理

```bash
# V1 模型
python inference.py --source <源音频> --target <参考音频> --output <输出目录>

# V2 模型
python inference_v2.py --source <源音频> --target <参考音频> --output <输出目录> \
    --cfm-checkpoint-path <cfm模型> --ar-checkpoint-path <ar模型>
```

### 训练

```bash
# V1 模型微调
python train.py --config <配置文件> --dataset-dir <数据集> --run-name <名称>

# V2 模型微调（支持多卡）
uv run accelerate launch train_v2.py --dataset-dir <数据集> --run-name <名称>
```

### Web UI

```bash
python app_vc.py --checkpoint <模型> --config <配置>
python app_vc_v2.py --cfm-checkpoint-path <cfm> --ar-checkpoint-path <ar>
python real-time-gui.py --checkpoint-path <模型> --config-path <配置>
```

### Lint / 测试

```bash
# 运行 ruff 检查
uv run ruff check .

# 自动修复可自动修复的问题
uv run ruff check . --fix

# 代码格式化
uv run ruff format .

# 运行所有测试（如有）
uv run pytest tests/ -v

# 运行单个测试
uv run pytest tests/test_file.py::test_function -v
```

**注意**: 当前项目暂无测试文件，pytest 命令仅作为预留。

### 类型检查（可选）

项目使用动态类型注解。如需严格类型检查，可使用 mypy：

```bash
uv run mypy . --ignore-missing-imports
```

## 代码规范

### 导入排序

遵循标准 Python 导入顺序，使用 ruff 自动管理：

```python
# 标准库
import os
import sys
from typing import Optional, Dict, List

# 第三方库
import torch
import numpy as np

# 本地模块
from modules.commons import str2bool
```

### 命名规范

- 类名：PascalCase（如 `AttrDict`, `Trainer`）
- 函数/变量：snake_case（如 `get_padding`, `max_steps`）
- 常量：全大写 snake_case（如 `MAX_SR`）

### 类型注解

为公共函数添加类型注解。

### 错误处理

捕获具体异常，避免空捕获。

### 格式化

- 最大行长度：120 字符
- 使用 ruff 格式化
- ruff 配置（pyproject.toml）：忽略 E402, F401, F841 等

### 代码风格

- 使用 `from torch import nn` 和 `from torch.nn import functional as F` 简化导入
- 使用 `munch.Munch` 或 `AttrDict` 处理配置字典
- 避免使用 `*` 导入（如 `from module import *`）
- 使用上下文管理器处理文件打开
- 优先使用列表推导式而非循环

### Docstrings

为公共函数和类添加 docstrings。

## 注意事项

- HuggingFace 访问：如遇网络问题，使用镜像 `HF_ENDPOINT=https://hf-mirror.com`
- 模型下载：首次运行自动下载，默认缓存 `./checkpoints/hf_cache`
- 实时转换延迟 ≈ Block Time × 2 + Extra context (right) + 设备延迟(~100ms)
- Mac 运行 `real-time-gui.py` 报错 `_tkinter` 缺失：需安装支持 Tkinter 的 Python

## 项目结构

```
seed-vc/
├── modules/          # 核心模型模块
├── configs/          # 配置文件
├── checkpoints/     # 模型权重
├── inference.py     # V1 推理
├── inference_v2.py  # V2 推理
├── train.py         # V1 训练
├── train_v2.py      # V2 训练
└── real-time-gui.py # 实时转换 GUI
```

## 预训练模型

| 版本 | 模型 | 用途 | 采样率 |
|------|------|------|--------|
| v1.0 | seed-uvit-tat-xlsr-tiny | 实时 VC | 22050 |
| v1.0 | seed-uvit-whisper-small-wavenet | 离线 VC | 22050 |
| v1.0 | seed-uvit-whisper-base | SVC | 44100 |
| v2.0 | hubert-bsqvae-small | VC/口音转换 | 22050 |

配置文件位于 `configs/presets/` 目录。
