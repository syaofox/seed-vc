# Seed-VC 开发指南

本文件为 AI 代理提供代码开发规范指导。

## 项目概述

Seed-VC 是一个零样本语音转换模型，支持：
- 零样本语音转换 (VC)
- 零样本实时语音转换
- 零样本歌声转换 (SVC)
- 自定义数据微调

**技术栈**: Python 3.10+, PyTorch, Gradio, Hydra

## 环境配置

### 安装依赖

```bash
# 推荐使用 uv
uv sync

# 或使用 pip
pip install -r requirements.txt

# Mac M 系列
pip install -r requirements-mac.txt
```

### 运行环境要求

- Python 3.10+
- GPU (实时转换推荐 NVIDIA RTX 3060+)
- 实时 GUI 需要 Tkinter 支持

## 常用命令

### 推理命令

```bash
# V1 模型推理
python inference.py --source <源音频> --target <参考音频> --output <输出目录> \
    --diffusion-steps 25 --fp16 True

# V2 模型推理
python inference_v2.py --source <源音频> --target <参考音频> --output <输出目录> \
    --diffusion-steps 25 --cfm-checkpoint-path <cfm模型路径> --ar-checkpoint-path <ar模型路径>
```

### 训练命令

```bash
# V1 模型微调
python train.py --config <配置文件> --dataset-dir <数据集路径> \
    --run-name <运行名称> --batch-size 2 --max-steps 1000 --max-epochs 1000 \
    --save-every 500 --num-workers 0

# V2 模型微调 (支持多卡)
accelerate launch train_v2.py --dataset-dir <数据集路径> \
    --run-name <运行名称> --batch-size 2 --max-steps 1000 --max-epochs 1000 \
    --save-every 500 --num-workers 0 --train-cfm --train-ar
```

### Web UI 启动

```bash
# 语音转换 Web UI (V1)
python app_vc.py --checkpoint <模型路径> --config <配置路径> --fp16 True

# 歌声转换 Web UI (V1)
python app_svc.py --checkpoint <模型路径> --config <配置路径> --fp16 True

# V2 模型 Web UI
python app_vc_v2.py --cfm-checkpoint-path <cfm路径> --ar-checkpoint-path <ar路径>

# 集成 Web UI
python app.py --enable-v1 --enable-v2

# 实时转换 GUI
python real-time-gui.py --checkpoint-path <模型路径> --config-path <配置路径>
```

### Lint 检查

```bash
# 运行 ruff 检查
ruff check .

# 或使用 uv
uv run ruff check .
```

### 运行单个测试

当前项目暂无测试套件，无需运行测试。

## 代码规范

### 导入排序

遵循标准 Python 导入顺序：
1. 标准库
2. 第三方库
3. 本地模块

```python
# 正确示例
import os
import sys
import json
from typing import Optional, Tuple

import torch
import numpy as np
import librosa
import yaml

from modules.commons import str2bool
from optimizers import build_optimizer
```

### 命名规范

- 类名：PascalCase (如 `AttrDict`, `Trainer`)
- 函数/变量：snake_case (如 `get_padding`, `max_steps`)
- 常量：全大写 snake_case (如 `MAX_SR`)

### 类型注解

推荐为公共函数添加类型注解：
- 函数参数
- 返回值

```python
# 推荐
def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)

# 可选：复杂类型
from typing import Optional, Dict, List
def build_model(config: Dict, stage: str) -> Optional[nn.Module]:
    ...
```

### 函数文档

当前代码库约定：不需要为函数添加 docstring。

### 错误处理

使用 try/except 捕获具体异常：

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

### 配置文件

- 使用 YAML 格式
- 配置文件位于 `configs/` 目录
- 使用 Hydra 管理配置时遵循其约定

### 特定模块说明

#### modules/ 目录

核心模型模块：
- `commons.py` - 通用工具函数
- `diffusion_transformer.py` - DiT 模型
- `rmvpe.py` - F0 提取
- `avenet.py` / `bigwvgan/` - 声码器

#### v2/ 模块

V2 模型专用：
- 需要单独下载模型权重
- 使用 `--compile` 标志可加速推理约 6 倍

## 注意事项

### HuggingFace 访问

如遇网络问题，使用镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 模型下载

首次运行时会自动从 HuggingFace 下载模型：
- 默认缓存目录：`./checkpoints/hf_cache`

### 实时转换延迟计算

```
总延迟 ≈ Block Time × 2 + Extra context (right) + 设备延迟(~100ms)
```

### 已知问题

- Mac 运行 `real-time-gui.py` 报错 `ModuleNotFoundError: No module named '_tkinter'`：需安装支持 Tkinter 的 Python 版本

## 预训练模型

| 版本 | 模型 | 用途 | 采样率 |
|------|------|------|--------|
| v1.0 | seed-uvit-tat-xlsr-tiny | 实时 VC | 22050 |
| v1.0 | seed-uvit-whisper-small-wavenet | 离线 VC | 22050 |
| v1.0 | seed-uvit-whisper-base | SVC | 44100 |
| v2.0 | hubert-bsqvae-small | VC/口音转换 | 22050 |

配置文件位于 `configs/presets/` 目录。
