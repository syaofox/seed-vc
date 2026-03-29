# Docker 部署指南

## 前置要求

- Docker 20.10+
- Docker Compose v2.0+
- NVIDIA Container Toolkit（GPU 推理/训练）

```bash
# 安装 NVIDIA Container Toolkit (Ubuntu/Debian)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 构建镜像

```bash
docker compose build
```

## 使用方式

通过 `SERVICE` 环境变量选择启动的服务：

```bash
SERVICE=<服务名> docker compose up
```

## 支持的服务

| SERVICE 值 | 说明 | 默认端口 |
|------------|------|----------|
| `vc-v1` | V1 语音转换 WebUI | 7860 |
| `vc-v2` | V2 语音转换 WebUI | 7860 |
| `svc` | 歌声转换 WebUI | 7860 |
| `train-v1` | V1 训练 WebUI | 7860 |
| `train-v2` | V2 训练 WebUI | 7861 |
| `inference` | V1 命令行推理 | - |
| `inference-v2` | V2 命令行推理 | - |
| `train` | V1 命令行训练 | - |
| `train-v2-cmd` | V2 命令行训练 | - |

## 示例

### 启动 WebUI

```bash
# V1 语音转换
SERVICE=vc-v1 docker compose up

# V2 语音转换
SERVICE=vc-v2 docker compose up

# 歌声转换
SERVICE=svc docker compose up

# 后台运行
SERVICE=vc-v1 docker compose up -d
```

### 修改端口

```bash
# 使用 8080 端口
SERVICE=vc-v1 PORT=8080 docker compose up
```

### 命令行推理

```bash
# V1 推理
docker compose run --rm \
  -e SERVICE=inference \
  -v ./data:/app/data \
  seed-vc --source /app/data/source.wav --target /app/data/reference.wav --output /app/outputs

# V2 推理
docker compose run --rm \
  -e SERVICE=inference-v2 \
  -v ./data:/app/data \
  seed-vc --source /app/data/source.wav --target /app/data/reference.wav --output /app/outputs
```

### 命令行训练

```bash
docker compose run --rm \
  -e SERVICE=train \
  -v ./data:/app/data \
  seed-vc --dataset-dir /app/data/dataset --run-name my_model
```

## 目录挂载

| 容器路径 | 主机路径 | 说明 |
|----------|----------|------|
| `/app/checkpoints` | `./checkpoints` | 模型权重（只读） |
| `/app/runs` | `./runs` | 微调模型（读写） |
| `/app/data` | `./data` | 数据集（只读） |
| `/app/outputs` | `./outputs` | 输出目录（读写） |

## 常用命令

```bash
# 查看日志
docker compose logs -f

# 停止服务
docker compose down

# 进入容器
docker compose exec seed-vc bash

# 重新构建（不使用缓存）
docker compose build --no-cache

# 查看运行状态
docker compose ps
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SERVICE` | `vc-v1` | 启动的服务名称 |
| `PORT` | `7860` | WebUI 端口 |
| `HF_HUB_OFFLINE` | `1` | 离线模式 |
| `HF_HUB_CACHE` | `/app/checkpoints/hf_cache` | HuggingFace 缓存目录 |

## 注意事项

1. **模型文件**：需提前下载到 `checkpoints/hf_cache` 目录，或从宿主机复制
2. **GPU 显存**：推荐 6GB+ 显存，V2 模型需要更多
3. **文件权限**：容器内以 uid=1000 用户运行，输出文件权限自动正确
4. **首次构建**：依赖安装需要较长时间，后续构建使用缓存

## 故障排除

### GPU 不可用

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 Docker 配置
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### 端口被占用

```bash
# 修改端口
SERVICE=vc-v1 PORT=8080 docker compose up
```

### 模型文件缺失

确保 `checkpoints/hf_cache` 目录存在并包含模型文件，或检查网络连接后删除 `HF_HUB_OFFLINE=1` 环境变量。
