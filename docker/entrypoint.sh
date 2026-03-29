#!/bin/bash
set -e

SERVICE=${SERVICE:-vc-v1}
PORT=${PORT:-7860}

# 检查模型目录
if [ ! -d "/app/checkpoints" ]; then
    echo "警告: /app/checkpoints 未挂载，模型文件可能缺失"
fi

# 设置离线模式
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/app/checkpoints/hf_cache}

# 设置 Gradio 配置
export GRADIO_SERVER_PORT=${PORT}
export GRADIO_SERVER_NAME=0.0.0.0

# 根据服务类型启动
case "$SERVICE" in
    vc-v1)
        echo "启动 V1 语音转换 WebUI (端口: $PORT)"
        exec python app_vc.py
        ;;
    vc-v2)
        echo "启动 V2 语音转换 WebUI (端口: $PORT)"
        exec python app_vc_v2.py
        ;;
    svc)
        echo "启动歌声转换 WebUI (端口: $PORT)"
        exec python app_svc.py
        ;;
    train-v1)
        echo "启动 V1 训练 WebUI (端口: $PORT)"
        exec python app_train.py
        ;;
    train-v2)
        echo "启动 V2 训练 WebUI (端口: $PORT)"
        exec python app_train_v2.py
        ;;
    inference)
        echo "运行命令行推理"
        exec python inference.py "$@"
        ;;
    inference-v2)
        echo "运行 V2 命令行推理"
        exec python inference_v2.py "$@"
        ;;
    train)
        echo "运行命令行训练"
        exec python train.py "$@"
        ;;
    train-v2-cmd)
        echo "运行 V2 命令行训练"
        exec python train_v2.py "$@"
        ;;
    *)
        echo "未知服务: $SERVICE"
        echo "支持的服务: vc-v1, vc-v2, svc, train-v1, train-v2, inference, inference-v2, train, train-v2-cmd"
        exit 1
        ;;
esac
