#!/usr/bin/env python3
"""
Seed-VC 模型下载脚本

用途:
    1. 下载 V1/V2 推理和训练所需的全部模型到本地缓存
    2. 检验已缓存的模型是否完整
    3. 支持离线模式运行

用法:
    # 检查已缓存模型状态
    uv run python download_models.py --check-only

    # 下载所有模型（需要网络）
    uv run python download_models.py

    # 下载所有模型，包含可选模型（44kHz, HiFT 等）
    uv run python download_models.py --optional

    # 仅下载 V1 模型
    uv run python download_models.py --v1-only

    # 仅下载 V2 模型
    uv run python download_models.py --v2-only

    # 使用镜像下载（如访问 HuggingFace 困难）
    HF_ENDPOINT=https://hf-mirror.com uv run python download_models.py
"""

import os
import sys
import argparse
from pathlib import Path

# HF_CACHE = Path("./checkpoints/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "./checkpoints/hf_cache")
HF_CACHE = Path(os.environ.get("HF_HUB_CACHE", "./checkpoints/hf_cache"))
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_ENDPOINT", None)
HF_CACHE.mkdir(parents=True, exist_ok=True)

MODELS = {
    "V1 推理/训练模型": [
        {
            "repo_id": "Plachta/Seed-VC",
            "filename": "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
            "description": "V1 DiT 主模型",
        },
        {
            "repo_id": "Plachta/Seed-VC",
            "filename": "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
            "description": "V1 配置文件",
        },
        {
            "repo_id": "funasr/campplus",
            "filename": "campplus_cn_common.bin",
            "description": "CAMPPlus 说话人编码器",
        },
        {
            "repo_id": "openai/whisper-small",
            "filename": "config.json",
            "description": "Whisper-small 配置文件",
        },
        {
            "repo_id": "openai/whisper-small",
            "filename": "model.safetensors",
            "description": "Whisper-small 模型权重",
        },
        {
            "repo_id": "openai/whisper-small",
            "filename": "tokenizer.json",
            "description": "Whisper-small 分词器",
        },
        {
            "repo_id": "openai/whisper-small",
            "filename": "preprocessor_config.json",
            "description": "Whisper-small 预处理器配置",
        },
        {
            "repo_id": "nvidia/bigvgan_v2_22khz_80band_256x",
            "filename": "config.json",
            "description": "BigVGAN 22kHz 配置文件",
        },
        {
            "repo_id": "nvidia/bigvgan_v2_22khz_80band_256x",
            "filename": "bigvgan_generator.pt",
            "description": "BigVGAN 22kHz 生成器权重",
        },
        {
            "repo_id": "Plachta/Seed-VC",
            "filename": "se_db.pt",
            "description": "说话人嵌入数据库",
        },
        {
            "repo_id": "myshell-ai/OpenVoiceV2",
            "filename": "converter/checkpoint.pth",
            "description": "OpenVoice V2 转换器权重",
        },
        {
            "repo_id": "myshell-ai/OpenVoiceV2",
            "filename": "converter/config.json",
            "description": "OpenVoice V2 转换器配置",
        },
        {
            "repo_id": "lj1995/VoiceConversionWebUI",
            "filename": "rmvpe.pt",
            "description": "RMVPE F0 预测模型",
        },
    ],
    "V2 推理/训练模型": [
        {
            "repo_id": "openai/whisper-small",
            "filename": "config.json",
            "description": "Whisper-small (V2 content tokenizer)",
        },
        {
            "repo_id": "openai/whisper-small",
            "filename": "model.safetensors",
            "description": "Whisper-small (V2 content tokenizer)",
        },
        {
            "repo_id": "openai/whisper-small",
            "filename": "tokenizer.json",
            "description": "Whisper-small (V2 content tokenizer)",
        },
        {
            "repo_id": "openai/whisper-small",
            "filename": "preprocessor_config.json",
            "description": "Whisper-small (V2 content tokenizer)",
        },
        {
            "repo_id": "facebook/hubert-large-ll60k",
            "filename": "config.json",
            "description": "HuBERT-large SSL 配置文件",
        },
        {
            "repo_id": "facebook/hubert-large-ll60k",
            "filename": "pytorch_model.bin",
            "description": "HuBERT-large SSL 权重",
        },
        {
            "repo_id": "facebook/hubert-large-ll60k",
            "filename": "preprocessor_config.json",
            "description": "HuBERT-large 预处理器配置",
        },
    ],
    "V1 其他可选模型": [
        {
            "repo_id": "Plachta/Seed-VC",
            "filename": "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
            "description": "V1 44kHz SVC 模型(可选)",
        },
        {
            "repo_id": "Plachta/Seed-VC",
            "filename": "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
            "description": "V1 44kHz 配置文件(可选)",
        },
        {
            "repo_id": "nvidia/bigvgan_v2_44khz_128band_512x",
            "filename": "config.json",
            "description": "BigVGAN 44kHz 配置文件(可选)",
        },
        {
            "repo_id": "nvidia/bigvgan_v2_44khz_128band_512x",
            "filename": "bigvgan_generator.pt",
            "description": "BigVGAN 44kHz 生成器权重(可选)",
        },
        {
            "repo_id": "FunAudioLLM/CosyVoice-300M",
            "filename": "hift.pt",
            "description": "HiFT vocoder (可选，用于 hifigan 类型)",
        },
    ],
}


def check_model_file(repo_id: str, filename: str) -> bool:
    """检查并显示模型缓存状态"""
    from huggingface_hub import try_to_load_from_cache

    try:
        cached_path = try_to_load_from_cache(
            repo_id=repo_id, filename=filename, cache_dir=str(HF_CACHE)
        )
    except Exception:
        return False

    if (
        cached_path is not None
        and isinstance(cached_path, (str, Path))
        and os.path.exists(cached_path)
    ):
        return True
    else:
        return False


def download_model(repo_id: str, filename: str, description: str) -> bool:
    """下载单个模型"""
    from huggingface_hub import hf_hub_download

    print(f"  下载 {description}...")
    print(f"    Repo: {repo_id}")
    print(f"    File: {filename}")

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(HF_CACHE),
            local_files_only=False,
        )
        print(f"    成功: {path}")
        return True
    except Exception as e:
        print(f"    失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载 Seed-VC 模型到本地缓存")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅检查已缓存模型，不下载",
    )
    parser.add_argument(
        "--v1-only",
        action="store_true",
        help="仅下载 V1 模型",
    )
    parser.add_argument(
        "--v2-only",
        action="store_true",
        help="仅下载 V2 模型",
    )
    parser.add_argument(
        "--optional",
        action="store_true",
        help="包含可选模型",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Seed-VC 模型下载工具")
    print("=" * 60)
    print(f"缓存目录: {HF_CACHE.absolute()}")
    print()

    model_groups = []
    if args.v1_only:
        model_groups.append(("V1 推理/训练模型", MODELS["V1 推理/训练模型"]))
        if args.optional:
            model_groups.append(("V1 其他可选模型", MODELS["V1 其他可选模型"]))
    elif args.v2_only:
        model_groups.append(("V2 推理/训练模型", MODELS["V2 推理/训练模型"]))
    else:
        for key in MODELS:
            model_groups.append((key, MODELS[key]))
        if not args.optional:
            model_groups.pop()

    total = 0
    cached = 0
    missing = 0

    for group_name, models in model_groups:
        print(f"\n{'=' * 60}")
        print(f"【{group_name}】")
        print("=" * 60)

        for model in models:
            repo_id = model["repo_id"]
            filename = model["filename"]
            description = model["description"]

            is_cached = check_model_file(repo_id, filename)
            total += 1

            if is_cached:
                cached += 1
                status = "✓ 已缓存"
            else:
                missing += 1
                status = "✗ 未缓存"

            print(f"  [{status}] {description}")
            print(f"    {repo_id}/{filename}")

            if not args.check_only and not is_cached:
                success = download_model(repo_id, filename, description)
                if success:
                    cached += 1
                    missing -= 1

    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    print(f"  总模型数: {total}")
    print(f"  已缓存:   {cached}")
    print(f"  缺失:     {missing}")

    if missing > 0:
        print(f"\n⚠️  有 {missing} 个模型未缓存，需要网络下载")
        print("   请运行: python download_models.py (需要网络)")
        return 1
    else:
        print("\n✅ 所有模型已缓存，可以离线运行！")
        return 0


if __name__ == "__main__":
    sys.exit(main())
