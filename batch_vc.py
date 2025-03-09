import argparse
import os

from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio

from tqdm import tqdm

from app_vc import load_models as app_load_models
from modules.commons import str2bool


def load_models(args, device):
    """从 app_vc 加载模型并返回必要的组件"""
    # 设置 app_vc.py 中的全局 device 变量
    import app_vc

    app_vc.device = device

    models = app_load_models(args)

    return {
        "inference_module": models[0],
        "semantic_fn": models[1],
        "vocoder_fn": models[2],
        "campplus_model": models[3],
        "to_mel": models[4],
        "sr": models[5]["sampling_rate"],
        "hop_length": models[5]["hop_size"],
    }


def process_audio(source_path, ref_path, output_path, model_dict, params):
    """处理单个音频文件的转换"""
    inference_module, semantic_fn, vocoder_fn, campplus_model, to_mel = (
        model_dict["inference_module"],
        model_dict["semantic_fn"],
        model_dict["vocoder_fn"],
        model_dict["campplus_model"],
        model_dict["to_mel"],
    )
    device = params["device"]
    sr = params["sr"]

    # 加载音频
    source_audio = librosa.load(source_path, sr=sr)[0]
    ref_audio = librosa.load(ref_path, sr=sr)[0]

    # 处理音频
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(device)

    # 重采样到16kHz
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)

    with torch.no_grad():
        # 提取语义特征
        S_alt = semantic_fn(converted_waves_16k)
        S_ori = semantic_fn(ref_waves_16k)

        # 生成mel谱
        mel = to_mel(source_audio.to(device).float())
        mel2 = to_mel(ref_audio.to(device).float())

        # 计算目标长度
        target_lengths = torch.LongTensor(
            [int(mel.size(2) * params["length_adjust"])]
        ).to(device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(device)

        # 提取参考音频特征
        feat2 = torchaudio.compliance.kaldi.fbank(
            ref_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = campplus_model(feat2.unsqueeze(0))

        # 长度调整
        cond, _, _, _, _ = inference_module.length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=None
        )
        prompt_condition, _, _, _, _ = inference_module.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
        )

        # 语音转换
        cat_condition = torch.cat([prompt_condition, cond], dim=1)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if params["fp16"] else torch.float32,
        ):
            vc_target = inference_module.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(device),
                mel2,
                style2,
                None,
                params["diffusion_steps"],
                inference_cfg_rate=params["inference_cfg_rate"],
            )
            vc_target = vc_target[:, :, mel2.size(-1) :]

        # 生成波形
        vc_wave = vocoder_fn(vc_target.float())[0][0].cpu().numpy()

        # 保存音频
        vc_wave = (vc_wave * 32768.0).astype(np.int16)
        torchaudio.save(output_path, torch.tensor(vc_wave).unsqueeze(0), sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list_file", type=str, required=True, help="Path to the input list file"
    )
    parser.add_argument(
        "--ref_audio", type=str, required=True, help="Path to the reference audio"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--character", type=str, required=True, help="Character name for output"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--fp16", type=str2bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--diffusion_steps", type=int, default=10)
    parser.add_argument("--length_adjust", type=float, default=1.0)
    parser.add_argument("--inference_cfg_rate", type=float, default=0.7)
    args = parser.parse_args()

    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # 加载模型
    model_dict = load_models(args, device)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理参数
    params = {
        "device": device,
        "sr": model_dict["sr"],
        "fp16": args.fp16,
        "diffusion_steps": args.diffusion_steps,
        "length_adjust": args.length_adjust,
        "inference_cfg_rate": args.inference_cfg_rate,
    }

    # 读取列表文件
    with open(args.list_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 创建新的列表文件
    output_list = []

    # 处理每个音频文件
    for line in tqdm(lines):
        source_path, role, lang, text = line.strip().split("|")

        # 使用文本内容作为文件名（只去掉空格，保留标点符号）
        safe_text = text.replace(" ", "").replace("\t", "").replace("\n", "")

        # 生成输出文件路径 (在角色名目录下，使用文本作为文件名)
        output_path = output_dir / args.character / f"{safe_text}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 处理音频转换
        process_audio(source_path, args.ref_audio, str(output_path), model_dict, params)

        # 添加到新列表
        output_list.append(f"{output_path}|{args.character}|{lang}|{text}")

    # 保存新的列表文件 (使用指定的角色名)
    list_filename = f"{args.character}.list"
    with open(output_dir / list_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(output_list))


if __name__ == "__main__":
    main()
