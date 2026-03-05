from seed_vc.config import *

import gradio as gr
import torch
import yaml
import os
import glob

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float16

current_model_name = None
vc_wrapper_global = None


def scan_fine_tuned_models():
    """扫描 runs/v2 下的微调模型目录"""
    models = [{"name": "默认模型", "path": None}]
    v2_dir = "runs/v2"
    if os.path.exists(v2_dir):
        for item in os.listdir(v2_dir):
            model_path = os.path.join(v2_dir, item)
            if os.path.isdir(model_path):
                ar_files = glob.glob(os.path.join(model_path, "AR_*.pth"))
                cfm_files = glob.glob(os.path.join(model_path, "CFM_*.pth"))
                if ar_files and cfm_files:
                    models.append({"name": item, "path": model_path})
    return models


def find_first_audio(model_dir):
    """查找模型目录下的第一个音频文件"""
    if not model_dir:
        return None
    audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"]
    for ext in audio_extensions:
        files = glob.glob(os.path.join(model_dir, ext))
        if files:
            return files[0]
    return None


def load_models(ar_checkpoint_path=None, cfm_checkpoint_path=None, compile=False):
    from hydra.utils import instantiate
    from omegaconf import DictConfig

    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc_wrapper = instantiate(cfg)
    vc_wrapper.load_checkpoints(ar_checkpoint_path=ar_checkpoint_path, cfm_checkpoint_path=cfm_checkpoint_path)
    vc_wrapper.to(device)
    vc_wrapper.eval()

    vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)

    if compile:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True

        if hasattr(torch._inductor.config, "fx_graph_cache"):
            torch._inductor.config.fx_graph_cache = True
        vc_wrapper.compile_ar()

    return vc_wrapper


def reload_model(model_name, compile_flag):
    global current_model_name, vc_wrapper_global

    models = scan_fine_tuned_models()
    selected = next((m for m in models if m["name"] == model_name), models[0])

    if selected["path"] is None:
        ar_path = None
        cfm_path = None
        status = "正在加载默认模型..."
    else:
        ar_files = glob.glob(os.path.join(selected["path"], "AR_*.pth"))
        cfm_files = glob.glob(os.path.join(selected["path"], "CFM_*.pth"))
        ar_path = ar_files[0] if ar_files else None
        cfm_path = cfm_files[0] if cfm_files else None
        status = f"正在加载微调模型: {model_name}..."

    vc_wrapper_global = load_models(ar_checkpoint_path=ar_path, cfm_checkpoint_path=cfm_path, compile=compile_flag)
    current_model_name = model_name

    ref_audio = find_first_audio(selected["path"])
    if ref_audio:
        status += f"\n已自动加载参考音频: {os.path.basename(ref_audio)}"

    return status, ref_audio


def convert_voice(
    source_audio,
    reference_audio,
    diffusion_steps,
    length_adjust,
    intelligibility_cfg_rate,
    similarity_cfg_rate,
    top_p,
    temperature,
    repetition_penalty,
    convert_style,
    anonymization_only,
):
    global vc_wrapper_global

    if source_audio is None or reference_audio is None:
        yield None, None
        return

    if vc_wrapper_global is None:
        yield None, "请先加载模型"
        return

    try:
        if hasattr(vc_wrapper_global, "convert_voice_with_streaming"):
            yield from vc_wrapper_global.convert_voice_with_streaming(
                source_audio_path=source_audio,
                target_audio_path=reference_audio,
                diffusion_steps=diffusion_steps,
                length_adjust=length_adjust,
                intelligebility_cfg_rate=intelligibility_cfg_rate,
                similarity_cfg_rate=similarity_cfg_rate,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                convert_style=convert_style,
                anonymization_only=anonymization_only,
                device=device,
                dtype=dtype,
                stream_output=True,
            )
        else:
            yield vc_wrapper_global.convert(
                source_audio=source_audio,
                reference_audio=reference_audio,
                diffusion_steps=diffusion_steps,
                length_adjust=length_adjust,
                intelligibility_cfg_rate=intelligibility_cfg_rate,
                similarity_cfg_rate=similarity_cfg_rate,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                convert_style=convert_style,
                anonymization_only=anonymization_only,
            )
    except Exception as e:
        import traceback

        error_msg = f"错误: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        yield None, error_msg


def main(args):
    global vc_wrapper_global, current_model_name
    models = scan_fine_tuned_models()
    model_choices = [m["name"] for m in models]
    default_model = model_choices[0] if model_choices else "默认模型"

    vc_wrapper_global = load_models(
        ar_checkpoint_path=args.ar_checkpoint_path,
        cfm_checkpoint_path=args.cfm_checkpoint_path,
        compile=args.compile,
    )
    current_model_name = default_model

    description = (
        "无需训练的 zero-shot 语音/歌声转换模型，若需本地部署查看[GitHub页面](https://github.com/Plachtaa/seed-vc)<br>"
        "请注意，参考音频若超过 25 秒，则会被自动裁剪至关长度。<br>若源音频和参考音频的总时长超过 30 秒，源音频将被分段处理。"
    )

    with gr.Blocks(title="Seed Voice Conversion V2") as demo:
        gr.Markdown(f"# Seed Voice Conversion V2\n{description}")

        gr.Markdown("---")
        gr.Markdown("### 模型选择 / Model Selection")
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=model_choices,
                value=default_model,
                label="选择模型 / Select Model",
                scale=4,
            )
            reload_btn = gr.Button("切换模型 / Reload Model", scale=1)
        model_status = gr.Textbox(value="默认模型已加载", label="状态 / Status", interactive=False, lines=2)

        gr.Markdown("---")
        gr.Markdown("### 语音转换 / Voice Conversion")

        inputs = [
            gr.Audio(type="filepath", label="Source Audio / 源音频"),
            gr.Audio(type="filepath", label="Reference Audio / 参考音频"),
            gr.Slider(
                minimum=1,
                maximum=200,
                value=30,
                step=1,
                label="Diffusion Steps / 扩散步数",
                info="30 by default, 50~100 for best quality / 默认为 30，50~100 为最佳质量",
            ),
            gr.Slider(
                minimum=0.5,
                maximum=2.0,
                step=0.1,
                value=1.0,
                label="Length Adjust / 长度调整",
                info="<1.0 for speed-up speech, >1.0 for slow-down speech / <1.0 加速语速，>1.0 减慢语速",
            ),
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=0.5,
                label="Intelligibility CFG Rate",
                info="has subtle influence / 有微小影响",
            ),
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=0.5,
                label="Similarity CFG Rate",
                info="has subtle influence / 有微小影响",
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                step=0.1,
                value=0.9,
                label="Top-p",
                info="Controls diversity of generated audio / 控制生成音频的多样性",
            ),
            gr.Slider(
                minimum=0.1,
                maximum=2.0,
                step=0.1,
                value=1.0,
                label="Temperature",
                info="Controls randomness of generated audio / 控制生成音频的随机性",
            ),
            gr.Slider(
                minimum=1.0,
                maximum=3.0,
                step=0.1,
                value=1.0,
                label="Repetition Penalty",
                info="Penalizes repetition in generated audio / 惩罚生成音频中的重复",
            ),
            gr.Checkbox(label="convert style", value=False),
            gr.Checkbox(label="anonymization only", value=False),
        ]

        outputs = [
            gr.Audio(label="Stream Output Audio / 流式输出", streaming=True, format="mp3"),
            gr.Audio(label="Full Output Audio / 完整输出", streaming=False, format="wav"),
        ]

        examples = [
            [
                "examples/source/yae_0.wav",
                "examples/reference/dingzhen_0.wav",
                50,
                1.0,
                0.5,
                0.5,
                0.9,
                1.0,
                1.0,
                False,
                False,
            ],
            [
                "examples/source/jay_0.wav",
                "examples/reference/azuma_0.wav",
                50,
                1.0,
                0.5,
                0.5,
                0.9,
                1.0,
                1.0,
                False,
                False,
            ],
        ]

        gr.Interface(
            fn=convert_voice,
            inputs=inputs,
            outputs=outputs,
            examples=examples,
            cache_examples=False,
            allow_flagging="never",
        )

        reload_btn.click(
            fn=reload_model,
            inputs=[model_dropdown, gr.State(args.compile)],
            outputs=[model_status, inputs[1]],
            queue=True,
        )

    demo.launch()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Compile the model using torch.compile")
    parser.add_argument(
        "--ar-checkpoint-path",
        type=str,
        default=None,
        help="Path to custom checkpoint file",
    )
    parser.add_argument(
        "--cfm-checkpoint-path",
        type=str,
        default=None,
        help="Path to custom checkpoint file",
    )
    args = parser.parse_args()
    main(args)
