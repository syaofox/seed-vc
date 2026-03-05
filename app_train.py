from seed_vc.config import *

import os
import signal

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gradio as gr
import yaml
import torch
import subprocess
import threading
import time
import shutil

RUNS_DIR = "runs/v1"
CONFIG_DIR = "configs/presets"


def get_run_list():
    if not os.path.exists(RUNS_DIR):
        return []
    runs = []
    for d in os.listdir(RUNS_DIR):
        full_path = os.path.join(RUNS_DIR, d)
        if os.path.isdir(full_path):
            has_model = os.path.exists(os.path.join(full_path, "ft_model.pth"))
            has_config = os.path.exists(
                os.path.join(
                    full_path, "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
                )
            )
            if has_model or has_config:
                runs.append({"name": d, "path": full_path})
    return runs


def get_config_list():
    if not os.path.exists(CONFIG_DIR):
        return []
    return [f for f in os.listdir(CONFIG_DIR) if f.endswith(".yml")]


def parse_checkpoint_info(run_path):
    ft_model = os.path.join(run_path, "ft_model.pth")
    if os.path.exists(ft_model):
        try:
            state = torch.load(ft_model, map_location="cpu")
            if "net" in state and "cfm" in state["net"]:
                return "cfm model"
            return "DiT model"
        except Exception:
            return "unknown"
    return "no model"


def get_run_info(run_name):
    run_path = os.path.join(RUNS_DIR, run_name)
    if not os.path.exists(run_path):
        return None

    config_files = [
        f
        for f in os.listdir(run_path)
        if f.startswith("config_") and f.endswith(".yml")
    ]
    config_file = config_files[0] if config_files else None

    model_exists = os.path.exists(os.path.join(run_path, "ft_model.pth"))

    return {"path": run_path, "config": config_file, "has_model": model_exists}


def prepare_config(
    config_path, run_name, batch_size, max_steps, max_epochs, save_interval, dataset_dir
):
    run_path = os.path.join(RUNS_DIR, run_name)
    existing_config = os.path.join(run_path, os.path.basename(config_path))

    if os.path.exists(existing_config):
        with open(existing_config, "r") as f:
            config = yaml.safe_load(f)
    else:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    config["log_dir"] = RUNS_DIR
    config["batch_size"] = batch_size
    config["max_steps"] = max_steps
    config["max_epochs"] = max_epochs
    config["save_interval"] = save_interval
    config["dataset_dir"] = dataset_dir

    os.makedirs(run_path, exist_ok=True)

    new_config_path = os.path.join(run_path, os.path.basename(config_path))
    with open(new_config_path, "w") as f:
        yaml.dump(config, f)

    return new_config_path


def load_run_params(run_name):
    if not run_name:
        return None, None, None, None, None, None

    run_path = os.path.join(RUNS_DIR, run_name)
    if not os.path.exists(run_path):
        return None, None, None, None, None, None

    config_files = [
        f
        for f in os.listdir(run_path)
        if f.startswith("config_") and f.endswith(".yml")
    ]
    if not config_files:
        return None, None, None, None, None, None

    config_path = os.path.join(run_path, config_files[0])
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        batch_size = config.get("batch_size", 2)
        max_steps = config.get("max_steps", 1000)
        max_epochs = config.get("max_epochs", 1000)
        save_interval = config.get("save_interval", 500)
        dataset_dir = config.get("dataset_dir", "")
        config_file = config_files[0]
        return (
            config_file,
            batch_size,
            max_steps,
            max_epochs,
            save_interval,
            dataset_dir,
        )
    except Exception:
        return None, None, None, None, None, None


class TrainingProcess:
    def __init__(self):
        self.process = None
        self.is_running = False
        self.log_lines = []
        self.lock = threading.Lock()

    def start(self, command):
        if self.is_running:
            return False

        self.log_lines = []
        self.process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        self.is_running = True

        def read_output():
            try:
                for line in self.process.stdout:
                    if not self.is_running:
                        break
                    with self.lock:
                        self.log_lines.append(line)
                    time.sleep(0.01)
            except Exception:
                pass

        thread = threading.Thread(target=read_output, daemon=True)
        thread.start()
        return True

    def stop(self):
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except Exception:
                pass
            self.process = None
        self.is_running = False
        return True

    def get_logs(self, last_n=100):
        with self.lock:
            if last_n:
                return "".join(self.log_lines[-last_n:])
            return "".join(self.log_lines)


trainer = TrainingProcess()


def start_training(
    run_name,
    config_file,
    dataset_dir,
    batch_size,
    max_steps,
    max_epochs,
    save_interval,
    gpu_id,
):
    if not run_name:
        yield "请输入 Run 名称", gr.update()
        return

    if not dataset_dir or not os.path.exists(dataset_dir):
        yield "请选择有效的 Dataset 目录", gr.update()
        return

    if not config_file:
        yield "请选择 Config 文件", gr.update()
        return

    run_path = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_path, exist_ok=True)

    pretrained_ckpt = ""
    run_info = get_run_info(run_name)
    if run_info and run_info["has_model"]:
        pretrained_ckpt = os.path.join(run_path, "ft_model.pth")
        config_file = run_info["config"] if run_info["config"] else config_file
        print(f"Continuing training from {pretrained_ckpt}")

    config_path = prepare_config(
        os.path.join(CONFIG_DIR, config_file),
        run_name,
        batch_size,
        max_steps,
        max_epochs,
        save_interval,
        dataset_dir,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd_parts = [
        "uv run python train.py",
        f'--config "{config_path}"',
        f'--dataset-dir "{dataset_dir}"',
        f'--run-name "{run_name}"',
        f"--batch-size {batch_size}",
        f"--max-steps {max_steps}",
        f"--max-epochs {max_epochs}",
        f"--save-every {save_interval}",
        f"--gpu {gpu_id}",
    ]
    if pretrained_ckpt:
        cmd_parts.append(f'--pretrained-ckpt "{pretrained_ckpt}"')

    cmd = " \\\n    ".join(cmd_parts)

    success = trainer.start(cmd)
    if not success:
        yield "训练已在运行中", gr.update()
        return

    yield "训练已启动", gr.update(value="⏹ 停止训练", variant="stop")

    last_update = 0
    while trainer.is_running:
        logs = trainer.get_logs(50)
        current_time = time.time()
        if current_time - last_update > 0.5:
            yield logs, gr.update()
            last_update = current_time
        time.sleep(0.1)

    final_logs = trainer.get_logs()
    yield (
        final_logs + "\n\n训练已结束",
        gr.update(value="🚀 开始训练", variant="primary"),
    )


def stop_training():
    trainer.stop()
    return "训练已停止", gr.update(value="🚀 开始训练", variant="primary")


def delete_run(run_name):
    if not run_name:
        return "请选择要删除的训练", gr.update(visible=True)

    run_path = os.path.join(RUNS_DIR, run_name)
    if os.path.exists(run_path):
        shutil.rmtree(run_path)
        return f"已删除训练: {run_name}", gr.update(visible=True)
    return f"训练不存在: {run_name}", gr.update(visible=True)


def refresh_run_list():
    runs = get_run_list()
    choices = [r["name"] for r in runs]
    return gr.update(choices=choices), gr.update(choices=choices)


def get_run_details(run_name):
    if not run_name:
        return "", "", False

    run_info = get_run_info(run_name)
    if not run_info:
        return "", "", False

    config = run_info["config"] or ""
    has_model = run_info["has_model"]
    model_type = parse_checkpoint_info(run_info["path"]) if has_model else "无模型"

    detail = f"Config: {config}\n模型: {model_type}"
    return detail, config, has_model


def build_ui():
    config_list = get_config_list()
    default_config = "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
    if default_config not in config_list:
        default_config = config_list[0] if config_list else ""

    run_list = [r["name"] for r in get_run_list()]

    with gr.Blocks(title="Seed-VC V1 训练") as demo:
        gr.Markdown("# Seed-VC V1 模型训练")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column(scale=3):
                        run_name_input = gr.Dropdown(
                            label="Run 名称",
                            choices=run_list,
                            allow_custom_value=True,
                            value=run_list[0] if run_list else "",
                            info="选择已有训练继续，或输入新名称全新训练",
                        )
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("🔄 刷新", size="sm")
                config_select = gr.Dropdown(
                    label="Config 文件", choices=config_list, value=default_config
                )
                dataset_dir = gr.Textbox(
                    label="Dataset 目录",
                    placeholder="/path/to/dataset",
                    info="包含音频文件的文件夹",
                )
            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column():
                        batch_size = gr.Number(label="Batch Size", value=2, precision=0)
                        max_steps = gr.Number(
                            label="Max Steps", value=1000, precision=0
                        )
                    with gr.Column():
                        max_epochs = gr.Number(
                            label="Max Epochs", value=1000, precision=0
                        )
                        save_interval = gr.Number(
                            label="Save Interval", value=500, precision=0
                        )
                with gr.Row():
                    gpu_id = gr.Number(label="GPU ID", value=0, precision=0)

        with gr.Row():
            start_btn = gr.Button("🚀 开始训练", variant="primary", size="lg")
            stop_btn = gr.Button("⏹ 停止训练", variant="stop", size="lg")

        run_details = gr.Textbox(label="训练详情", lines=2, interactive=False)

        training_log = gr.Textbox(label="训练日志", lines=20, interactive=False)

        with gr.Row():
            with gr.Column(scale=3):
                delete_run_name = gr.Dropdown(
                    label="已保存的训练",
                    choices=run_list,
                    allow_custom_value=True,
                )
            with gr.Column(scale=1):
                delete_btn = gr.Button("🗑 删除", variant="stop", size="sm")

        delete_msg = gr.Textbox(label="操作信息", lines=1, interactive=False)

        def on_run_change(run_name):
            details, cfg, has_model = get_run_details(run_name)
            cfg_file, bs, steps, epochs, save_int, ds_dir = load_run_params(run_name)
            if cfg_file:
                return details, cfg_file, bs, steps, epochs, save_int, ds_dir
            return details, cfg, 2, 1000, 1000, 500, ""

        run_name_input.change(
            on_run_change,
            inputs=[run_name_input],
            outputs=[
                run_details,
                config_select,
                batch_size,
                max_steps,
                max_epochs,
                save_interval,
                dataset_dir,
            ],
        )

        refresh_btn.click(refresh_run_list, outputs=[run_name_input, delete_run_name])

        def on_start(
            run_name,
            config_file,
            dataset_dir,
            batch_size,
            max_steps,
            max_epochs,
            save_interval,
            gpu_id,
        ):
            gen = start_training(
                run_name,
                config_file,
                dataset_dir,
                int(batch_size),
                int(max_steps),
                int(max_epochs),
                int(save_interval),
                int(gpu_id),
            )
            for log, btn_update in gen:
                yield log, btn_update

        start_btn.click(
            on_start,
            inputs=[
                run_name_input,
                config_select,
                dataset_dir,
                batch_size,
                max_steps,
                max_epochs,
                save_interval,
                gpu_id,
            ],
            outputs=[training_log, stop_btn],
        )

        stop_btn.click(stop_training, outputs=[training_log, start_btn])

        delete_btn.click(
            delete_run, inputs=[delete_run_name], outputs=[delete_msg, delete_run_name]
        )

        demo.load(refresh_run_list, outputs=[run_name_input, delete_run_name])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
