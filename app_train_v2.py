import os
import signal

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gradio as gr
import yaml
import subprocess
import threading
import time
import shutil

RUNS_DIR = "runs/v2"
CONFIG_DIR = "configs"


def get_run_list():
    if not os.path.exists(RUNS_DIR):
        return []
    runs = []
    for d in os.listdir(RUNS_DIR):
        full_path = os.path.join(RUNS_DIR, d)
        if os.path.isdir(full_path):
            has_cfm = os.path.exists(os.path.join(full_path, "CFM_epoch_"))
            has_ar = os.path.exists(os.path.join(full_path, "AR_epoch_"))
            has_config = os.path.exists(os.path.join(full_path, "vc_wrapper.yaml"))
            if has_cfm or has_ar or has_config:
                runs.append({"name": d, "path": full_path})
    return runs


def get_config_list():
    if not os.path.exists(CONFIG_DIR):
        return []
    configs = []
    v2_dir = os.path.join(CONFIG_DIR, "v2")
    if os.path.exists(v2_dir):
        configs = [f for f in os.listdir(v2_dir) if f.endswith((".yml", ".yaml"))]
    return configs


def parse_checkpoint_info(run_path):
    cfm_models = [f for f in os.listdir(run_path) if f.startswith("CFM_epoch_")]
    ar_models = [f for f in os.listdir(run_path) if f.startswith("AR_epoch_")]

    parts = []
    if cfm_models:
        parts.append("CFM")
    if ar_models:
        parts.append("AR")

    if parts:
        return " + ".join(parts)
    return "无模型"


def get_run_info(run_name):
    run_path = os.path.join(RUNS_DIR, run_name)
    if not os.path.exists(run_path):
        return None

    config_file = (
        "v2/vc_wrapper.yaml"
        if os.path.exists(os.path.join(run_path, "vc_wrapper.yaml"))
        else None
    )

    cfm_models = [f for f in os.listdir(run_path) if f.startswith("CFM_epoch_")]
    ar_models = [f for f in os.listdir(run_path) if f.startswith("AR_epoch_")]

    return {
        "path": run_path,
        "config": config_file,
        "has_cfm": len(cfm_models) > 0,
        "has_ar": len(ar_models) > 0,
        "cfm_ckpt": cfm_models[-1] if cfm_models else "",
        "ar_ckpt": ar_models[-1] if ar_models else "",
    }


METADATA_FILE = "metadata.yaml"


def prepare_config(
    config_path,
    run_name,
    batch_size,
    max_steps,
    max_epochs,
    save_interval,
    dataset_dir,
    train_cfm=True,
    train_ar=False,
):
    run_path = os.path.join(RUNS_DIR, run_name)
    basename = os.path.basename(config_path)

    os.makedirs(run_path, exist_ok=True)

    existing_config = os.path.join(run_path, basename)
    if not os.path.exists(existing_config):
        shutil.copy(config_path, existing_config)

    new_config_path = os.path.join(run_path, basename)

    metadata_path = os.path.join(run_path, METADATA_FILE)
    metadata = {
        "config_file": basename,
        "dataset_dir": dataset_dir,
        "batch_size": batch_size,
        "max_steps": max_steps,
        "max_epochs": max_epochs,
        "save_interval": save_interval,
        "train_cfm": train_cfm,
        "train_ar": train_ar,
    }
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f)

    return new_config_path


def load_run_params(run_name):
    if not run_name:
        return None, None, None, None, None, None, None, None

    run_path = os.path.join(RUNS_DIR, run_name)
    if not os.path.exists(run_path):
        return None, None, None, None, None, None, None, None

    dataset_dir = ""
    batch_size = 2
    max_steps = 1000
    max_epochs = 1000
    save_interval = 500
    train_cfm = True
    train_ar = False
    config_file = ""

    metadata_path = os.path.join(run_path, METADATA_FILE)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f)
                config_file = metadata.get("config_file", "")
                dataset_dir = metadata.get("dataset_dir", "")
                batch_size = metadata.get("batch_size", 2)
                max_steps = metadata.get("max_steps", 1000)
                max_epochs = metadata.get("max_epochs", 1000)
                save_interval = metadata.get("save_interval", 500)
                train_cfm = metadata.get("train_cfm", True)
                train_ar = metadata.get("train_ar", False)
        except Exception:
            pass

    if not config_file:
        config_files = [
            f
            for f in os.listdir(run_path)
            if f.endswith((".yml", ".yaml")) and f != METADATA_FILE
        ]
        if config_files:
            config_file = config_files[0]

    return (
        config_file,
        batch_size,
        max_steps,
        max_epochs,
        save_interval,
        dataset_dir,
        train_cfm,
        train_ar,
    )


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
    train_cfm,
    train_ar,
    num_workers,
    pretrained_cfm_ckpt,
    pretrained_ar_ckpt,
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

    if not train_cfm and not train_ar:
        yield "请至少选择一种训练目标 (CFM 或 AR)", gr.update()
        return

    run_path = os.path.join(RUNS_DIR, run_name)

    run_info = get_run_info(run_name)

    cfm_ckpt = pretrained_cfm_ckpt if pretrained_cfm_ckpt else ""
    ar_ckpt = pretrained_ar_ckpt if pretrained_ar_ckpt else ""

    if run_info:
        if run_info["has_cfm"]:
            if cfm_ckpt and not os.path.isabs(cfm_ckpt):
                cfm_ckpt = os.path.join(run_path, cfm_ckpt)
            elif not cfm_ckpt:
                cfm_ckpt = os.path.join(run_path, run_info["cfm_ckpt"])
        if run_info["has_ar"]:
            if ar_ckpt and not os.path.isabs(ar_ckpt):
                ar_ckpt = os.path.join(run_path, ar_ckpt)
            elif not ar_ckpt:
                ar_ckpt = os.path.join(run_path, run_info["ar_ckpt"])
        if run_info["config"]:
            config_file = run_info["config"]
        print(f"Continuing training - CFM: {cfm_ckpt}, AR: {ar_ckpt}")

    if config_file and not config_file.startswith("v2/"):
        config_file = "v2/" + config_file

    config_path = os.path.join(CONFIG_DIR, config_file) if config_file else ""

    new_config_path = prepare_config(
        config_path,
        run_name,
        batch_size,
        max_steps,
        max_epochs,
        save_interval,
        dataset_dir,
        train_cfm,
        train_ar,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_dir = os.path.join(RUNS_DIR, run_name)
    cmd_parts = [
        "uv run accelerate launch train_v2.py",
        f'--config "{new_config_path}"',
        f'--dataset-dir "{dataset_dir}"',
        f'--run-name "{run_name}"',
        f'--log-dir "{log_dir}"',
        f"--batch-size {batch_size}",
        f"--max-steps {max_steps}",
        f"--max-epochs {max_epochs}",
        f"--save-every {save_interval}",
        f"--num-workers {num_workers}",
    ]

    if train_cfm:
        cmd_parts.append("--train-cfm")
    if train_ar:
        cmd_parts.append("--train-ar")

    if cfm_ckpt:
        cmd_parts.append(f'--pretrained-cfm-ckpt "{cfm_ckpt}"')
    if ar_ckpt:
        cmd_parts.append(f'--pretrained-ar-ckpt "{ar_ckpt}"')

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
    first = choices[0] if choices else ""
    return gr.update(choices=choices, value=first), gr.update(choices=choices)


def get_run_details(run_name):
    if not run_name:
        return "", "", False, False, "", ""

    run_info = get_run_info(run_name)
    if not run_info:
        return "", "", False, False, "", ""

    config = run_info["config"] or ""
    has_cfm = run_info["has_cfm"]
    has_ar = run_info["has_ar"]
    model_type = parse_checkpoint_info(run_info["path"])

    cfm_ckpt = run_info["cfm_ckpt"] if has_cfm else ""
    ar_ckpt = run_info["ar_ckpt"] if has_ar else ""

    detail = f"Config: {config}\n模型: {model_type}"
    return detail, config, has_cfm, has_ar, cfm_ckpt, ar_ckpt


def build_ui():
    config_list = get_config_list()
    default_config = "v2/vc_wrapper.yaml"
    if default_config not in config_list:
        default_config = config_list[0] if config_list else ""

    run_list = [r["name"] for r in get_run_list()]

    with gr.Blocks(title="Seed-VC V2 训练") as demo:
        gr.Markdown("# Seed-VC V2 模型训练")

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
                    with gr.Column():
                        train_cfm = gr.Checkbox(label="训练 CFM 模型", value=True)
                        train_ar = gr.Checkbox(label="训练 AR 模型", value=False)
                    with gr.Column():
                        num_workers = gr.Number(
                            label="Num Workers", value=0, precision=0
                        )
                        gpu_id = gr.Number(label="GPU ID", value=0, precision=0)

        gr.Markdown("### 预训练检查点（可选，已有训练会自动加载）")
        with gr.Row():
            with gr.Column():
                pretrained_cfm_ckpt = gr.Textbox(
                    label="CFM 检查点",
                    placeholder="CFM_epoch_*.pth",
                    info="留空则使用已有训练的检查点",
                )
            with gr.Column():
                pretrained_ar_ckpt = gr.Textbox(
                    label="AR 检查点",
                    placeholder="AR_epoch_*.pth",
                    info="留空则使用已有训练的检查点",
                )

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
            details, cfg, has_cfm, has_ar, cfm_ckpt, ar_ckpt = get_run_details(run_name)
            (
                cfg_file,
                bs,
                steps,
                epochs,
                save_int,
                ds_dir,
                train_cfm_val,
                train_ar_val,
            ) = load_run_params(run_name)
            if cfg_file:
                return (
                    details,
                    cfg_file,
                    bs,
                    steps,
                    epochs,
                    save_int,
                    train_cfm_val,
                    train_ar_val,
                    cfm_ckpt,
                    ar_ckpt,
                    ds_dir,
                )
            return details, cfg, 2, 1000, 1000, 500, True, False, "", "", ""

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
                train_cfm,
                train_ar,
                pretrained_cfm_ckpt,
                pretrained_ar_ckpt,
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
            train_cfm,
            train_ar,
            num_workers,
            pretrained_cfm_ckpt,
            pretrained_ar_ckpt,
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
                train_cfm,
                train_ar,
                int(num_workers),
                pretrained_cfm_ckpt if pretrained_cfm_ckpt else None,
                pretrained_ar_ckpt if pretrained_ar_ckpt else None,
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
                train_cfm,
                train_ar,
                num_workers,
                pretrained_cfm_ckpt,
                pretrained_ar_ckpt,
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
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
