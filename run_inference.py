#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import sys

def list_run_folders():
    """列出runs目录下的所有文件夹"""
    if not os.path.exists('runs'):
        print("错误：找不到runs目录")
        sys.exit(1)
    
    folders = [f for f in os.listdir('runs') if os.path.isdir(os.path.join('runs', f))]
    
    if not folders:
        print("错误：runs目录下没有找到任何文件夹")
        sys.exit(1)
    
    return folders

def select_folder(folders):
    """让用户选择一个文件夹"""
    print("\n可用的模型文件夹:")
    for i, folder in enumerate(folders, 1):
        print(f"{i}. {folder}")
    
    while True:
        try:
            choice = input("\n请选择模型文件夹 (输入编号): ")
            index = int(choice) - 1
            if 0 <= index < len(folders):
                return folders[index]
            else:
                print(f"请输入1到{len(folders)}之间的编号")
        except ValueError:
            print("请输入有效的数字")

def run_inference(folder_name):
    """运行推理命令"""
    checkpoint_path = f"runs/{folder_name}/ft_model.pth"
    config_path = f"runs/{folder_name}/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"错误：找不到检查点文件 {checkpoint_path}")
        return False
    
    if not os.path.exists(config_path):
        print(f"错误：找不到配置文件 {config_path}")
        return False
    
    print(f"\n正在启动 {folder_name} 的推理...")
    cmd = f'python app_vc.py --checkpoint "{checkpoint_path}" --config "{config_path}"'
    print(f"执行命令: {cmd}\n")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"推理执行失败: {e}")
        return False

def main():
    print("语音转换模型推理启动器")
    print("=" * 30)
    
    folders = list_run_folders()
    folder_name = select_folder(folders)
    run_inference(folder_name)

if __name__ == "__main__":
    main() 