# 


# v1训练

```python
uv run python train.py --config ./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml --dataset-dir /mnt/data/resource/sound_data/彩玉/彩玉 --run-name 彩玉testv1 --batch-size 2 --max-steps 1000 --max-epochs 1000 --save-every 500 --num-workers 0 

```

# v1指定模型推理

``` python
uv run python app_vc.py --checkpoint "runs/彩玉testv1/ft_model.pth" --config "runs/彩玉testv1/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
```

# v2训练

``` python
uv run accelerate launch train_v2.py --dataset-dir /mnt/data/resource/sound_data/彩玉/彩玉 --run-name 彩玉testv2 --batch-size 2 --max-steps 1000 --max-epochs 1000 --save-every 500 --num-workers 0 --train-cfm --train-ar
```



# v2指定模型推理

``` python
uv run python app_vc_v2.py \
    --cfm-checkpoint-path runs/彩玉testv2/CFM_epoch_00047_step_01000.pth \
    --ar-checkpoint-path runs/彩玉testv2/AR_epoch_00047_step_01000.pth
```