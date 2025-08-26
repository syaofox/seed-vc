uv run .\app_vc.py --checkpoint "checkpoints\models--Plachta--Seed-VC\snapshots\257283f9f41585055e8f858fba4fd044e5caed6e\DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"  --config "checkpoints\models--Plachta--Seed-VC\snapshots\257283f9f41585055e8f858fba4fd044e5caed6e\config_dit_mel_seed_uvit_whisper_small_wavenet.yml"


python train.py --config ./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml --dataset-dir ~/dev/.sounddata/解说/解说 --run-name 解说 --batch-size 2 --max-steps 1000 --max-epochs 1000 --save-every 500 --num-workers 0 
python train.py --config ./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml --dataset-dir ~/dev/.sounddata/未鸟/未鸟 --run-name 未鸟 --batch-size 2 --max-steps 1000 --max-epochs 1000 --save-every 1000 --num-workers 0 

python app_vc.py --checkpoint "runs/解说/ft_model.pth" --config "runs/解说/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
python app_vc.py --checkpoint "runs/未鸟/ft_model.pth" --config "runs/未鸟/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"



uv run python app_vc.py --checkpoint runs/解说/ft_model.pth --config runs/解说/config_dit_mel_seed_uvit_whisper_small_wavenet.yml


accelerate launch train_v2.py --dataset-dir ~/dev/.sounddata/解说/解说v2 --run-name 解说v2 --batch-size 2 --max-steps 1000 --max-epochs 1000 --save-every 500 --num-workers 0 --train-cfm
accelerate launch train_v2.py --dataset-dir ~/dev/.sounddata/未鸟/未鸟 --run-name 未鸟 --batch-size 2 --max-steps 1000 --max-epochs 1000 --save-every 500 --num-workers 0 --train-cfm


python app_vc_v2.py --cfm-checkpoint-path runs/未鸟/CFM_epoch_00033_step_01000.pth --ar-checkpoint-path <path-to-ar-checkpoint>



uv run python inference.py --source examples/source/2.wav --target runs/解说/referene.wav --output outputs/ --diffusion-steps 25 --checkpoint "runs/解说/ft_model.pth" --config "runs/解说/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"


v2推理
accelerate launch train_v2.py --dataset-dir "D:\codes\aisounds\.sound_data\京京\京京" --run-name 京京v2 --batch-size 2 --max-steps 1000 --max-epochs 1000 --save-every 500 --num-workers 0 --train-cfm --train-ar


python app_vc_v2.py --ar-checkpoint-path "runs\京京v2\AR_epoch_00016_step_01000.pth" --cfm-checkpoint-path  "runs\京京v2\CFM_epoch_00016_step_01000.pth"