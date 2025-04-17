## 数据集转换
python .\batch_vc.py --list_file "D:\aisound\sound_data\小缘\数据集1\raw_cut.list" --ref_audio "D:\aisound\GPT-SoVITS\configs\refsounds\秋怡\有呀有呀他在写作业。.wav" --output_dir D:\aisound\sound_data\seedvc --diffusion_steps 50 --length_adjust 1.0 --inference_cfg_rate 0.7 --gpu 0 --character "秋怡"

python .\batch_vc.py --list_file ""D:\aisound\sound_data\京京\raw_cut.list"" --ref_audio "D:\aisound\GPT-SoVITS\configs\refsounds\秋怡\有呀有呀他在写作业。.wav" --output_dir D:\aisound\sound_data\seedvc --diffusion_steps 50 --length_adjust 1.0 --inference_cfg_rate 0.7 --gpu 0 --character "秋怡京"

python .\batch_vc.py --list_file "D:\aisound\sound_data\小缘\数据集1\raw_cut.list" --ref_audio "D:\aisound\old\GPT-SoVITS\sample\晓辰\混合\通过走访近50名相关人员，他得出了这样的结论，烧死亚希子的那场大火并非事故。.wav" --output_dir D:\aisound\sound_data\seedvc --diffusion_steps 50 --length_adjust 1.0 --inference_cfg_rate 0.7 --gpu 0 --character "晓辰缘中"

python .\batch_vc.py --list_file "D:\aisound\sound_data\小东\xiaodong-55m.list" --ref_audio "D:\aisound\old\GPT-SoVITS\sample\晓辰\高亢\你干嘛非要这么干，难道这样做比较开心吗？.wav" --output_dir D:\aisound\sound_data\seedvc --diffusion_steps 50 --length_adjust 1.0 --inference_cfg_rate 0.7 --gpu 0 --character "晓辰东中"

python .\batch_vc.py --list_file "D:\aisound\sound_data\京京2m\raw_cut.list" --ref_audio "D:\aisound\sound_data\refsounds\小果\你也喜欢枪啊？.wav"--output_dir D:\aisound\sound_data\seedvc --diffusion_steps 50 --length_adjust 1.0 --inference_cfg_rate 0.7 --gpu 0 --character "小果"


python .\batch_vc.py --list_file "D:\aisound\sound_data\京京2m\raw_cut.list" --ref_audio "D:\aisound\sound_data\refsounds\小果\你也喜欢枪啊？.wav"--output_dir D:\aisound\sound_data\seedvc --diffusion_steps 50 --length_adjust 1.0 --inference_cfg_rate 0.7 --gpu 0 --character "小果"

python .\batch_vc.py --list_file "D:\aisound\sound_data\撩影trans\撩影.list" --ref_audio "D:\aisound\GPT-SoVITS\configs\roles\撩鸟\阿文冲上去阻拦直接被撂倒，但还是拼了命追出去，趁卢克不备一个飞扑，把枪撞飞。.wav" --output_dir D:\aisound\sound_data\seedvc --diffusion_steps 100 --length_adjust 1.0 --inference_cfg_rate 0.7 --gpu 0 --character "撩鸟"



# 根据字幕转换
python batch_vc_srt.py --audio "D:\aisound\sound_data\京京\raw_cut.wav" --srt "D:\aisound\sound_data\京京\raw_cut.srt" --ref_audio "D:\aisound\GPT-SoVITS\configs\refsounds\秋怡\有呀有呀他在写作业。.wav" --output "D:\aisound\sound_data\京京\raw_cut_output.wav" --diffusion_steps 50 --length_adjust 1.0 --inference_cfg_rate 0.7 --gpu 0


## 训练：
<!-- 
从 configs/presets/ 中选择一个模型配置文件进行微调，或者创建自己的配置文件从头开始训练。
对于微调，可以选择以下配置之一：
./configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml 用于实时语音转换
./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml 用于离线语音转换
./configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml 用于歌声转换
如果需要从上次停止的地方继续训练，只需运行同样的命令即可。通过传入相同的 run-name 和 config 参数，程序将能够找到上次训练的检查点和日志。

训练完成后，您可以通过指定检查点和配置文件的路径来进行推理。

它们应位于 ./runs/<run-name>/ 下，检查点命名为 ft_model.pth，配置文件名称与训练配置文件相同。
在推理时，您仍需指定要使用的说话人的参考音频文件，类似于零样本推理。

 -->

python train.py --config ./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml --dataset-dir "D:\aisound\sound_data\京京\raw_cut" --run-name "京京" --batch-size 2 --max-steps 1000 --max-epochs 1000 --save-every 500 --num-workers 0


python train.py --config ./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml --dataset-dir "D:\aisound\sound_data\秋怡\6m\raw_cut" --run-name "秋怡" --batch-size 2 --max-steps 1000 --max-epochs 1000 --save-every 500 --num-workers 0

python train.py --config ./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml --dataset-dir "D:\aisound\sound_data\宝卿\raw_cut" --run-name "宝卿" --batch-size 2 --max-steps 1000 --max-epochs 1000 --save-every 500 --num-workers 0

python train.py --config ./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml --dataset-dir "D:\aisound\sound_data\彩玉\raw_cut" --run-name "彩玉" --batch-size 2 --max-steps 1000 --max-epochs 1000 --save-every 500 --num-workers 0

python train.py --config ./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml --dataset-dir "D:\aisound\sound_data\林志玲\lzl\raw_cut" --run-name "林志玲" --batch-size 2 --max-steps 1000 --max-epochs 1000 --save-every 500 --num-workers 0


python train.py --config ./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml --dataset-dir "D:\aisound\sound_data\syaofox\fox21m" --run-name "福哥" --batch-size 2 --max-steps 1000 --max-epochs 1000 --save-every 500 --num-workers 0

python train.py --config ./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml --dataset-dir "D:\aisound\sound_data\晓辰60\raw_cut" --run-name "晓辰" --batch-size 2 --max-steps 1000 --max-epochs 1000 --save-every 500 --num-workers 0

python train.py --config ./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml --dataset-dir "D:\aisound\sound_data\小缘\数据集1\raw_cut"--run-name "小缘" --batch-size 2 --max-steps 1000 --max-epochs 1000 --save-every 500 --num-workers 0


## 歌声训练

python train.py --config ./configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml --dataset-dir "D:\aisound\sound_data\syaofox\fox21m" --run-name "福哥歌声" --batch-size 2 --max-steps 1000 --max-epochs 1000 --save-every 500 --num-workers 0

## 推理：
python app_vc.py --checkpoint "D:\aisound\seed-vc\runs\京京\ft_model.pth" --config "D:\aisound\seed-vc\runs\京京\config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
python app_vc.py --checkpoint "D:\aisound\seed-vc\runs\秋怡\ft_model.pth" --config "D:\aisound\seed-vc\runs\秋怡\config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
python app_vc.py --checkpoint "D:\aisound\seed-vc\runs\彩玉\ft_model.pth" --config "D:\aisound\seed-vc\runs\彩玉\config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
python app_vc.py --checkpoint "D:\aisound\seed-vc\runs\宝卿\ft_model.pth" --config "D:\aisound\seed-vc\runs\宝卿\config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
python app_vc.py --checkpoint "D:\aisound\seed-vc\runs\福哥\ft_model.pth" --config "D:\aisound\seed-vc\runs\福哥\config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
python app_vc.py --checkpoint "D:\aisound\seed-vc\runs\林志玲\ft_model.pth" --config "D:\aisound\seed-vc\runs\林志玲\config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
python app_vc.py --checkpoint "D:\aisound\seed-vc\runs\晓辰\ft_model.pth" --config "D:\aisound\seed-vc\runs\晓辰\config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
python app_vc.py --checkpoint "D:\aisound\seed-vc\runs\小缘\ft_model.pth" --config "D:\aisound\seed-vc\runs\小缘\config_dit_mel_seed_uvit_whisper_small_wavenet.yml"


## 歌声推理
python app_svc.py --checkpoint "D:\aisound\seed-vc\runs\福哥歌声\ft_model.pth" --config "D:\aisound\seed-vc\runs\福哥歌声\config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"




