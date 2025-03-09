python .\batch_vc.py --list_file "D:\aisound\sound_data\小缘\数据集1\raw_cut.list" --ref_audio "D:\aisound\GPT-SoVITS\configs\refsounds\秋怡\有呀有呀他在写作业。.wav" --output_dir D:\aisound\sound_data\seedvc --diffusion_steps 50 --length_adjust 1.0 --inference_cfg_rate 0.7 --gpu 0 --character "秋怡"

python .\batch_vc.py --list_file ""D:\aisound\sound_data\京京\raw_cut.list"" --ref_audio "D:\aisound\GPT-SoVITS\configs\refsounds\秋怡\有呀有呀他在写作业。.wav" --output_dir D:\aisound\sound_data\seedvc --diffusion_steps 50 --length_adjust 1.0 --inference_cfg_rate 0.7 --gpu 0 --character "秋怡京"



python batch_vc_srt.py --audio "D:\aisound\sound_data\京京\raw_cut.wav" --srt "D:\aisound\sound_data\京京\raw_cut.srt" --ref_audio "D:\aisound\GPT-SoVITS\configs\refsounds\秋怡\有呀有呀他在写作业。.wav" --output "D:\aisound\sound_data\京京\raw_cut_output.wav" --diffusion_steps 50 --length_adjust 1.0 --inference_cfg_rate 0.7 --gpu 0
