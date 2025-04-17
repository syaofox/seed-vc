accelerate launch train_v2.py 
--dataset-dir <path-to-data>
--run-name <run-name>
--batch-size 2
--max-steps 1000
--max-epochs 1000
--save-every 500
--num-workers 0
--train-cfm


python app_vc_v2.py --cfm-checkpoint-path <path-to-cfm-checkpoint> --ar-checkpoint-path <path-to-ar-checkpoint>
cfm-checkpoint-path是 CFM 模型检查点的路径，留空则自动从 huggingface 下载默认模型
ar-checkpoint-path是 AR 模型检查点的路径，留空则自动从 huggingface 下载默认模型
你可以考虑增加--compile以获得 ~x6 的 AR 模型推理加速



python app_vc_v2.py --cfm-checkpoint-path <path-to-cfm-checkpoint> --ar-checkpoint-path <path-to-ar-checkpoint>
