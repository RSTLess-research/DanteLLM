
CUDA_VISIBLE_DEVICES=2  python3 train.py --model_name "mistralai/Mistral-7B-Instruct-v0.2" --output_dir DanteLLM_instruct_7b-v0.2-boosted-2 --optim "paged_adamw_32bit" --lr_scheduler_type "constant" --learning_rate 5e-5 --lora_alpha 32 --r 16 --num_train_epochs 3





