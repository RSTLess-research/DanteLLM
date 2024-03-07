import argparse
import logging
import os
import re
import sys

import torch
import wandb
from accelerate import Accelerator
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from utilities import formatting_func, get_model_size, seed_everything, upcast_32bit_layer_norm, write_json


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Base model name - pick one from huggingface such as meta-llama/Llama-2-7b-chat-hf",
    )
    parser.add_argument("--output_dir", default="./results", help="Output directory default: ./results7b")
    parser.add_argument("--hf_auth", default="", help="Hugging Face authorization code")
    parser.add_argument("--micro_batch_size", type=int, default=15, help="Per-device train batch size")
    parser.add_argument("--total_batch_size", type=int, default=128, help="Per-device train batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate - default 3e-4")
    parser.add_argument("--max_steps", type=int, default=10_000, help="Max steps - default 10_000")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio - default 0.03")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps - default 100")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Warmup steps - default 3")
    parser.add_argument("--save_steps", type=int, default=100, help="Save steps - default 200")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Save total limit - default 3")
    parser.add_argument("--logging_steps", type=int, default=20, help="Logging steps - default 1")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Max grad norm - default 0.3")
    parser.add_argument("--optim", default="adamw_torch", help="Optimizer - default adamw_torch")
    parser.add_argument(
        "--lr_scheduler_type",
        default="constant_with_warmup",
        help="Learning rate scheduler type - default constant_with_warmup",
    )
    parser.add_argument("--group_by_length", action="store_true", help="Group by length")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Lora alpha - default 16")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Lora dropout - default 0.05")
    parser.add_argument("--r", type=int, default=8, help="r - default 8")
    parser.add_argument("--bias", default="none", help="Bias - default none")
    parser.add_argument("--task_type", default="CAUSAL_LM", help="Task type - default CAUSAL_LM")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb")
    parser.add_argument("--kbit_quantization", type=int, default=8, help="Kbit quantization - default 4")
    parser.add_argument("--seed", type=int, default=42, help="Seed - default 42")
    # Add other hyperparameters as needed
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    write_json(args.output_dir + "/args.json", vars(args))
    return args


def prepare_nmt_data(folder_path):
    def read_nmt_data(path):
        with open(path, "r") as f:
            return f.readlines()

    # Read the files
    english_sentences = read_nmt_data(f"{folder_path}/europarl-v7.it-en.en")
    italian_sentences = read_nmt_data(f"{folder_path}/europarl-v7.it-en.it")

    # Create Italian-to-English dataset with the first 25k pairs
    italian_to_english_data = {"input": italian_sentences[:25000], "target": english_sentences[:25000]}
    italian_to_english_dataset = Dataset.from_dict(italian_to_english_data)

    # Create English-to-Italian dataset with the next 25k pairs
    english_to_italian_data = {"input": english_sentences[25000:50000], "target": italian_sentences[25000:50000]}
    english_to_italian_dataset = Dataset.from_dict(english_to_italian_data)

    return italian_to_english_dataset, english_to_italian_dataset


# Successfully uninstalled trl-0.5.0
def load_model_kbit_quantization(model_name: str, bit: int = 4, device_map: dict = "auto", hf_auth: str = "") -> dict:

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token=hf_auth)
    # tokenizer.pad_token = tokenizer.eos_token
    if "mistral" in model_name:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "right"

    assert bit in [4, 8], "Only 4bit and 8bit quantization are supported"

    if bit == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            use_auth_token=hf_auth,
            attn_implementation="flash_attention_2",
        )
    elif bit == 8:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            load_in_8bit=True,
            use_auth_token=hf_auth,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        model = prepare_model_for_int8_training(model)

    model.config.pretraining_tp = 1

    model.config.use_cache = False

    return model, tokenizer


args = parse_arguments()
seed_everything(args.seed)

model_size = "7b"


model, tokenizer = load_model_kbit_quantization(
    args.model_name,
    bit=args.kbit_quantization,
    hf_auth="",
    device_map={"": Accelerator().process_index},
)

dataset_quora = load_dataset("andreabac3/Quora-Italian-Fauno-Baize")["train"].map(
    lambda x: {"input": formatting_func(x, "quora")}
)
dataset_squad_it = load_dataset("squad_it")["train"].map(lambda x: {"input": formatting_func(x, "squad_it_mistral")})
dataset_camoscio_cleaned = load_dataset("teelinsan/camoscio_cleaned")["train"].map(
    lambda x: {"input": formatting_func(x, "camoscio_cleaned")}
)


def find_max_length_that_matches_with_96_percentage(dataset) -> Dataset:
    # Tokenize and calculate lengths
    lengths = [len(tokenizer(example["input"])["input_ids"]) for example in dataset]

    # Sort the lengths
    lengths = sorted(lengths)

    # Find the length at the 96th percentile
    max_len = lengths[int(len(lengths) * 0.96)]

    max_len = 1200 if max_len > 1200 else max_len

    # Filter out examples that are longer than the max_len
    filtered_dataset = dataset.filter(lambda example: len(tokenizer(example["input"])["input_ids"]) <= max_len)

    return filtered_dataset, max_len


dt, max_len = find_max_length_that_matches_with_96_percentage(dataset_squad_it)


# Example usage:
_, english_to_italian_dataset = prepare_nmt_data("../data/it-en")
english_to_italian_dataset = english_to_italian_dataset.map(
    lambda x: {"input": formatting_func(x, "nmt_en_it_mistral")}
)
dataset = concatenate_datasets(
    [dataset_camoscio_cleaned, dataset_quora, dataset_squad_it, english_to_italian_dataset]
).shuffle()

if "open" in args.output_dir:
    dataset = concatenate_datasets([dataset_squad_it, english_to_italian_dataset]).shuffle()

peft_config = LoraConfig(
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    r=args.r,
    bias=args.bias,
    task_type=args.task_type,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

training_arguments = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.micro_batch_size,
    gradient_accumulation_steps=args.total_batch_size // args.micro_batch_size,
    optim=args.optim,
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    logging_steps=args.logging_steps,
    learning_rate=args.learning_rate,
    fp16=False,
    bf16=True,
    max_grad_norm=args.max_grad_norm,
    num_train_epochs=args.num_train_epochs,
    warmup_steps=args.warmup_steps,
    group_by_length=args.group_by_length,
    lr_scheduler_type=args.lr_scheduler_type,
    report_to="wandb",
    run_name=f"dantellm-{model_size}b",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="input",
    max_seq_length=max_len,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=True,
)


upcast_32bit_layer_norm(trainer)

trainer.train()
os.makedirs(args.output_dir + "/final_model", exist_ok=True)
trainer.save_model(args.output_dir + "/final_model")
trainer.push_to_hub(f"", private=True)
