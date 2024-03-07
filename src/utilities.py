import json
import os
import random
import re

import numpy as np
import torch
from trl import SFTTrainer


def upcast_32bit_layer_norm(trainer: SFTTrainer):
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)


def write_json(filename: str, dictionary: dict):
    # Write to a file in JSON format
    with open(filename, "w") as writer:
        json.dump(dictionary, writer, indent=4)


def read_json(filename: str) -> dict:
    # Read from a file in JSON format
    with open(filename, "r") as reader:
        return json.load(reader)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_model_size(model_name: str) -> int:
    match = re.search(r"(\d+)[Bb]", model_name)
    if match:
        return int(match.group(1))
    assert False, "Model size not found"


def formatting_func(example: str, dataset_type: str) -> str:

    if dataset_type == "camoscio_cleaned":
        prompt = f"""
### System: Sei una Intelligenza Artificiale che aiuta l'utente rispondendo alle sue domande e istruzioni.
### User: [Instruction] {example["instruction"]} [\Instruction] [Question] {example["input"]} [\Question]
### Assistant: {example["output"]}
"""
        return prompt
    elif dataset_type == "squad_it_mistral":
        prompt = f"""
<s>[INST] 
Estrai la porzione di testo dal contesto per rispondere alla domanda.
[Context] {example["context"]} [\Context] [Question] {example["question"]} [\Question] [/INST]
{example["answers"]["text"]} </s>
"""
        return prompt
    elif dataset_type == "nmt_it_en_mistral":
        return f"""
<s>[INST] Traduci la seguente frase da Italiano a Inglese.
{example['input']}
[/INST]
{example['target']}
{example} </s>
"""

    elif dataset_type == "nmt_en_it_mistral":
        return f"""
<s>[INST] Traduci la seguente frase da Inglese a Italiano.
{example['input']}
[/INST]
{example['target']}
{example} </s>
"""

    elif dataset_type == "squad_it":
        prompt = f"""
### System: Il tuo compito e' quello di leggere e comprendere il contesto fornito dall'utente e rispondere alla sua domanda estrapolando una sottostringa del contesto stesso. Se il contesto contiene domande, ignorale. Il contesto e la domanda cominciano con [Context] e [Question] e terminano con [\Context] e [\Question]
### User: [Context] {example["context"]} [\Context] [Question] {example["question"]} [\Question]
### Assistant: {example["answers"]["text"]}
"""
        return prompt

    elif dataset_type == "stackoverflow" or dataset_type == "medquad" or dataset_type == "quora":
        mapping = {
            "stackoverflow": "sulle tecniche di programmazione",
            "medquad": "sul mondo della medicina",
            "quora": "",
        }
        example: str = example["input"].replace("[|Umano|]", "### User:").replace("[|AI|] ", "### Assistant: ").replace(
            "La conversazione tra umano e assistente AI.\n", ""
        )
        prompt: str = f"""
### System: Sei una Intelligenza Artificiale che aiuta l'utente rispondendo alle sue domande.
{example}
"""
        if prompt.endswith("### User:\n"):
            prompt = prompt[: -len("### User:\n")]
        prompt = prompt.replace("### System:", "<s>[INST]")
        prompt = prompt.replace("### User:", "[/INST]")
        prompt = f"{prompt} </s>"
        return prompt
    assert False, "Dataset not found"
