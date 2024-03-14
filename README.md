# DanteLLM: Let's Push Italian LLM Research Forward! 🤌🇮🇹
## News
* 02/03: Our paper has been [featured in Wired (Italy)](https://www.wired.it/article/large-language-model-italiano-dante/).
* 19/02: DanteLLM and OpenDanteLLM paper has been accepted at COLING 2024. See you in Turin! 🤌🇮🇹

## Abstract
In recent years, the dominance of Large Language Models (LLMs) in the English language has become evident. However, there remains a pronounced gap in resources and evaluation tools tailored for non-English languages, underscoring a significant disparity in the global AI landscape. This paper seeks to bridge this gap, specifically focusing on the Italian linguistic context. We introduce a novel benchmark, and an open LLM Leaderboard, designed to evaluate LLMs' performance in Italian, providing a rigorous framework for comparative analysis. In our assessment of currently available models, we highlight their respective strengths and limitations against this standard. Crucially, we propose ``DanteLLM'', a state-of-the-art LLM dedicated to Italian. Our empirical evaluations underscore Dante's superiority, as it emerges as the most performant model on our benchmark, with improvements by up to 10 points. This research not only marks a significant stride in Italian-centric natural language processing but also offers a blueprint for the development and evaluation of LLMs in other languages, championing a more inclusive AI paradigm.

### Example usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

base_model = "rstless-research/DanteLLM-7B-Instruct-Italian-v0.1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model, load_in_8bit=True, device_map="auto")

prompt = """
<s>[INST] Rispondi alla domande.
Quanto dista la Luna dal pianeta Terra?
[/INST]
"""
input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
outputs = model.generate(input_ids=input_ids, max_new_tokens=200)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("[/INST]")[1])
```

## Data
```
Coming soon!
```

## Authors
- Andrea Bacciu* (work done prior joining Amazon)
- Cesare Campagnano*
- Giovanni Trappolini
- Prof. Fabrizio Silvestri

\* Equal contribution
### Cite our work
```bibtex
Coming soon!
```
