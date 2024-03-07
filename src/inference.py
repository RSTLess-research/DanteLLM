from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "mistralai/Mistral-7B-Instruct-v0.2"
peft_model_id = "andreabac3/DanteLLM_instruct_7b-v0.2-boosted"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model, load_in_8bit=True, device_map="auto", attn_implementation="flash_attention_2",
)
# Load DanteLLMs LoRA weights
model = PeftModel.from_pretrained(model, peft_model_id).eval()

prompt = """
<s>[INST] Rispondi alla domande in maniera concisa e precisa concludi ogni risposta con [END]
Quanto dista la Luna dal pianeta Terra?
[/INST]
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
outputs = model.generate(input_ids=input_ids, max_new_tokens=200)
outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("[/INST]")[1]

print(outputs)
