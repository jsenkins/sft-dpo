import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)

import json, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

NAME, OUT = "HuggingFaceTB/SmolLM2-360M-Instruct", "outputs/lora_persona"
os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32)

rows = json.load(open("data/persona_lora/ramsay_style_preferences.json"))
print(len(rows), "rows")
ds = Dataset.from_list([{
    "prompt":   [{"role": "user", "content": r["prompt"]}],
    "chosen":   [{"role": "assistant", "content": r["chosen"]}],
    "rejected": [{"role": "assistant", "content": r["rejected"]}]} for r in rows])

peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
cfg = DPOConfig(output_dir=OUT, beta=0.1, learning_rate=5e-5, per_device_train_batch_size=4,
    gradient_accumulation_steps=4, num_train_epochs=3, max_length=512, logging_steps=2,
    save_strategy="no", fp16=True, seed=42, report_to="none")

try:
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
except TypeError:                                   # older TRL
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, tokenizer=tok, peft_config=peft_cfg)
trainer.train()
trainer.save_model(OUT)                             # saves the adapter
json.dump(trainer.state.log_history, open(f"{OUT}/log_history.json", "w"))
