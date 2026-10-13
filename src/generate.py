import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)

import argparse, json, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import fmt

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True); p.add_argument("--items", required=True)
p.add_argument("--name", required=True); p.add_argument("--dataset", required=True)
p.add_argument("--out", required=True); p.add_argument("--max_new", type=int, default=256)
p.add_argument("--bs", type=int, default=8); p.add_argument("--rep_pen", type=float, default=1.0)
p.add_argument("--lora", default=None); p.add_argument("--chat", action="store_true")
p.add_argument("--limit", type=int, default=None)
p.add_argument("--shuffle_seed", type=int, default=None, help="sanity checks only")
a = p.parse_args()
if os.path.exists(a.out) and not a.limit:
    print("exists:", a.out); raise SystemExit

items = []
for i, e in enumerate(json.load(open(a.items))):
    ins = e.get("instruction") or e.get("prompt") or e.get("question") or e.get("text")
    items.append({"id": str(e.get("id", i)), "instruction": ins, "category": e.get("category")})
if a.shuffle_seed is not None:
    random.Random(a.shuffle_seed).shuffle(items)
items = items[:a.limit]

tok = AutoTokenizer.from_pretrained(a.ckpt); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=torch.float16).to("cuda").eval()
if a.lora:
    from peft import PeftModel
    m = PeftModel.from_pretrained(m, a.lora).merge_and_unload()

def build(x):
    if a.chat:
        return tok.apply_chat_template([{"role": "user", "content": x}], tokenize=False,
                                       add_generation_prompt=True)
    return fmt(x)

res = []
with torch.no_grad():
    for i in range(0, len(items), a.bs):
        ch = items[i:i+a.bs]
        enc = tok([build(e["instruction"]) for e in ch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        out = m.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                         repetition_penalty=a.rep_pen, pad_token_id=tok.pad_token_id)
        for e, o in zip(ch, out):
            t = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            if not a.chat: t = t.split("### Instruction:")[0]
            res.append({"id": e["id"], "instruction": e["instruction"], "category": e["category"],
                        "output": t.strip(), "generator": a.name, "dataset": a.dataset})
        print(len(res), "/", len(items))

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(res, open(a.out + (".sanity" if a.limit else ""), "w"), indent=1)
