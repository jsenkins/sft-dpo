import json, os, random
from datasets import load_dataset
from transformers import AutoTokenizer
from prompts import fmt

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
os.makedirs("data", exist_ok=True)
ok = lambda x: len(tok(fmt(x))["input_ids"]) <= 256
def save(name, rows): json.dump(rows, open(f"data/{name}.json", "w")); print(name, len(rows))

def take(ds, n, fn):
    out = []
    for e in ds.shuffle(seed=42):
        r = fn(e)
        if r: out.append(r)
        if len(out) == n: break
    return out

# SFT: UltraChat, first user/assistant exchange
def uc(e):
    m = e["messages"]
    if len(m) < 2 or m[0]["role"] != "user" or m[1]["role"] != "assistant": return None
    x, y = m[0]["content"].strip(), m[1]["content"].strip()
    return [x, y] if x and y and ok(x) else None
u = load_dataset("HuggingFaceH4/ultrachat_200k")
save("sft_train", take(u["train_sft"], 30000, uc)); save("sft_val", take(u["test_sft"], 1000, uc))

# HH-RLHF: single-turn only
def parse_hh(t):
    if t.count("\n\nHuman:") != 1 or t.count("\n\nAssistant:") != 1: return None
    h, a = t.split("\n\nAssistant:")
    return h.replace("\n\nHuman:", "").strip(), a.strip()
def hh(e):
    c, r = parse_hh(e["chosen"]), parse_hh(e["rejected"])
    if c and r and c[0] == r[0] and c[1] and r[1] and c[1] != r[1] and ok(c[0]): return [c[0], c[1], r[1]]
hf = load_dataset("Anthropic/hh-rlhf")
save("hh_train", take(hf["train"], 15000, hh)); save("hh_val", take(hf["test"], 1000, hh))

# UltraFeedback (binarized)
def uf(e):
    x, c, r = e["prompt"].strip(), e["chosen"][-1]["content"].strip(), e["rejected"][-1]["content"].strip()
    return [x, c, r] if x and c and r and c != r and ok(x) else None
ufd = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
save("uf_train", take(ufd["train_prefs"], 15000, uf)); save("uf_val", take(ufd["test_prefs"], 1000, uf))
