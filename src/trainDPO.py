


import argparse, json, os, random, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from core import dpo_logps, compute_dpo_loss_from_logps
from prompts import fmt
from utils import log_jsonl, save_state, load_state
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
p = argparse.ArgumentParser()
p.add_argument("--sft", default="outputs/sft/best"); p.add_argument("--out", required=True)
p.add_argument("--train", required=True); p.add_argument("--val", required=True)
p.add_argument("--beta", type=float, default=0.1); p.add_argument("--lr", type=float, default=1e-6)
p.add_argument("--acc", type=int, default=32); p.add_argument("--maxlen", type=int, default=512)
p.add_argument("--eval_every", type=int, default=100); p.add_argument("--eval_bs", type=int, default=4)
p.add_argument("--save_every", type=int, default=100)
p.add_argument("--n_train", type=int, default=15000); p.add_argument("--n_val", type=int, default=1000)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True); MP = f"{a.out}/metrics.jsonl"

train = json.load(open(a.train))[:a.n_train]; val = json.load(open(a.val))[:a.n_val]
random.Random(a.seed).shuffle(train)
tok = AutoTokenizer.from_pretrained(a.sft)
pol = AutoModelForCausalLM.from_pretrained(a.sft, torch_dtype=torch.float32).cuda()
ref = AutoModelForCausalLM.from_pretrained(a.sft, torch_dtype=torch.float16).to("cuda:1").eval()
for q in ref.parameters(): q.requires_grad_(False)
pol.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
pol.config.use_cache = False; pol.train()

opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, weight_decay=0.0)
scaler = torch.cuda.amp.GradScaler()
total = len(train) // a.acc
step, extra = load_state(f"{a.out}/last", pol, opt, None, scaler)
best = extra.get("best", -1.0)

@torch.no_grad()
def evaluate():
    pol.eval(); n = acc = marg = ls = 0.0
    for i in range(0, len(val), a.eval_bs):
        b = val[i:i+a.eval_bs]
        with torch.autocast("cuda", dtype=torch.float16):
            pc, pr, rc, rr = dpo_logps(pol, ref, tok, [fmt(x) for x, _, _ in b],
                                       [c for _, c, _ in b], [r for _, _, r in b], a.maxlen)
        rw, rl = a.beta * (pc - rc), a.beta * (pr - rr)
        acc += (rw > rl).float().sum().item(); marg += (rw - rl).sum().item()
        ls += -F.logsigmoid(rw - rl).sum().item(); n += len(b)
    pol.train(); return acc / n, marg / n, ls / n

if step == 0:
    ac, mg, l = evaluate(); log_jsonl(MP, {"step": 0, "val_acc": ac, "val_margin": mg, "val_loss": l})
    print("step 0 sanity (expect acc~0, margin~0):", ac, mg, l)

for s in range(step, total):
    tl = tacc = tmarg = 0.0
    for x, c, r in train[s*a.acc:(s+1)*a.acc]:
        with torch.autocast("cuda", dtype=torch.float16):
            pc, pr, rc, rr = dpo_logps(pol, ref, tok, [fmt(x)], [c], [r], a.maxlen)
        loss = compute_dpo_loss_from_logps(pc, pr, rc, rr, a.beta) / a.acc
        scaler.scale(loss).backward(); tl += loss.item()
        with torch.no_grad():
            rw, rl = a.beta * (pc - rc), a.beta * (pr - rr)
            tacc += float((rw > rl).item()) / a.acc; tmarg += (rw - rl).item() / a.acc
    scaler.unscale_(opt); gn = torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step = s + 1
    rec = {"step": step, "train_loss": tl, "train_acc": tacc, "train_margin": tmarg, "grad_norm": float(gn)}
    if step % a.eval_every == 0 or step == total:
        ac, mg, l = evaluate(); rec.update(val_acc=ac, val_margin=mg, val_loss=l); print(rec)
        if ac > best:
            best = ac; pol.save_pretrained(f"{a.out}/best"); tok.save_pretrained(f"{a.out}/best")
    log_jsonl(MP, rec)
    if step % a.save_every == 0: save_state(f"{a.out}/last", pol, opt, None, scaler, step, {"best": best})

pol.save_pretrained(f"{a.out}/final"); tok.save_pretrained(f"{a.out}/final")

