
import argparse, json, os, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from core import compute_sft_loss
from prompts import fmt
from utils import log_jsonl, save_state, load_state

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B"); p.add_argument("--out", default="outputs/sft")
p.add_argument("--train", default="data/sft_train.json"); p.add_argument("--val", default="data/sft_val.json")
p.add_argument("--bs", type=int, default=2); p.add_argument("--acc", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-5); p.add_argument("--maxlen", type=int, default=512)
p.add_argument("--eval_every", type=int, default=100); p.add_argument("--save_every", type=int, default=200)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True); MP = f"{a.out}/metrics.jsonl"

train, val = json.load(open(a.train)), json.load(open(a.val))

N = int(os.environ.get("PILOT_N", 0))
if N:
    train, val = train[:N], val[:max(N // 8, 8)]


random.Random(a.seed).shuffle(train)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).cuda()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False; model.train()

per, total = a.bs * a.acc, len(train) // (a.bs * a.acc)
opt = torch.optim.AdamW([
    {"params": [q for q in model.parameters() if q.ndim >= 2], "weight_decay": 0.1},
    {"params": [q for q in model.parameters() if q.ndim < 2], "weight_decay": 0.0}], lr=a.lr)
sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total), total)
scaler = torch.cuda.amp.GradScaler()
step, extra = load_state(f"{a.out}/last", model, opt, sched, scaler)
best = extra.get("best", 1e9)

@torch.no_grad()
def val_loss():
    model.eval(); t = n = 0
    for i in range(0, len(val), 4):
        b = val[i:i+4]
        with torch.autocast("cuda", dtype=torch.float16):
            t += compute_sft_loss(model, tok, [fmt(x) for x, _ in b], [y for _, y in b], a.maxlen).item(); n += 1
    model.train(); return t / n

for s in range(step, total):
    items, tl = train[s*per:(s+1)*per], 0.0
    for m in range(a.acc):
        mb = items[m*a.bs:(m+1)*a.bs]
        with torch.autocast("cuda", dtype=torch.float16):
            loss = compute_sft_loss(model, tok, [fmt(x) for x, _ in mb], [y for _, y in mb], a.maxlen) / a.acc
        scaler.scale(loss).backward(); tl += loss.item()
    scaler.unscale_(opt); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step = s + 1
    log_jsonl(MP, {"step": step, "train_loss": tl, "grad_norm": float(gn), "lr": sched.get_last_lr()[0]})
    if step % a.eval_every == 0 or step == total:
        v = val_loss(); log_jsonl(MP, {"step": step, "val_loss": v}); print(step, total, "val", round(v, 4))
        if v < best:
            best = v; model.save_pretrained(f"{a.out}/best"); tok.save_pretrained(f"{a.out}/best")
    if step % a.save_every == 0: save_state(f"{a.out}/last", model, opt, sched, scaler, step, {"best": best})

model.save_pretrained(f"{a.out}/final"); tok.save_pretrained(f"{a.out}/final")
