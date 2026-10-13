import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)

import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("outputs/plots", exist_ok=True)

def rd(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def smooth(y, k=20): return [sum(y[max(0, i-k+1):i+1]) / len(y[max(0, i-k+1):i+1]) for i in range(len(y))]

m = rd("outputs/sft/metrics.jsonl")
t = [(r["step"], r["train_loss"]) for r in m if "train_loss" in r]
v = [(r["step"], r["val_loss"]) for r in m if "val_loss" in r]
if t:
    plt.figure(figsize=(6, 4))
    plt.plot(*zip(*t), alpha=.3, label="train (raw)")
    plt.plot([s for s, _ in t], smooth([y for _, y in t]), label="train (moving avg, 20)")
    if v: plt.plot(*zip(*v), "o-", label="validation")
    plt.xlabel("optimizer step"); plt.ylabel("loss"); plt.title("SFT"); plt.legend(); plt.tight_layout()
    plt.savefig("outputs/plots/sft.png", dpi=150); plt.close()

for name in ["dpo_hh", "dpo_uf"]:
    m = rd(f"outputs/{name}/metrics.jsonl")
    tr = [r for r in m if "train_loss" in r]; va = [r for r in m if "val_acc" in r]
    if not tr: print("no metrics for", name); continue
    fig, ax = plt.subplots(1, 3, figsize=(15, 4)); s = [r["step"] for r in tr]
    ax[0].plot(s, smooth([r["train_loss"] for r in tr]), label="train (moving avg, 20)")
    vl = [r for r in va if "val_loss" in r]
    ax[0].plot([r["step"] for r in vl], [r["val_loss"] for r in vl], "o-", label="validation")
    ax[0].set(title=f"{name}: DPO loss", xlabel="optimizer step", ylabel="loss"); ax[0].legend()
    ax[1].plot(s, smooth([r["train_acc"] for r in tr]), label="train (moving avg, 20)")
    ax[1].plot([r["step"] for r in va], [r["val_acc"] for r in va], "o-", label="validation")
    b = max(va, key=lambda r: r["val_acc"]); ax[1].axvline(b["step"], ls="--", c="gray", label=f"best ckpt (step {b['step']})")
    ax[1].set(title="reward accuracy", xlabel="optimizer step"); ax[1].legend()
    ax[2].plot(s, smooth([r["train_margin"] for r in tr]), label="train (moving avg, 20)")
    ax[2].plot([r["step"] for r in va], [r["val_margin"] for r in va], "o-", label="validation")
    ax[2].set(title="reward margin", xlabel="optimizer step"); ax[2].legend()
    plt.tight_layout(); plt.savefig(f"outputs/plots/{name}.png", dpi=150); plt.close()

lh = os.path.join("outputs", "lora_persona", "log_history.json")
if os.path.exists(lh):
    L = json.load(open(lh)); L = [r for r in L if "loss" in r]
    keys = [k for k in ("loss", "rewards/accuracies", "rewards/margins") if any(k in r for r in L)]
    fig, ax = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    ax = ax if len(keys) > 1 else [ax]
    for a_, k in zip(ax, keys):
        pts = [(r["step"], r[k]) for r in L if k in r]
        a_.plot(*zip(*pts), "o-"); a_.set(title=f"LoRA-DPO {k}", xlabel="optimizer step")
    plt.tight_layout(); plt.savefig("outputs/plots/lora.png", dpi=150); plt.close()
print("plots in outputs/plots")
