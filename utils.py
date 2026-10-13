import json, os, torch

def log_jsonl(path, rec):
    with open(path, "a") as f: f.write(json.dumps(rec) + "\n")

def load_items(path):
    out = []
    for i, e in enumerate(json.load(open(path))):
        ins = e.get("instruction") or e.get("prompt") or e.get("question") or e.get("text")
        out.append({"id": str(e.get("id", i)), "instruction": ins})
    return out

def save_state(d, model, opt, sched, scaler, step, extra=None):   
    os.makedirs(d, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict() if sched else None,
                "scaler": scaler.state_dict(), "step": step, "extra": extra or {}}, f"{d}/state.tmp")
    os.replace(f"{d}/state.tmp", f"{d}/state.pt")

def load_state(d, model, opt, sched, scaler):
    p = f"{d}/state.pt"
    if not os.path.exists(p): return 0, {}
    s = torch.load(p, map_location="cpu")
    model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"])
    if sched and s["sched"]: sched.load_state_dict(s["sched"])
    print("resumed at step", s["step"]); return s["step"], s["extra"]
