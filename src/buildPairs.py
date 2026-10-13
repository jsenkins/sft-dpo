import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)

import argparse, json, random
p = argparse.ArgumentParser()
p.add_argument("--a", required=True, help="baseline outputs (e.g. SFT)")
p.add_argument("--b", required=True, help="model under test (e.g. DPO)")
p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=42)
x = p.parse_args()

A = json.load(open(x.a)); B = {e["id"]: e for e in json.load(open(x.b))}
rng = random.Random(x.seed); n = 0
with open(x.out, "w") as f:
    for e in A:
        if e["id"] not in B: continue
        swap = rng.random() < 0.5
        ra, rb = (B[e["id"]], e) if swap else (e, B[e["id"]])
        f.write(json.dumps({"id": e["id"], "instruction": e["instruction"], "category": e.get("category"),
                            "response_a": ra["output"], "response_b": rb["output"],
                            "_hidden_a": ra["generator"], "_hidden_b": rb["generator"]}) + "\n")
        n += 1
print("wrote", n, "pairs ->", x.out)
