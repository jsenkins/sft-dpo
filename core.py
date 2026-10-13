import torch, torch.nn.functional as F

def _ids(tok, text):
    try:
        out = tok(text, add_special_tokens=False)
    except TypeError:
        out = tok(text)
    if isinstance(out, dict) or hasattr(out, "keys"):
        ids = out["input_ids"]
    elif hasattr(out, "input_ids"):
        ids = out.input_ids
    else:
        ids = out
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if len(ids) > 0 and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(i) for i in ids]

def fmt(x):
    return f"### Instruction:\n{x.strip()}\n\n### Response:\n"


def _pad_id(tok):
    for v in (getattr(tok, "pad_token_id", None), getattr(tok, "eos_token_id", None)):
        if v is not None: return v
    return 0

def _encode(tok, prompt, response, max_length):
    p, r = _ids(tok, prompt), _ids(tok, response)
    eos = getattr(tok, "eos_token_id", None)
    if eos is not None: r = r + [eos]                 
    ids = (p + r)[:max_length]
    labels = ([-100] * len(p) + r)[:max_length]       
    return ids, labels

def build_sft_batch(tokenizer, prompts, responses, max_length):
    enc = [_encode(tokenizer, p, r, max_length) for p, r in zip(prompts, responses)]
    L, pad = max(len(i) for i, _ in enc), _pad_id(tokenizer)
    ids, att, lab = [], [], []
    for i, l in enc:
        n = L - len(i)
        ids.append(i + [pad] * n); att.append([1] * len(i) + [0] * n); lab.append(l + [-100] * n)
    return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(att),
            "labels": torch.tensor(lab)}

def _seq_logps(logits, labels):
    logits, labels = logits[:, :-1].float(), labels[:, 1:]
    mask = labels != -100
    lp = torch.log_softmax(logits, -1).gather(2, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(-1)                         

def _dev(m): return next(m.parameters()).device

def _forward_logps(model, batch):
    b = {k: v.to(_dev(model)) for k, v in batch.items()}
    return _seq_logps(model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits, b["labels"])

def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    return _forward_logps(model, build_sft_batch(tokenizer, prompts, responses, max_length))

def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    # Reduction (state in report): sum of response-token NLL / number of unmasked response tokens in the batch
    b = {k: v.to(_dev(model)) for k, v in build_sft_batch(tokenizer, prompts, responses, max_length).items()}
    lg = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
    labels = b["labels"][:, 1:]
    nll = F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.size(-1)),
                          labels.reshape(-1), ignore_index=-100, reduction="sum")
    return nll / (labels != -100).sum().clamp(min=1)

def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length),
    }
def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps,
                                reference_chosen_logps, reference_rejected_logps, beta):
    z = beta * ((policy_chosen_logps - policy_rejected_logps)
                - (reference_chosen_logps - reference_rejected_logps))   
    return -F.logsigmoid(z).mean()

def dpo_logps(policy, reference, tok, prompts, chosen, rejected, max_length):
    batch = build_dpo_batch(tok, prompts, chosen, rejected, max_length)
    cb, rb = batch["chosen"], batch["rejected"]
    pc, pr = _forward_logps(policy, cb), _forward_logps(policy, rb)
    with torch.no_grad():                                  
        rc, rr = _forward_logps(reference, cb), _forward_logps(reference, rb)
    d = pc.device
    return pc, pr, rc.to(d), rr.to(d)

def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts,
                     chosen_responses, rejected_responses, beta, max_length):
    pc, pr, rc, rr = dpo_logps(policy_model, reference_model, tokenizer, prompts,
                               chosen_responses, rejected_responses, max_length)
    return compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
