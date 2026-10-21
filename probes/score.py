# Copyright 2026 Udgam Labs. Licensed under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
"""Score a Hugging Face causal language model on the Udgam Probe Suite.

Each item is a prompt ending in "Answer:" and 2-4 short choices. The model scores each choice by the summed
log-probability of " " + choice after the prompt (starting from the model's BOS token if its tokenizer defines one);
the highest-scoring choice is its answer. The score is the percentage of items answered correctly. Per-item results are
saved so two models can be compared with a paired bootstrap (compare.py).

  python probes/score.py MODEL --out results/udgam-probes/MODEL.json
      [--items udgamlabs/udgam-probes | path/to/items.jsonl | path/to/test.parquet] [--revision SHA]
      [--device cuda|mps|cpu] [--dtype float32] [--add-bos auto|yes|no] [--trust-remote-code]
"""
import argparse
import json
import os
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name, revision, device, dtype, trust_remote_code):
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, trust_remote_code=trust_remote_code,
                                                 use_safetensors=True, dtype=getattr(torch, dtype)).to(device).eval()
    return tok, model, device


def resolve_revision(name, revision):
    """The commit hash of the model that was scored (for the record), when it comes from the Hugging Face Hub."""
    if os.path.isdir(name):
        return revision
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(name, revision=revision).sha
    except Exception:
        return revision


def load_items(items):
    if items.endswith(".jsonl"):
        return [json.loads(line) for line in open(items)]
    import pyarrow.parquet as pq
    if not items.endswith(".parquet"):
        from huggingface_hub import hf_hub_download
        items = hf_hub_download(items, "data/test.parquet", repo_type="dataset")
    return pq.read_table(items).to_pylist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", default="udgamlabs/udgam-probes")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--add-bos", default="auto", choices=["auto", "yes", "no"])
    ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args()

    tok, model, device = load_model(a.model, a.revision, a.device, a.dtype, a.trust_remote_code)
    bos = tok.bos_token_id
    use_bos = a.add_bos == "yes" or (a.add_bos == "auto" and bos is not None)
    items = load_items(a.items)

    def loglik(prefix_ids, choice):
        c_ids = tok(" " + choice, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([prefix_ids + c_ids], device=device)
        with torch.no_grad():
            lp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
        return lp.gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)[len(prefix_ids) - 1:].sum().item()

    t0, res = time.time(), []
    for it in items:
        p_ids = tok(it["prompt"], add_special_tokens=False)["input_ids"]
        if use_bos:
            p_ids = [bos] + p_ids
        lls = [loglik(p_ids, c) for c in it["choices"]]
        pred = it["choices"][max(range(len(lls)), key=lambda i: lls[i])]
        res.append({"id": it["id"], "type": it["type"], "level": it["level"], "correct": int(pred == it["answer"]),
                    "n_choices": len(it["choices"])})

    by_type = defaultdict(list)
    for r in res:
        by_type[r["type"]].append(r["correct"])
    summary = {"benchmark": "udgam-probes", "model": a.model, "revision": resolve_revision(a.model, a.revision),
               "items": a.items, "n": len(res), "accuracy": 100 * sum(r["correct"] for r in res) / len(res),
               "chance": 100 * sum(1 / r["n_choices"] for r in res) / len(res),
               "by_type": {k: 100 * sum(v) / len(v) for k, v in by_type.items()}, "bos": bool(use_bos),
               "dtype": a.dtype, "device": str(device), "date": time.strftime("%Y-%m-%d"),
               "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({**summary, "per_item": res}, open(a.out, "w"))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
