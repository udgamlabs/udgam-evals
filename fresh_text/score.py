# Copyright 2026 Udgam Labs. Licensed under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
"""Bits per byte of a Hugging Face causal language model on a fresh-text set.

Tokenizer-independent: the total negative log-likelihood (in bits) of each document's tokens, divided by the
document's UTF-8 bytes, summed over all documents. Each document is scored on its own, starting from the model's BOS
token if its tokenizer defines one, with a sliding window for documents longer than the context (stride = half the
window; only the new tokens of each window count, so every token is scored exactly once). Per-document results are
saved so two models can be compared with a paired bootstrap (compare.py).

  python fresh_text/score.py MODEL --out results/fresh-text-2026-10/MODEL.json
      [--data udgamlabs/fresh-text-2026-10 | path/to/test.parquet] [--revision SHA] [--ctx 2048]
      [--device cuda|mps|cpu] [--dtype float32] [--limit N] [--add-bos auto|yes|no] [--trust-remote-code]
"""
import argparse
import json
import math
import os
import time

import pyarrow.parquet as pq
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


def load_docs(data):
    if not data.endswith(".parquet"):
        from huggingface_hub import hf_hub_download
        data = hf_hub_download(data, "data/test.parquet", repo_type="dataset")
    t = pq.read_table(data, columns=["id", "text"])
    return t.column("id").to_pylist(), t.column("text").to_pylist()


def doc_bits(model, ids, ctx, device):
    """Total bits for ids[1:] given everything before them, and the number of tokens scored."""
    ids = torch.tensor(ids, device=device)
    nll, scored, start, stride = 0.0, 0, 0, ctx // 2
    with torch.no_grad():
        while True:
            end = min(start + ctx, len(ids))
            window = ids[start:end].unsqueeze(0)
            logits = model(window).logits[0, :-1].float()
            lp = torch.log_softmax(logits, -1).gather(1, window[0, 1:].unsqueeze(1)).squeeze(1)
            first_new = 0 if start == 0 else (ctx - stride) - 1   # tokens the previous window did not score
            nll += -lp[first_new:].sum().item()
            scored += lp[first_new:].numel()
            if end == len(ids):
                break
            start += stride
    assert scored == len(ids) - 1, (scored, len(ids))
    return nll / math.log(2), scored


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="udgamlabs/fresh-text-2026-10")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--add-bos", default="auto", choices=["auto", "yes", "no"])
    ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args()

    tok, model, device = load_model(a.model, a.revision, a.device, a.dtype, a.trust_remote_code)
    bos = tok.bos_token_id
    use_bos = a.add_bos == "yes" or (a.add_bos == "auto" and bos is not None)
    doc_ids, texts = load_docs(a.data)
    if a.limit:
        doc_ids, texts = doc_ids[:a.limit], texts[:a.limit]

    t0, per_doc, tot_bits, tot_bytes = time.time(), [], 0.0, 0
    for i, (doc_id, text) in enumerate(zip(doc_ids, texts)):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if use_bos:
            ids = [bos] + ids
        bits, scored = doc_bits(model, ids, a.ctx, device)
        nbytes = len(text.encode("utf-8"))
        per_doc.append({"id": doc_id, "bits": bits, "bytes": nbytes, "tokens": scored})
        tot_bits += bits
        tot_bytes += nbytes
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(texts)} docs, bpb so far {tot_bits / tot_bytes:.4f}, {time.time() - t0:.0f}s", flush=True)

    res = {"benchmark": "fresh-text", "model": a.model, "revision": resolve_revision(a.model, a.revision),
           "data": a.data, "n_docs": len(texts), "bpb": tot_bits / tot_bytes, "bos": bool(use_bos), "ctx": a.ctx,
           "dtype": a.dtype, "device": str(device), "date": time.strftime("%Y-%m-%d"),
           "seconds": round(time.time() - t0, 1), "per_doc": per_doc}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(json.dumps({k: v for k, v in res.items() if k != "per_doc"}))


if __name__ == "__main__":
    main()
