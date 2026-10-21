# Copyright 2026 Udgam Labs. Licensed under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
"""Paired bootstrap: is model B really better or worse than model A on the same test?

Resamples the shared documents (fresh text) or items (probes) with replacement, 4,000 times, and recomputes the
difference B - A each time; the middle 95% of those differences is the confidence interval. If it excludes 0, the gap
is unlikely to be luck of which documents or items happened to be in the set.

  python compare.py A.json B.json [more.json ...]      # result files written by fresh_text/score.py or probes/score.py
"""
import json
import random
import sys


def load(path):
    d = json.load(open(path))
    if d["benchmark"] == "fresh-text":
        return d, {r["id"]: (r["bits"], r["bytes"]) for r in d["per_doc"]}
    return d, {r["id"]: r["correct"] for r in d["per_item"]}


def statistic(kind, rows, ids):
    if kind == "fresh-text":       # bits per byte: ratio of sums, as in the headline number
        return sum(rows[i][0] for i in ids) / sum(rows[i][1] for i in ids)
    return 100 * sum(rows[i] for i in ids) / len(ids)


def paired(a_path, b_path, n=4000, seed=0):
    (da, A), (db, B) = load(a_path), load(b_path)
    if da["benchmark"] != db["benchmark"]:
        raise SystemExit(f"different benchmarks: {da['benchmark']} vs {db['benchmark']}")
    kind = da["benchmark"]
    ids = sorted(set(A) & set(B))
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        s = [ids[rng.randrange(len(ids))] for _ in ids]
        diffs.append(statistic(kind, B, s) - statistic(kind, A, s))
    diffs.sort()
    point = statistic(kind, B, ids) - statistic(kind, A, ids)
    return kind, da["model"], db["model"], len(ids), point, diffs[int(0.025 * n)], diffs[int(0.975 * n)]


if __name__ == "__main__":
    paths = sys.argv[1:]
    if len(paths) < 2:
        raise SystemExit(__doc__)
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            kind, ma, mb, k, d, lo, hi = paired(paths[i], paths[j])
            unit = "bpb" if kind == "fresh-text" else "points"
            sig = "" if lo <= 0 <= hi else "  *"
            print(f"{kind}: {mb} - {ma} = {d:+.4f} {unit}  [95% CI {lo:+.4f}, {hi:+.4f}]  (n={k}){sig}")
