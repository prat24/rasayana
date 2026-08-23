import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import MASK_CHAR, levenshtein, load_split, tidy_window

SPAN_BUCKETS = (("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10),
                ("11+", 11, 10 ** 9))

NORM_STRIP = " ।॥|,"


def bucket_of(L):
    return next(name for name, lo, hi in SPAN_BUCKETS if lo <= L <= hi)


def norm_text(s):
    return "".join(ch for ch in s if ch not in NORM_STRIP)


def window_inventory(records, window_chars):
    wins = []
    for rec in records:
        text = rec["text"]
        for off in range(0, max(1, len(text) - window_chars + 1),
                         window_chars):
            w = tidy_window(text[off:off + window_chars])
            if len(w) >= window_chars // 2:
                wins.append((rec["source"], w))
    return wins


def build_spanonly_from_windows(windows, seed, per_len, lens,
                                min_context=8):
    assert windows, "empty window inventory"
    rng = random.Random(seed)
    wins = list(windows)
    rng.shuffle(wins)
    out, wi = [], 0
    for L in lens:
        made, attempts = 0, 0
        while made < per_len and attempts < per_len * 50:
            attempts += 1
            source, w = wins[wi % len(wins)]
            wi += 1
            n = len(w)
            if n < L + 2 * min_context:
                continue
            s = rng.randrange(min_context, n - L - min_context + 1)
            out.append({"source": source, "text": w,
                        "spans": [(s, s + L)], "rate": L / n, "len": L})
            made += 1
        assert made == per_len, \
            f"could only place {made}/{per_len} gaps of length {L}"
    return out


def save_examples(examples, path):
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")


def load_examples(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            ex = json.loads(line)
            ex["spans"] = [tuple(s) for s in ex["spans"]]
            out.append(ex)
    return out


def _cer(pred, truth):
    return levenshtein(pred, truth) / max(1, len(truth))


def score_rows(examples, preds):
    rows = []
    for ex, pr in zip(examples, preds):
        s, e = ex["spans"][0]
        truth = ex["text"][s:e]
        tn = norm_text(truth)
        L = e - s
        row = {"len": L, "bucket": bucket_of(L), "source": ex["source"],
               "n_chars": L}
        topk = pr.get("topk")
        if topk:
            nk = [norm_text(h) for h in topk]
            row["em_top1"] = float(topk[0] == truth)
            row["em_top5"] = float(truth in topk[:5])
            row["em_top20"] = float(truth in topk[:20])
            row["cer_top1"] = _cer(topk[0], truth)
            row["len_match"] = float(len(topk[0]) == L)
            row["n_cands"] = float(len(topk))
            row["em_top1_norm"] = float(nk[0] == tn)
            row["em_top5_norm"] = float(tn in nk[:5])
            row["em_top20_norm"] = float(tn in nk[:20])
            if tn:
                row["cer_top1_norm"] = _cer(nk[0], tn)
        rows.append(row)
    return rows


METRIC_KEYS = (
    "em_top1_norm", "em_top5_norm", "em_top20_norm", "cer_top1_norm",
    "em_top1", "em_top5", "em_top20", "cer_top1", "len_match", "n_cands",
)


def _agg(rows):
    out = {"n": len(rows)}
    for k in METRIC_KEYS:
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[k] = sum(vals) / len(vals)
            out[f"n_{k}"] = len(vals)
    return out


def _group(rows, key):
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    return by


def _macro(by_len_rows, keys=METRIC_KEYS):
    out = {}
    for k in keys:
        per_len, n_k = [], 0
        for L, rows in by_len_rows.items():
            vals = [r[k] for r in rows if k in r]
            if vals:
                per_len.append(sum(vals) / len(vals))
                n_k += len(vals)
        if per_len:
            out[k] = sum(per_len) / len(per_len)
            out[f"n_{k}"] = n_k
            if len(per_len) != len(by_len_rows):
                out.setdefault("partial_metrics", []).append(k)
    return out


def stratified_bootstrap(by_len_rows, key, iters=1000, seed=0):
    rng = random.Random(seed)
    strata = []
    for L in sorted(by_len_rows):
        vals = [r[key] for r in by_len_rows[L] if key in r]
        if vals:
            strata.append(vals)
    if not strata:
        return None
    stats = []
    for _ in range(iters):
        acc = 0.0
        for vals in strata:
            n = len(vals)
            acc += sum(vals[rng.randrange(n)] for _ in range(n)) / n
        stats.append(acc / len(strata))
    stats.sort()
    return [stats[int(0.025 * iters)], stats[int(0.975 * iters)]]


def score_spanonly(examples, preds, model_name, unit, extra=None):
    rows = score_rows(examples, preds)
    by_len = _group(rows, "len")
    by_bucket = _group(rows, "bucket")
    by_source = _group(rows, "source")
    results = {
        "model": model_name,
        "protocol": "spanonly",
        "unit": unit,
        "n_examples": len(rows),
        "lens": sorted(by_len),
        "headline_macro_over_lengths": _macro(by_len),
        "micro_pooled": _agg(rows),
        "by_len": {str(L): _agg(v) for L, v in sorted(by_len.items())},
        "by_bucket": {b: _agg(v) for b, v in sorted(by_bucket.items())},
        "by_source": {s: _agg(v) for s, v in sorted(by_source.items())},
    }
    cis = {}
    for key in ("em_top1_norm", "em_top20_norm", "cer_top1_norm",
                "em_top1", "em_top20", "cer_top1"):
        ci = stratified_bootstrap(by_len, key)
        if ci:
            cis[f"{key}_macro_ci95"] = ci
    results["headline_macro_over_lengths"].update(cis)
    if extra:
        results["diagnostics"] = extra
    return results


class NgramLM:

    def __init__(self, order=6, alpha=0.4):
        self.order = order
        self.alpha = alpha
        self.counts = [defaultdict(int) for _ in range(order + 1)]
        self.ctx = [defaultdict(int) for _ in range(order + 1)]
        self.cont = [defaultdict(set) for _ in range(order + 1)]

    def fit(self, texts):
        for t in texts:
            t = " " + t + " "
            for n in range(1, self.order + 1):
                for i in range(len(t) - n + 1):
                    g = t[i:i + n]
                    self.counts[n][g] += 1
                    self.ctx[n][g[:-1]] += 1
                    self.cont[n][g[:-1]].add(g[-1])
        self.total_uni = sum(self.counts[1].values())
        uni = sorted(self.counts[1].items(), key=lambda kv: -kv[1])
        self.top_uni = [c for c, _ in uni[:12] if c != MASK_CHAR]

    def candidates(self, context):
        cands = set()
        for n in range(self.order, 1, -1):
            c = context[-(n - 1):]
            if c in self.cont[n]:
                cands |= self.cont[n][c]
                if len(cands) >= 24:
                    break
        cands.update(self.top_uni)
        cands.discard(MASK_CHAR)
        return cands

    def logp(self, ch, context):
        for n in range(self.order, 0, -1):
            c = context[-(n - 1):] if n > 1 else ""
            g = c + ch
            if self.counts[n].get(g, 0) > 0:
                denom = self.ctx[n][c] if n > 1 else self.total_uni
                back = (self.order - n)
                return math.log(self.counts[n][g] / denom) + \
                    back * math.log(self.alpha)
        return math.log(1e-9)


def ngram_span_topk(lm, left, right, L, k=20, beam=20):
    beams = [(0.0, "")]
    for _ in range(L):
        nxt = []
        for score, seq in beams:
            ctx = (left + seq)[-(lm.order - 1):]
            for ch in lm.candidates(ctx):
                nxt.append((score + lm.logp(ch, ctx), seq + ch))
        nxt.sort(key=lambda x: -x[0])
        beams = nxt[:max(beam, k)]
    rescored = []
    for score, seq in beams:
        s = score
        ctx = left + seq
        for j, ch in enumerate(right[:lm.order - 1]):
            s += lm.logp(ch, (ctx + right[:j])[-(lm.order - 1):])
        rescored.append((s, seq))
    rescored.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for _, seq in rescored:
        if seq not in seen:
            seen.add(seq)
            out.append(seq)
        if len(out) == k:
            break
    return out


def unigram_topk(char_logps, L, k=20):
    ranked = sorted(char_logps.items(), key=lambda kv: -kv[1])[:max(k, 20)]
    beams = [(0.0, "")]
    for _ in range(L):
        nxt = [(sc + lp, seq + ch) for sc, seq in beams for ch, lp in ranked]
        nxt.sort(key=lambda x: -x[0])
        beams = nxt[:k]
    return [seq for _, seq in beams]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--window_chars", type=int, default=384)
    ap.add_argument("--per_len", type=int, default=500)
    ap.add_argument("--min_len", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=18)
    ap.add_argument("--min_context", type=int, default=8)
    ap.add_argument("--seed", type=int, default=45)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    recs = load_split(args.data_dir, args.split)
    wins = window_inventory(recs, args.window_chars)
    lens = list(range(args.min_len, args.max_len + 1))
    examples = build_spanonly_from_windows(
        wins, args.seed, args.per_len, lens, args.min_context)
    out = args.out or os.path.join(args.data_dir,
                                   f"spanonly_{args.split}.jsonl")
    save_examples(examples, out)
    print(f"{len(wins)} windows -> {len(examples)} span-only examples "
          f"(lengths {lens[0]}-{lens[-1]}, {args.per_len}/length) -> {out}")


if __name__ == "__main__":
    main()
