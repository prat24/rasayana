import argparse
import json
import os
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          LogitsProcessor, LogitsProcessorList)

from byt5_data import (SEP, GAP_CHAR, WINDOW_IAST_CHARS, to_iast,
                       to_deva, make_pair, chunk_text)
from common import load_split
import protocol as P


def iast_window_inventory(records, window=WINDOW_IAST_CHARS):
    wins = []
    for r in records:
        for w in chunk_text(to_iast(r["text"]), max_chars=window):
            wins.append((r["source"], w))
    return wins


def build_spanonly(data_dir, split, seed, per_len, max_span, window,
                   min_context, out_path):
    recs = load_split(data_dir, split)
    wins = iast_window_inventory(recs, window)
    lens = list(range(1, max_span + 1))
    examples = P.build_spanonly_from_windows(wins, seed, per_len, lens,
                                             min_context)
    for ex in examples:
        src, tgt, truth = make_pair(ex["text"], ex["spans"])
        ex["src"], ex["tgt"], ex["truth"], ex["iast"] = (src, tgt, truth,
                                                         ex["text"])
    P.save_examples(examples, out_path)
    print(f"{len(wins)} IAST windows -> {len(examples)} span-only examples "
          f"(lengths 1-{max_span}, {per_len}/length) -> {out_path}")
    return examples


def load_byt5(model_dir, base_model, device):
    try:
        tok = AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        print(f"no tokenizer in {model_dir}; using {base_model}'s "
              f"(identical byte vocab)")
        tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tok


def gen_kwargs(topk, max_new):
    return dict(num_beams=topk, num_return_sequences=topk,
                max_new_tokens=max_new, early_stopping=True)


def dedup(seq_list):
    seen, out = set(), []
    for s in seq_list:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def fill_banned_ids(tok):
    ids = set()
    for ch in (SEP, GAP_CHAR):
        ids.update(tok(ch, add_special_tokens=False).input_ids)
    return sorted(ids)


class ExactCharLengthProcessor(LogitsProcessor):

    def __init__(self, target_lens, num_beams, tokenizer, eos_id, banned_ids):
        self.target_lens = target_lens
        self.num_beams = num_beams
        self.tok = tokenizer
        self.eos_id = eos_id
        self.banned_ids = banned_ids

    def __call__(self, input_ids, scores):
        texts = self.tok.batch_decode(input_ids, skip_special_tokens=True)
        neg = torch.finfo(scores.dtype).min
        for row, text in enumerate(texts):
            if len(text) < self.target_lens[row // self.num_beams]:
                scores[row, self.eos_id] = neg
                if self.banned_ids:
                    scores[row, self.banned_ids] = neg
            else:
                keep = scores[row, self.eos_id].clone()
                scores[row].fill_(neg)
                scores[row, self.eos_id] = keep
        return scores


@torch.no_grad()
def generate_batches(model, tok, sources, device, batch_size, topk, kwargs,
                     max_length=1024, target_lens=None):
    outs = [None] * len(sources)
    forced = target_lens is not None
    if forced:
        eos_id = tok.eos_token_id
        banned = fill_banned_ids(tok)
    t0 = time.time()
    for c0 in range(0, len(sources), batch_size):
        chunk = sources[c0:c0 + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(device)
        gk = kwargs
        if forced:
            tl = target_lens[c0:c0 + batch_size]
            gk = dict(kwargs, logits_processor=LogitsProcessorList(
                [ExactCharLengthProcessor(tl, topk, tok, eos_id, banned)]))
            gk["max_new_tokens"] = 3 * max(tl) + 8
        gen = model.generate(**enc, **gk)
        seqs = tok.batch_decode(gen, skip_special_tokens=True)
        for j in range(len(chunk)):
            outs[c0 + j] = seqs[j * topk:(j + 1) * topk]
        if (c0 // batch_size) % 10 == 0:
            done = c0 + len(chunk)
            print(f"  generated {done}/{len(sources)} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    return outs


def run_spanonly(model, tok, examples, device, args, target_lens=None):
    sources = []
    for ex in examples:
        src, _tgt, _truth = make_pair(ex["text"], ex["spans"])
        sources.append(src)
    kwargs = gen_kwargs(args.topk, args.max_new_spanonly)
    raw = generate_batches(model, tok, sources, device, args.batch_size,
                           args.topk, kwargs, args.max_length, target_lens)
    preds = []
    for cands in raw:
        fills = dedup([c.split(SEP)[0] for c in cands])
        preds.append({"topk": fills})
    return preds


def write_samples(path, examples, preds, n=40):
    with open(path, "w", encoding="utf-8") as fh:
        step = max(1, len(examples) // n)
        for ex, pr in list(zip(examples, preds))[::step][:n]:
            s, e = ex["spans"][0]
            fh.write(f"len={ex['len']}  source={ex['source']}\n")
            fh.write("SRC (IAST) : "
                     + make_pair(ex["text"], ex["spans"])[0] + "\n")
            fh.write("TRUTH      : " + ex["text"][s:e] + "\n")
            fh.write("TOP-5      : " + " | ".join(pr["topk"][:5]) + "\n")
            fh.write("TRUTH deva : " + to_deva(ex["text"][s:e]) + "\n")
            fh.write("TOP1 deva  : " + to_deva(pr["topk"][0]) + "\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_spanonly", action="store_true",
                    help="only build the span-only IAST eval file and exit")
    ap.add_argument("--model_dir", default="runs_byt5/final")
    ap.add_argument("--base_model", default="buddhist-nlp/byt5-sanskrit")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results2/byt5")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max_length", type=int, default=1024,
                    help="source truncation; matches training max_length")
    ap.add_argument("--per_len", type=int, default=250,
                    help="span-only examples per gap length (build + eval)")
    ap.add_argument("--max_span", type=int, default=23)
    ap.add_argument("--min_context", type=int, default=8)
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--batch_size", type=int, default=0, help="0 = auto")
    ap.add_argument("--max_new_spanonly", type=int, default=48)
    ap.add_argument("--no_force_length", action="store_true",
                    help="span-only: disable exact-length decoding (fills may "
                         "deviate from the known gap length L; default forces "
                         "len(fill) == L, so len_match reflects content only)")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--name", default=None,
                    help="model name in results.json")
    args = ap.parse_args()

    span_path = os.path.join(args.data_dir,
                             f"byt5_spanonly_{args.split}.jsonl")
    if args.build_spanonly:
        build_spanonly(args.data_dir, args.split, args.seed, args.per_len,
                       args.max_span, WINDOW_IAST_CHARS, args.min_context,
                       span_path)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_byt5(args.model_dir, args.base_model, device)
    os.makedirs(args.out, exist_ok=True)
    name = args.name or "byt5-sanskrit-finetuned-beam"
    t0 = time.time()

    if not os.path.exists(span_path):
        build_spanonly(args.data_dir, args.split, args.seed,
                       args.per_len, args.max_span, WINDOW_IAST_CHARS,
                       args.min_context, span_path)
    examples = P.load_examples(span_path)
    if args.per_len:
        per = Counter()
        kept = []
        for ex in examples:
            per[ex["len"]] += 1
            if per[ex["len"]] <= args.per_len:
                kept.append(ex)
        examples = kept
    args.batch_size = args.batch_size or 4
    target_lens = (None if args.no_force_length
                   else [ex["len"] for ex in examples])
    preds = run_spanonly(model, tok, examples, device, args, target_lens)
    results = P.score_spanonly(
        examples, preds, name, unit="iast_chars",
        extra={"decoding": ("beam" if args.no_force_length
                            else "beam+length_forced"),
               "runtime_s": round(time.time() - t0, 1)})
    with open(os.path.join(args.out, "results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    write_samples(os.path.join(args.out, "samples.txt"),
                  examples, preds, n=args.samples)
    h = results["headline_macro_over_lengths"]
    print(f"\n=== {name} | span-only ({results['n_examples']} ex, "
          f"macro over lengths 1-{args.max_span}, IAST units) ===")
    for k in ("em_top1_norm", "em_top5_norm", "em_top20_norm",
              "cer_top1_norm",
              "len_match", "n_cands"):
        if k in h:
            nk = h.get(f"n_{k}")
            note = ("" if nk in (None, results["n_examples"])
                    else f"   (n={nk})")
            print(f"  {k:_<22} {h[k]:.4f}{note}")
    print(f"-> {args.out}/results.json  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
