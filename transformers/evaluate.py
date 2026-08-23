import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import get_tokenizer, load_split, apply_spans, MASK_CHAR
from data import encode_example
from model import ModelConfig, RestorationModel
import protocol as P


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    kind = ck["args"]["tokenizer"]
    tok = get_tokenizer(kind, itos=ck["vocab"])
    model = RestorationModel(ModelConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, tok, kind, ck["args"]["max_len"], ck.get("step", -1)


def banned_ids(tok):
    ids = [tok.pad_id, tok.mask_id]
    if tok.unk_id is not None:
        ids.append(tok.unk_id)
    return ids


def cp_prefix(text):
    pre = [0]
    for ch in text:
        pre.append(pre[-1] + len(ch.encode("utf-8")))
    return pre


def gap_positions(text, spans, kind):
    """Token-position list per lacuna, from the damage spans."""
    if kind == "char":
        return [list(range(s, e)) for s, e in spans]
    pre = cp_prefix(text)
    return [list(range(pre[s], pre[e])) for s, e in spans]


def mask_groups(text, spans, kind):
    groups = []
    if kind == "char":
        for s, e in spans:
            groups += [[i] for i in range(s, e)]
    else:
        pre = cp_prefix(text)
        for s, e in spans:
            for i in range(s, e):
                groups.append(list(range(pre[i], pre[i + 1])))
    return groups


def collate_examples(exs, tok, kind, max_len, device):
    rows = [encode_example(kind, tok, ex["text"], ex["spans"], max_len)
            for ex in exs]
    B = len(rows)
    L = max(len(r[0]) for r in rows)
    inp = torch.full((B, L), tok.pad_id, dtype=torch.long)
    km = torch.zeros((B, L), dtype=torch.bool)
    for i, (iids, _tgt, _m, _slot) in enumerate(rows):
        n = len(iids)
        inp[i, :n] = torch.tensor(iids, dtype=torch.long)
        km[i, :n] = True
    return inp.to(device), km.to(device)


def make_fwd(model, device, amp_dtype):
    def fwd(ids, km):
        with torch.autocast(device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            return model(ids, km)
    return fwd


def masked_logp(logits, banned):
    lp = F.log_softmax(logits.float(), dim=-1)
    lp[..., banned] = float("-inf")
    return lp


def legal_cands(tok, kind, device):
    if kind == "byte":
        return {n: torch.tensor(v, dtype=torch.long, device=device)
                for n, v in tok.cand_bytes.items()}
    ban = set(banned_ids(tok))
    ids = [[i] for i in range(tok.vocab_size) if i not in ban]
    return {1: torch.tensor(ids, dtype=torch.long, device=device)}


@torch.no_grad()
def seq_beam(fwd, inp, km, pos, sizes, banned, cands, W=20):
    B, L = inp.shape
    dev = inp.device
    ids = inp.unsqueeze(1).repeat(1, W, 1)
    scores = torch.full((B, W), float("-inf"), device=dev)
    scores[:, 0] = 0.0
    bidx = torch.arange(B, device=dev)
    kmf = km.unsqueeze(1).expand(B, W, L).reshape(B * W, L)
    off = 0
    for s in sizes:
        logits = fwd(ids.reshape(B * W, L), kmf)
        gp = pos[:, off:off + s]
        gpf = gp.unsqueeze(1).expand(B, W, s).reshape(B * W, s)
        lp = masked_logp(
            logits.gather(1, gpf.unsqueeze(-1).expand(
                -1, -1, logits.shape[-1])), banned)
        cd = cands.get(s)
        if cd is None:
            plp, pid = lp.topk(W, dim=-1)
            jlp = plp[:, 0]
            jid = pid[:, 0].unsqueeze(-1)
            for t in range(1, s):
                comb = (jlp.unsqueeze(2)
                        + plp[:, t].unsqueeze(1)).reshape(B * W, W * W)
                jlp, ci = comb.topk(W, dim=-1)
                prev, nxt = ci // W, ci % W
                jid = torch.cat(
                    [jid.gather(1, prev.unsqueeze(-1).expand(-1, -1, t)),
                     pid[:, t].gather(1, nxt).unsqueeze(-1)], dim=-1)
            K = jlp.shape[1]
        else:
            cs = None
            for t in range(s):
                x = lp[:, t].index_select(1, cd[:, t])
                cs = x if cs is None else cs + x
            K = min(W, cd.shape[0])
            jlp, ci = cs.topk(K, dim=-1)
            jid = cd.index_select(0, ci.reshape(-1)).reshape(B * W, K, s)
        cand = (scores.reshape(B * W, 1) + jlp).reshape(B, W * K)
        scores, flat_i = cand.topk(W, dim=-1)
        parent, choice = flat_i // K, flat_i % K
        tok = jid.reshape(B, W, K, s)[bidx[:, None], parent, choice]
        ids = ids[bidx[:, None], parent]
        ids.scatter_(2, gp.unsqueeze(1).expand(B, W, s), tok)
        off += s
    return ids, scores


def dedup(seq_list):
    seen, out = set(), []
    for s in seq_list:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


@torch.no_grad()
def run_spanonly_model(model, tok, kind, examples, device, amp_dtype,
                       topk=20, seq_chunk=8):
    fwd = make_fwd(model, device, amp_dtype)
    banned = banned_ids(tok)
    cands = legal_cands(tok, kind, device)
    preds = [None] * len(examples)

    by_sig = defaultdict(list)
    for i, ex in enumerate(examples):
        sig = tuple(len(g) for g in
                    mask_groups(ex["text"], ex["spans"], kind))
        by_sig[sig].append(i)

    t0 = time.time()
    done = 0
    for sig, seq_idx in sorted(by_sig.items()):
        for c0 in range(0, len(seq_idx), seq_chunk):
            chunk = seq_idx[c0:c0 + seq_chunk]
            exs = [examples[i] for i in chunk]
            inp, km = collate_examples(exs, tok, kind, 10 ** 9, device)
            assert inp.shape[1] <= model.cfg.max_len, (
                f"window is {inp.shape[1]} tokens but the checkpoint's "
                f"max_len is {model.cfg.max_len}")
            pos = torch.tensor(
                [gap_positions(ex["text"], ex["spans"], kind)[0]
                 for ex in exs], dtype=torch.long, device=device)
            ids, scores = seq_beam(fwd, inp, km, pos, sig, banned,
                                   cands, W=topk)
            ids_cpu = ids.cpu()
            sc_cpu = scores.cpu()
            for b, i in enumerate(chunk):
                plist = pos[b].tolist()
                fills = [tok.decode([ids_cpu[b, w, p].item()
                                     for p in plist])
                         for w in range(topk)
                         if math.isfinite(sc_cpu[b, w].item())]
                preds[i] = {"topk": dedup(fills)}
        done += len(seq_idx)
        print(f"  seq-beam {done}/{len(examples)} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    return preds


def spanonly_diagnostics(examples, preds, kind, step, n_params,
                         runtime_s, tok=None):
    diag = {"model_step": step, "n_params": n_params,
            "runtime_s": round(runtime_s, 1)}
    known = getattr(tok, "known", None) or None
    tops = [pr["topk"][0] for pr in preds if pr.get("topk")]
    if kind == "byte":
        diag["utf8_valid_seq_top1"] = (
            sum("�" not in f for f in tops) / max(1, len(tops)))
    if known:
        diag["in_language_seq_top1"] = (
            sum(all(c in known for c in f) for f in tops) / max(1, len(tops)))
    return diag


def run_spanonly_baseline(kind, examples, train_texts, topk=20):
    preds = []
    if kind == "ngram":
        lm = P.NgramLM(order=6)
        lm.fit(train_texts)
        for ex in examples:
            s, e = ex["spans"][0]
            left = ex["text"][max(0, s - (lm.order - 1)):s]
            right = ex["text"][e:e + lm.order - 1]
            cands = P.ngram_span_topk(lm, left, right, e - s, k=topk)
            preds.append({"topk": cands})
    else:
        cnt = Counter("".join(train_texts))
        cnt.pop(MASK_CHAR, None)
        total = sum(cnt.values())
        logps = {c: math.log(n / total) for c, n in cnt.items()}
        for ex in examples:
            s, e = ex["spans"][0]
            cands = P.unigram_topk(logps, e - s, k=topk)
            preds.append({"topk": cands})
    return preds


def write_spanonly_samples(path, examples, preds, n=40):
    with open(path, "w", encoding="utf-8") as fh:
        step = max(1, len(examples) // n)
        for ex, pr in list(zip(examples, preds))[::step][:n]:
            s, e = ex["spans"][0]
            fh.write(f"len={ex['len']}  source={ex['source']}\n")
            fh.write("DAMAGED : " + apply_spans(ex["text"], ex["spans"])
                     + "\n")
            fh.write("TRUTH   : " + ex["text"][s:e] + "\n")
            if pr.get("topk"):
                fh.write("SEQ-BEAM: " + " | ".join(pr["topk"][:5]) + "\n")
            fh.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--baseline", choices=("ngram", "unigram"), default=None)
    ap.add_argument("--eval_file", default=None,
                    help="span-only protocol file "
                         "(default data/spanonly_<split>.jsonl)")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results2/run")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap examples per gap length")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--seq_chunk", type=int, default=0, help="0 = auto")
    ap.add_argument("--samples", type=int, default=40)
    args = ap.parse_args()
    assert args.ckpt or args.baseline, "need --ckpt or --baseline"
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    path = args.eval_file or os.path.join(
        args.data_dir, f"spanonly_{args.split}.jsonl")
    examples = P.load_examples(path)
    if args.limit:
        per = Counter()
        kept = []
        for ex in examples:
            per[ex["len"]] += 1
            if per[ex["len"]] <= args.limit:
                kept.append(ex)
        examples = kept

    if args.baseline:
        name = args.baseline
        train_texts = [r["text"] for r in load_split(args.data_dir, "train")]
        preds = run_spanonly_baseline(args.baseline, examples,
                                      train_texts, topk=args.topk)
        results = P.score_spanonly(
            examples, preds, name, unit="devanagari_codepoints",
            extra={"runtime_s": round(time.time() - t0, 1)})
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        amp_dtype = torch.float16 if device.type == "cuda" else None
        model, tok, kind, _max_len, step = load_model(args.ckpt, device)
        name = f"{kind}-transformer"
        sc = args.seq_chunk or (12 if kind == "char" else 4)
        preds = run_spanonly_model(
            model, tok, kind, examples, device, amp_dtype,
            topk=args.topk, seq_chunk=sc)
        diag = spanonly_diagnostics(
            examples, preds, kind, step,
            model.num_params(), time.time() - t0, tok)
        results = P.score_spanonly(examples, preds, name,
                                   unit="devanagari_codepoints",
                                   extra=diag)

    with open(os.path.join(args.out, "results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    write_spanonly_samples(os.path.join(args.out, "samples.txt"),
                           examples, preds, n=args.samples)
    h = results["headline_macro_over_lengths"]
    print(f"\n=== {name} | span-only protocol "
          f"({results['n_examples']} examples, macro over lengths "
          f"{results['lens'][0]}-{results['lens'][-1]}) ===")
    for k in ("em_top1_norm", "em_top5_norm", "em_top20_norm",
              "cer_top1_norm"):
        if k in h:
            nk = h.get(f"n_{k}")
            note = ("" if nk in (None, results["n_examples"])
                    else f"   (n={nk})")
            print(f"  {k:_<22} {h[k]:.4f}{note}")
    print(f"-> {args.out}/results.json  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
