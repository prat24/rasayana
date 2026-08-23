import bisect
import random

import torch
from torch.utils.data import Dataset

from common import N_SLOT_KEYS, damage_fn, slot_key, tidy_window


def cp_prefix(text):
    prefix = [0]
    for ch in text:
        prefix.append(prefix[-1] + len(ch.encode("utf-8")))
    return prefix


def encode_example(kind, tok, text, char_spans, max_len):
    tgt = tok.encode(text)
    inp = list(tgt)
    mask = [0] * len(tgt)
    slot = [0] * len(tgt)
    if kind == "char":
        for s, e in char_spans:
            for i in range(s, min(e, len(tgt))):
                inp[i] = tok.mask_id
                if tgt[i] != tok.unk_id:
                    mask[i] = 1
        cut = max_len
    else:
        pre = cp_prefix(text)
        for s, e in char_spans:
            for ci in range(s, min(e, len(text))):
                b0, n = pre[ci], pre[ci + 1] - pre[ci]
                supervise = (not tok.known) or (text[ci] in tok.known)
                for j in range(n):
                    i = b0 + j
                    if i >= len(tgt):
                        break
                    inp[i] = tok.mask_id
                    slot[i] = slot_key(n, j)
                    if supervise:
                        mask[i] = 1
        cut = (pre[bisect.bisect_right(pre, max_len) - 1]
               if len(tgt) > max_len else max_len)
    return inp[:cut], tgt[:cut], mask[:cut], slot[:cut]


def legal_matrix(tok, kind, device=None):
    if kind != "byte" or not getattr(tok, "constrained", False):
        return None
    m = torch.zeros(N_SLOT_KEYS, tok.vocab_size, dtype=torch.bool)
    m[0, :256] = True
    for k, vals in tok.slot_legal.items():
        m[k, vals] = True
    return m if device is None else m.to(device)


class RestorationTrainDataset(Dataset):

    def __init__(self, records, tok, kind, window_chars, max_len,
                 size, seed, min_ratio=0.05, max_ratio=0.5,
                 damage="field"):
        self.records = records
        self.tok = tok
        self.kind = kind
        self.damage = damage
        self.damage_fn = damage_fn(damage)
        self.window_chars = window_chars
        self.max_len = max_len
        self.size = size
        self.seed = seed
        self.epoch = 0
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.cum = []
        total = 0
        for r in records:
            total += len(r["text"])
            self.cum.append(total)
        self.total_chars = total

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        rng = random.Random(self.seed * 10**12 + self.epoch * 10**7 + idx)
        for _ in range(8):
            pos = rng.randrange(self.total_chars)
            ri = bisect.bisect_right(self.cum, pos)
            text = self.records[ri]["text"]
            target = (rng.randint(25, self.window_chars)
                      if self.window_chars > 25 else self.window_chars)
            if len(text) <= target:
                window = text
            else:
                off = rng.randrange(len(text) - target + 1)
                window = text[off:off + target]
            window = tidy_window(window)
            if len(window) < 25:
                continue
            damaged, spans, _rate = self.damage_fn(
                window, rng, self.min_ratio, self.max_ratio)
            if spans:
                inp, tgt, mask, slot = encode_example(
                    self.kind, self.tok, window, spans, self.max_len)
                if sum(mask) > 0:
                    return {"input": inp, "target": tgt, "loss_mask": mask,
                            "slot": slot}
        inp, tgt, mask, slot = encode_example(self.kind, self.tok, window,
                                              [(0, min(4, len(window)))],
                                              self.max_len)
        return {"input": inp, "target": tgt, "loss_mask": mask, "slot": slot}


class RestorationEvalDataset(Dataset):

    def __init__(self, examples, tok, kind, max_len):
        self.examples = examples
        self.tok = tok
        self.kind = kind
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        inp, tgt, mask, slot = encode_example(self.kind, self.tok, ex["text"],
                                              ex["spans"], self.max_len)
        return {"input": inp, "target": tgt, "loss_mask": mask, "slot": slot,
                "idx": idx}


def collate(batch, pad_id):
    L = max(len(b["input"]) for b in batch)
    B = len(batch)
    inp = torch.full((B, L), pad_id, dtype=torch.long)
    tgt = torch.full((B, L), pad_id, dtype=torch.long)
    lm = torch.zeros((B, L), dtype=torch.bool)
    km = torch.zeros((B, L), dtype=torch.bool)
    slot = torch.zeros((B, L), dtype=torch.long)
    idxs = []
    for i, b in enumerate(batch):
        n = len(b["input"])
        inp[i, :n] = torch.tensor(b["input"], dtype=torch.long)
        tgt[i, :n] = torch.tensor(b["target"], dtype=torch.long)
        lm[i, :n] = torch.tensor(b["loss_mask"], dtype=torch.bool)
        slot[i, :n] = torch.tensor(b["slot"], dtype=torch.long)
        km[i, :n] = True
        idxs.append(b.get("idx", -1))
    return {"input": inp, "target": tgt, "loss_mask": lm, "key_mask": km,
            "slot": slot, "idx": torch.tensor(idxs, dtype=torch.long)}


def make_collate(pad_id):
    return lambda batch: collate(batch, pad_id)
