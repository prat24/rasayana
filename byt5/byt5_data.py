import json
import random
import unicodedata

try:
    from torch.utils.data import Dataset
except ImportError:
    Dataset = object

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except ImportError:
    sanscript = transliterate = None

CHUNK_MAX = 512
CHUNK_MIN = 128
CHUNK_OVERLAP = 64
WINDOW_IAST_CHARS = CHUNK_MAX
GAP_CHAR = "_"
SEP = ";"
PREFIX = "R "


def to_iast(deva: str) -> str:
    return unicodedata.normalize(
        "NFC", transliterate(deva, sanscript.DEVANAGARI, sanscript.IAST))


def to_deva(iast: str) -> str:
    return transliterate(iast, sanscript.IAST, sanscript.DEVANAGARI)


def chunk_text(text, max_chars=CHUNK_MAX, min_chars=CHUNK_MIN,
               overlap=CHUNK_OVERLAP):
    text = text.strip()
    n = len(text)
    if n < min_chars:
        return []
    if n <= max_chars:
        return [text]
    stride = max(1, max_chars - overlap)
    chunks, off = [], 0
    while off < n:
        w = text[off:off + max_chars].strip()
        if len(w) >= min_chars:
            chunks.append(w)
        if off + max_chars >= n:
            break
        off += stride
    return chunks


MAX_FILL = 23
DAMAGE = dict(min_ratio=0.0, max_ratio=0.75, geo_p=0.2, max_fill=MAX_FILL)


def _fill_len(rng, lmax):
    if rng.random() < 0.5:
        L = 1
        while rng.random() > DAMAGE["geo_p"] and L < lmax:
            L += 1
        return L
    return rng.randint(1, lmax)


def sample_spans(n, rng, damage="tasklen", min_ratio=DAMAGE["min_ratio"],
                 max_ratio=DAMAGE["max_ratio"], max_fill=MAX_FILL, edge=8,
                 sep=1):
    """Rate-budget span sampling, identical to the transformer recipe
    (common.sample_damage_tasklen): draw a sequence-level mask rate
    r ~ U(min_ratio, max_ratio), then spend the whole budget floor(r*n)
    on continuous spans whose length follows the 50/50 truncated-geometric
    /uniform mixture (field mode forces L=1)."""
    if n < 16:
        return []
    total = int(rng.uniform(min_ratio, max_ratio) * n)
    if total <= 0:
        return []
    lmax = min(max_fill, n - 2 * edge)
    if lmax < 1:
        return []
    spans, remaining, misses = [], total, 0
    while remaining > 0 and misses < 200:
        L = min(1 if damage == "field" else _fill_len(rng, lmax), remaining)
        s = rng.randrange(edge, n - edge - L + 1)
        e = s + L
        if all(s >= e0 + sep or e + sep <= s0 for s0, e0 in spans):
            spans.append((s, e))
            remaining -= L
            misses = 0
        else:
            misses += 1
    spans.sort()
    return spans


def make_pair(iast_text, spans):
    src = list(iast_text)
    for s, e in spans:
        for i in range(s, e):
            src[i] = GAP_CHAR
    truth = [iast_text[s:e] for s, e in spans]
    return PREFIX + "".join(src), SEP.join(truth), truth


class ByT5TrainDataset(Dataset):

    def __init__(self, records, tokenizer, size, seed,
                 max_length=1024, damage="field"):
        self.tok = tokenizer
        self.size = size
        self.seed = seed
        self.max_length = max_length
        self.damage = damage
        self.chunks = []
        for r in records:
            self.chunks.extend(chunk_text(to_iast(r["text"])))

    def __len__(self):
        return self.size

    def _encode(self, src, tgt):
        model_in = self.tok(src, max_length=self.max_length, truncation=True)
        model_in["labels"] = self.tok(text_target=tgt,
                                      max_length=self.max_length,
                                      truncation=True)["input_ids"]
        return model_in

    def __getitem__(self, idx):
        rng = random.Random(self.seed * 10**12 + idx)
        for _ in range(8):
            w = self.chunks[rng.randrange(len(self.chunks))]
            spans = sample_spans(len(w), rng, self.damage)
            if spans:
                src, tgt, _ = make_pair(w, spans)
                return self._encode(src, tgt)
        src, tgt, _ = make_pair(w, [(0, min(4, len(w)))])
        return self._encode(src, tgt)


class ByT5EvalDataset(Dataset):

    def __init__(self, examples, tokenizer, max_length=512):
        self.ex = examples
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, idx):
        e = self.ex[idx]
        enc = self.tok(e["src"], max_length=self.max_length, truncation=True)
        enc["labels"] = self.tok(text_target=e["tgt"],
                                 max_length=self.max_length,
                                 truncation=True)["input_ids"]
        return enc


def load_eval(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]
