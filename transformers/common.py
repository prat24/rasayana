import glob
import hashlib
import json
import os
import random
import re
import unicodedata
from collections import Counter


MASK_CHAR = "◌"
PAD, UNK, MASK = "<pad>", "<unk>", MASK_CHAR

NUKTA_MAP = {
    "क़": "क",
    "ख़": "ख",
    "ग़": "ग",
    "ज़": "ज",
    "ड़": "ड",
    "ढ़": "ढ",
    "फ़": "फ",
    "य़": "य",
    "ऱ": "र",
}

_ALLOWED_RANGES = (
    ("ँ", "ः"),
    ("अ", "औ"),
    ("क", "ह"),
    ("ऽ", "ऽ"),
    ("ा", "ौ"),
    ("्", "्"),
    ("ॐ", "ॐ"),
    ("ॠ", "ॣ"),
)


def _is_allowed(ch: str) -> bool:
    if ch == " ":
        return True
    return any(lo <= ch <= hi for lo, hi in _ALLOWED_RANGES)


_DEVA_DIGITS = "०-९"


def normalize_line(line: str) -> str:
    t = unicodedata.normalize("NFC", line)
    for src, dst in NUKTA_MAP.items():
        t = t.replace(src, dst)
    t = t.replace("़", "")
    t = t.replace("‌", "").replace("‍", "")
    t = re.sub(r"\([^()]*\)", " ", t)
    t = re.sub(r"\[[^\[\]]*\]", " ", t)
    t = re.sub(r"-+", " ", t)
    t = re.sub(r"\.+", " ", t)
    t = re.sub(rf"[{_DEVA_DIGITS}0-9]+", "", t)
    t = "".join(ch if _is_allowed(ch) else " " for ch in t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


_HEADING_RE = re.compile(r"^\s*(अध्यायः|अध्याय:|स्थानम्)(\s|$)")


def is_heading(raw_line: str) -> bool:
    return bool(_HEADING_RE.match(raw_line.strip()))


def devanagari_ratio(t: str) -> float:
    if not t:
        return 0.0
    core = [c for c in t if c != " "]
    if not core:
        return 0.0
    dev = sum(1 for c in core if "ऀ" <= c <= "ॿ")
    return dev / len(core)


MIN_BLOCK_CHARS = 128


def load_blocks(raw_dir: str):
    blocks = []
    files = sorted(glob.glob(os.path.join(raw_dir, "*_raw_formatted.txt")))
    if not files:
        raise FileNotFoundError(f"no *_raw_formatted.txt under {raw_dir}")
    for path in files:
        source = os.path.basename(path).replace("_raw_formatted.txt", "")
        cur = {"source": source, "chapter_idx": 0, "lines": []}
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                if is_heading(raw):
                    if cur["lines"]:
                        blocks.append(cur)
                    cur = {"source": source,
                           "chapter_idx": cur["chapter_idx"] + 1,
                           "lines": []}
                    continue
                t = normalize_line(raw)
                if len(t) < 4 or devanagari_ratio(t) < 0.6:
                    continue
                cur["lines"].append(t)
        if cur["lines"]:
            blocks.append(cur)
    return [b for b in blocks
            if sum(len(l) for l in b["lines"]) >= MIN_BLOCK_CHARS]


GRETIL_EXCLUDE = ("bower", "aSTAGgahRdayasUtra",
                  "rasAdhyaya1-302-com", "rasahRdayatantra-comm",
                  "rasendracintAmaNi")

_GRETIL_COMMENTARY_TAG = re.compile(r"cm(_|\d|$)")

_GRETIL_REF_CLOSED = re.compile(r"//\s*[^/\s]*[\d_][^/]*?//")
_GRETIL_REF_OPEN = re.compile(r"//\s*\S*[\d_]\S*\s*$")
_GRETIL_ANNOT = re.compile(r"<[^<>]*>|\{[^{}]*\}")
_GRETIL_SKIP = re.compile(
    r"^\s*(start|end)\b|^\*|^[\sA-Za-zĀ-ſ]*,[\s\d,.\-]*$")


def gretil_clean_line(line: str) -> str:
    t = unicodedata.normalize("NFC", line.strip())
    if not t or _GRETIL_SKIP.match(t):
        return ""
    t = _GRETIL_ANNOT.sub(" ", t)
    t = re.sub(r"\([^()]*\)|\[[^\[\]]*\]", " ", t)
    t = _GRETIL_REF_CLOSED.sub("॥", t)
    t = _GRETIL_REF_OPEN.sub("॥", t)
    t = t.replace("//", "॥").replace("/", "।")
    t = t.replace("-", "")
    t = re.sub(r"[*+^_=\[\]()]", " ", t)
    if re.search(r"\.\.|\?", t):
        return ""
    return re.sub(r"\s+", " ", t).strip()


def _gretil_chapter_key(line: str) -> str:
    """Chapter id from a locator ref, e.g. '// āk_1,2.91 //' -> 'āk_1,2'."""
    m = re.search(r"//\s*([^/\s]*[_][^/\s]*)", line)
    if not m:
        return ""
    tag = m.group(1)
    if re.search(r"[.:]", tag):
        tag = re.sub(r"[.:][\w.:,\-]*$", "", tag)
    else:
        tag = re.sub(r"_\d[\d,\-]*$", "", tag)
    return tag


def load_gretil_blocks(raw_dir: str):
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

    files = sorted(glob.glob(os.path.join(raw_dir, "sa_*.txt")))
    files = [f for f in files
             if not any(x in os.path.basename(f) for x in GRETIL_EXCLUDE)]
    if not files:
        raise FileNotFoundError(f"no usable sa_*.txt under {raw_dir}")
    blocks = []
    for path in files:
        source = "gretil_" + re.sub(
            r"^sa_|\.txt$", "", os.path.basename(path)).lower()
        raw = open(path, encoding="utf-8").read()
        body = raw.split("# Text", 1)[1] if "# Text" in raw else raw
        def to_deva(iast):
            deva = unicodedata.normalize(
                "NFC", transliterate(iast, sanscript.IAST,
                                     sanscript.DEVANAGARI))
            t = normalize_line(deva)
            return t if len(t) >= 4 and devanagari_ratio(t) >= 0.6 else ""

        cur = {"source": source, "chapter_idx": 0, "lines": [], "tag": ""}
        cur_key, pending = None, []
        for raw_line in body.split("\n"):
            t = gretil_clean_line(raw_line)
            if t:
                t = to_deva(t)
                if t:
                    pending.append(t)
            key = _gretil_chapter_key(raw_line)
            if key:
                if key != cur_key:
                    if cur["lines"]:
                        blocks.append(cur)
                        cur = {"source": source,
                               "chapter_idx": cur["chapter_idx"] + 1,
                               "lines": [], "tag": ""}
                    cur_key = key
                    cur["tag"] = key
                cur["lines"] += pending
                pending = []
        cur["lines"] += pending
        if cur["lines"]:
            blocks.append(cur)
    blocks = [b for b in blocks
              if not _GRETIL_COMMENTARY_TAG.search(b.pop("tag", "") or "")]
    return [b for b in blocks
            if sum(len(l) for l in b["lines"]) >= MIN_BLOCK_CHARS]


def _dedup_key(line: str) -> str:
    core = re.sub(r"[^ऀ-ॿ]", "", line)
    core = core.replace("ऽ", "")
    core = re.sub(r"[ङञणनम]्(?=[क-ह])", "ं", core)
    return hashlib.md5(core.encode("utf-8")).hexdigest() if len(core) >= 20 else ""


def build_splits(raw_dir: str, out_dir: str, seed: int = 42,
                 ratios=(0.90, 0.05, 0.05), include_gretil: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    blocks = load_blocks(raw_dir)
    rng = random.Random(seed)
    order = list(range(len(blocks)))
    rng.shuffle(order)

    total = sum(sum(len(l) for l in b["lines"]) for b in blocks)
    targets = [ratios[0] * total, ratios[1] * total, ratios[2] * total]
    split_of = {}
    filled = [0.0, 0.0, 0.0]
    for bi in order:
        size = sum(len(l) for l in blocks[bi]["lines"])
        s = min(range(3), key=lambda i: filled[i] / max(targets[i], 1.0))
        split_of[bi] = s
        filled[s] += size

    heldout_keys = set()
    for bi, s in split_of.items():
        if s > 0:
            for line in blocks[bi]["lines"]:
                k = _dedup_key(line)
                if k:
                    heldout_keys.add(k)

    names = ["train", "val", "test"]
    stats = {n: {"blocks": 0, "chars": 0, "lines": 0, "dropped_dup": 0}
             for n in names}
    writers = {n: open(os.path.join(out_dir, f"{n}.jsonl"), "w",
                       encoding="utf-8") for n in names}
    train_char_counter = Counter()
    for bi, block in enumerate(blocks):
        name = names[split_of[bi]]
        lines = block["lines"]
        if name == "train":
            kept = []
            for line in lines:
                k = _dedup_key(line)
                if k and k in heldout_keys:
                    stats[name]["dropped_dup"] += 1
                    continue
                kept.append(line)
            lines = kept
        if not lines:
            continue
        text = " ".join(lines)
        if name == "train":
            train_char_counter.update(text)
        rec = {"source": block["source"], "chapter": block["chapter_idx"],
               "text": text}
        writers[name].write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats[name]["blocks"] += 1
        stats[name]["chars"] += len(text)
        stats[name]["lines"] += len(lines)

    if include_gretil:
        stats["train"]["gretil_blocks"] = 0
        stats["train"]["gretil_chars"] = 0
        for block in load_gretil_blocks(raw_dir):
            kept = []
            for line in block["lines"]:
                k = _dedup_key(line)
                if k and k in heldout_keys:
                    stats["train"]["dropped_dup"] += 1
                    continue
                kept.append(line)
            if not kept:
                continue
            text = " ".join(kept)
            train_char_counter.update(text)
            rec = {"source": block["source"],
                   "chapter": block["chapter_idx"], "text": text}
            writers["train"].write(json.dumps(rec, ensure_ascii=False)
                                   + "\n")
            stats["train"]["blocks"] += 1
            stats["train"]["chars"] += len(text)
            stats["train"]["lines"] += len(kept)
            stats["train"]["gretil_blocks"] += 1
            stats["train"]["gretil_chars"] += len(text)

    for w in writers.values():
        w.close()

    vocab = build_vocab(train_char_counter)
    meta = {"seed": seed, "ratios": ratios, "stats": stats, "vocab": vocab}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return meta


def load_split(out_dir: str, name: str):
    recs = []
    with open(os.path.join(out_dir, f"{name}.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            recs.append(json.loads(line))
    return recs


def build_vocab(char_counter: Counter, min_count: int = 3):
    chars = sorted([c for c, n in char_counter.items()
                    if n >= min_count and c != MASK_CHAR])
    itos = [PAD, UNK, MASK] + chars
    return itos


class CharTokenizer:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {c: i for i, c in enumerate(itos)}
        self.pad_id, self.unk_id, self.mask_id = 0, 1, 2
        self.vocab_size = len(itos)
        self.known = {c for c in itos if c not in (PAD, UNK, MASK)}

    def encode(self, text: str):
        unk = self.unk_id
        return [self.stoi.get(c, unk) for c in text]

    def decode(self, ids):
        out = []
        for i in ids:
            c = self.itos[i]
            out.append("?" if c in (PAD, UNK) else c)
        return "".join(out)


MAX_CP_BYTES = 4
N_SLOT_KEYS = MAX_CP_BYTES * MAX_CP_BYTES + 1


def slot_key(n_bytes: int, j: int) -> int:
    return (n_bytes - 1) * MAX_CP_BYTES + j + 1


class ByteTokenizer:

    def __init__(self, itos=None):
        self.pad_id, self.mask_id = 256, 257
        self.unk_id = None
        self.vocab_size = 258
        self.itos = itos
        self.known = set()
        cands, legal = {}, {}
        for c in (itos or []):
            if c in (PAD, UNK, MASK):
                continue
            b = list(c.encode("utf-8"))
            self.known.add(c)
            cands.setdefault(len(b), []).append(b)
            for j, v in enumerate(b):
                legal.setdefault(slot_key(len(b), j), set()).add(v)
        self.cand_bytes = {n: sorted(v) for n, v in cands.items()}
        self.slot_legal = {k: sorted(v) for k, v in legal.items()}

    @property
    def constrained(self):
        return bool(self.cand_bytes)

    def encode(self, text: str):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        bs = bytes(i for i in ids if i < 256)
        return bs.decode("utf-8", errors="replace")


def get_tokenizer(kind: str, itos=None):
    if kind == "char":
        assert itos is not None
        return CharTokenizer(itos)
    if kind == "byte":
        return ByteTokenizer(itos)
    raise ValueError(kind)


DAMAGE = dict(span_mask_ratio=0.15, span_geometric_p=0.1)


def sample_damage(text: str, rng: random.Random,
                  min_ratio=0.05, max_ratio=0.5,
                  span_mask_ratio=DAMAGE["span_mask_ratio"],
                  span_geometric_p=DAMAGE["span_geometric_p"]):
    n = len(text)
    if n < 16:
        return text, [], 0.0
    rate = rng.uniform(min_ratio, max_ratio)
    total = int(rate * n)
    if total <= 0:
        return text, [], rate
    n_span = int(total * span_mask_ratio)
    n_single = total - n_span
    masked = set(rng.sample(range(n), min(n_single, n)))
    remaining, guard = n_span, 0
    while remaining > 0 and guard < 1000:
        guard += 1
        length = 1
        while rng.random() > span_geometric_p:
            length += 1
        length = min(length, remaining, n)
        s = rng.randrange(0, n - length + 1)
        masked.update(range(s, s + length))
        remaining -= length
    idx = sorted(masked)
    spans, s, prev = [], None, None
    for i in idx:
        if s is None:
            s = prev = i
        elif i == prev + 1:
            prev = i
        else:
            spans.append((s, prev + 1))
            s = prev = i
    if s is not None:
        spans.append((s, prev + 1))
    return apply_spans(text, spans), spans, rate


TASKLEN_DAMAGE = dict(geo_p=0.2, max_fill=18, edge=8, sep=1)


def _tasklen_fill(rng, lmax, geo_p):
    if rng.random() < 0.5:
        L = 1
        while rng.random() > geo_p and L < lmax:
            L += 1
        return L
    return rng.randint(1, lmax)


def sample_damage_tasklen(text: str, rng: random.Random,
                          min_ratio=0.05, max_ratio=0.5,
                          max_fill=TASKLEN_DAMAGE["max_fill"],
                          geo_p=TASKLEN_DAMAGE["geo_p"],
                          edge=TASKLEN_DAMAGE["edge"],
                          sep=TASKLEN_DAMAGE["sep"]):
    n = len(text)
    if n < 16:
        return text, [], 0.0
    rate = rng.uniform(min_ratio, max_ratio)
    total = int(rate * n)
    if total <= 0:
        return text, [], rate
    lmax = min(max_fill, n - 2 * edge)
    if lmax < 1:
        return text, [], rate
    spans, remaining, misses = [], total, 0
    while remaining > 0 and misses < 200:
        L = min(_tasklen_fill(rng, lmax, geo_p), remaining)
        s = rng.randrange(edge, n - edge - L + 1)
        e = s + L
        if all(s >= e0 + sep or e + sep <= s0 for s0, e0 in spans):
            spans.append((s, e))
            remaining -= L
            misses = 0
        else:
            misses += 1
    spans.sort()
    return apply_spans(text, spans), spans, rate


DAMAGE_MODES = {"field": sample_damage, "tasklen": sample_damage_tasklen}


def damage_fn(mode: str):
    try:
        return DAMAGE_MODES[mode]
    except KeyError:
        raise ValueError(f"unknown damage mode {mode!r}; "
                         f"choose from {sorted(DAMAGE_MODES)}")


def tidy_window(window: str) -> str:
    while window and unicodedata.category(window[0]) in ("Mn", "Mc"):
        window = window[1:]
    return window.strip()


def apply_spans(text: str, spans) -> str:
    """Materialise lacunae: every codepoint inside a span becomes ◌."""
    out = list(text)
    for cs, ce in spans:
        for i in range(cs, ce):
            out[i] = MASK_CHAR
    return "".join(out)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def find_raw_dir():
    candidates = [os.environ.get("RAW_DIR", ""), ".", ".."]
    for c in candidates:
        if c and glob.glob(os.path.join(c, "*_raw_formatted.txt")):
            return c
    hits = glob.glob("/kaggle/input/**/*_raw_formatted.txt", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError("could not locate *_raw_formatted.txt corpus")


def prepare(raw_dir=None, out_dir="data", seed=42, include_gretil=False):
    raw_dir = raw_dir or find_raw_dir()
    meta = build_splits(raw_dir, out_dir, seed=seed,
                        include_gretil=include_gretil)
    meta["damage"] = dict(DAMAGE)
    meta["include_gretil"] = include_gretil
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=None)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gretil", action="store_true",
                    help="append GRETIL sa_*.txt texts to TRAIN only")
    a = ap.parse_args()
    m = prepare(a.raw_dir, a.out_dir, a.seed, include_gretil=a.gretil)
    print(json.dumps(m["stats"], ensure_ascii=False, indent=2))
    print(f"vocab size (char): {len(m['vocab'])}")
