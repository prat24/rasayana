import argparse
import glob
import json
import math
import os
import random
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import get_tokenizer, load_split, levenshtein
from protocol import load_examples
from data import (RestorationTrainDataset, RestorationEvalDataset,
                  legal_matrix, make_collate)
from model import ModelConfig, RestorationModel


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", choices=("char", "byte"), required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="runs/run")
    ap.add_argument("--window_chars", type=int, default=384)
    ap.add_argument("--max_len", type=int, default=0,
                    help="0 = window_chars (char) / 3*window_chars (byte)")
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=14)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=1376)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=48, help="per GPU")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--damage", choices=("field", "tasklen"),
                    default="field",
                    help="training corruption recipe: field = the "
                         "Ithaca/Aeneas port (85%% of budget on isolated "
                         "singles); tasklen = budget-matched ablation "
                         "that spends the SAME masked-codepoint budget "
                         "on eval-shaped gap lengths (see common.py)")
    ap.add_argument("--min_damage", type=float, default=0.0)
    ap.add_argument("--max_damage", type=float, default=0.75)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_examples", type=int, default=1500)
    ap.add_argument("--val_file", default="spanonly_val.jsonl",
                    help="span-only validation set (Ithaca/Aeneas valid "
                         "mode); built by protocol.py --split val")
    ap.add_argument("--patience", type=int, default=8,
                    help="early stop after N evals without val CER improving")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no_resume", dest="resume", action="store_false")
    return ap.parse_args()


def setup_dist():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available()
                                else "gloo",
                                timeout=timedelta(minutes=30))
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world, local = 0, 1, 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    return rank, world, device


def find_input_ckpt(cfg_dict, is_main, damage=None):
    best, best_step = None, -1
    for path in sorted(glob.glob("/kaggle/input/**/last.pt",
                                 recursive=True)):
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            if is_main:
                print(f"resume scan: unreadable {path}: {e}", flush=True)
            continue
        if ck.get("cfg") != cfg_dict:
            if is_main:
                print(f"resume scan: {path} config mismatch "
                      f"(different run/flags), skipping", flush=True)
            continue
        ck_damage = ck.get("args", {}).get("damage", "field")
        if damage is not None and ck_damage != damage:
            if is_main:
                print(f"resume scan: {path} damage mismatch "
                      f"({ck_damage} != {damage}), skipping", flush=True)
            continue
        if ck["step"] > best_step:
            best, best_step = path, ck["step"]
    return best


def restoration_loss(logits, tgt, lm, slot, legal, eps):
    lg, tg = logits[lm].float(), tgt[lm]
    if legal is None:
        return F.cross_entropy(lg, tg, label_smoothing=eps)
    ok = legal[slot[lm]]
    lp = torch.log_softmax(lg.masked_fill(~ok, float("-inf")), dim=-1)
    nll = -lp.gather(1, tg[:, None]).squeeze(1)
    if eps <= 0:
        return nll.mean()
    n_ok = ok.sum(-1).clamp(min=1).float()
    smooth = -(lp.masked_fill(~ok, 0.0).sum(-1) / n_ok)
    return ((1 - eps) * nll + eps * smooth).mean()


def lr_lambda_fn(warmup, steps, min_frac):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        t = (step - warmup) / max(1, steps - warmup)
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t))
    return f


@torch.no_grad()
def validate(model, loader, tok, device, amp_dtype, legal=None):
    model.eval()
    n_correct, n_masked = 0, 0
    cer_num, cer_den = 0.0, 0
    losses = []
    for batch in loader:
        inp = batch["input"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)
        lm = batch["loss_mask"].to(device, non_blocking=True)
        km = batch["key_mask"].to(device, non_blocking=True)
        slot = batch["slot"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            logits = model(inp, km)
        if legal is not None:
            logits = logits.float().masked_fill(~legal[slot], float("-inf"))
        if lm.any():
            losses.append(
                restoration_loss(logits, tgt, lm, slot, None, 0.0).item())
        pred = logits.argmax(-1)
        n_correct += ((pred == tgt) & lm).sum().item()
        n_masked += lm.sum().item()
        for i in range(inp.shape[0]):
            pos = lm[i].nonzero(as_tuple=True)[0].tolist()
            if not pos:
                continue
            pred_i, tgt_i = pred[i].tolist(), tgt[i].tolist()
            runs, cur = [], []
            for p in pos:
                if cur and p != cur[-1] + 1:
                    runs.append(cur)
                    cur = []
                cur.append(p)
            runs.append(cur)
            for r in runs:
                t = tok.decode([tgt_i[p] for p in r])
                cer_num += levenshtein(tok.decode([pred_i[p] for p in r]), t)
                cer_den += len(t)
    model.train()
    return {"val_loss": sum(losses) / max(1, len(losses)),
            "val_acc": n_correct / max(1, n_masked),
            "val_cer": cer_num / max(1, cer_den)}


def main():
    args = parse_args()
    rank, world, device = setup_dist()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)

    if args.max_len == 0:
        args.max_len = args.window_chars * (3 if args.tokenizer == "byte"
                                            else 1)

    meta = json.load(open(os.path.join(args.data_dir, "meta.json"),
                          encoding="utf-8"))
    tok = get_tokenizer(args.tokenizer, itos=meta["vocab"])

    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=args.d_model,
                      n_layers=args.n_layers, n_heads=args.n_heads,
                      d_ff=args.d_ff, max_len=args.max_len + 64,
                      dropout=args.dropout)
    model = RestorationModel(cfg).to(device)
    legal = legal_matrix(tok, args.tokenizer, device)
    if is_main:
        print(f"[{args.tokenizer}] params: {model.num_params()/1e6:.1f}M  "
              f"vocab: {tok.vocab_size}  max_len: {args.max_len}  "
              f"world: {world}", flush=True)
        if legal is not None:
            per_slot = {k: len(v) for k, v in sorted(tok.slot_legal.items())}
            print(f"[byte] output constrained to the char inventory: "
                  f"{sum(len(v) for v in tok.cand_bytes.values())} characters, "
                  f"legal values per slot {per_slot}", flush=True)
    if world > 1:
        model = DDP(model, device_ids=[device.index])
    raw_model = model.module if world > 1 else model

    steps_per_epoch = args.eval_every
    total_epochs = math.ceil(args.steps / steps_per_epoch)
    train_recs = load_split(args.data_dir, "train")
    ds = RestorationTrainDataset(
        train_recs, tok, args.tokenizer, args.window_chars, args.max_len,
        size=steps_per_epoch * args.batch_size * world, seed=args.seed,
        min_ratio=args.min_damage, max_ratio=args.max_damage,
        damage=args.damage)
    sampler = (DistributedSampler(ds, num_replicas=world, rank=rank,
                                  shuffle=True, seed=args.seed)
               if world > 1 else None)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        shuffle=(sampler is None),
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, collate_fn=make_collate(tok.pad_id),
                        persistent_workers=False)

    val_loader = None
    if is_main:
        val_examples = load_examples(
            os.path.join(args.data_dir, args.val_file))
        random.Random(0).shuffle(val_examples)
        val_examples = val_examples[:args.eval_examples]
        val_ds = RestorationEvalDataset(val_examples, tok, args.tokenizer,
                                        args.max_len)
        val_loader = DataLoader(val_ds, batch_size=max(8, args.batch_size),
                                shuffle=False, num_workers=0,
                                collate_fn=make_collate(tok.pad_id))

    decay, no_decay = [], []
    for n, p in raw_model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.98), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_fn(args.warmup, args.steps, args.min_lr_frac))
    use_amp = device.type == "cuda"
    amp_dtype = torch.float16 if use_amp else None
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    os.makedirs(args.out_dir, exist_ok=True)
    last_path = os.path.join(args.out_dir, "last.pt")
    best_path = os.path.join(args.out_dir, "best.pt")
    log_path = os.path.join(args.out_dir, "log.jsonl")

    step, best_cer, evals_since_best = 0, float("inf"), 0
    resume_path = None
    if args.resume:
        if os.path.exists(last_path):
            resume_path = last_path
        elif os.path.isdir("/kaggle/input"):
            resume_path = find_input_ckpt(cfg.to_dict(), is_main, args.damage)
    if resume_path:
        ck = torch.load(resume_path, map_location="cpu",
                        weights_only=False)
        if ck["cfg"] != cfg.to_dict():
            raise ValueError(
                f"checkpoint at {resume_path} was trained with a different "
                f"config:\n  ckpt: {ck['cfg']}\n  now : {cfg.to_dict()}\n"
                "resume with identical --tokenizer/--d_model/... flags, or "
                "use --no_resume / a fresh --out_dir")
        if ck["args"].get("eval_every") != args.eval_every and is_main:
            print("WARNING: --eval_every changed across resume; the data "
                  "stream will not continue exactly where it left off",
                  flush=True)
        raw_model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        best_cer = ck.get("best_cer", best_cer)
        evals_since_best = ck.get("evals_since_best", 0)
        del ck
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if is_main:
            print(f"resumed from {resume_path} at step {step} "
                  f"(best val CER {best_cer:.4f})", flush=True)

    def save(path, full):
        payload = {"model": raw_model.state_dict(),
                   "cfg": cfg.to_dict(), "vocab": meta["vocab"],
                   "args": vars(args), "step": step, "best_cer": best_cer}
        if full:
            payload.update({"opt": opt.state_dict(),
                            "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(),
                            "evals_since_best": evals_since_best})
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    stop = torch.zeros(1, device=device)
    ema_loss, t0 = None, time.time()
    start_epoch = step // steps_per_epoch
    for epoch in range(start_epoch, total_epochs):
        ds.epoch = epoch
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for batch in loader:
            if step >= args.steps:
                break
            inp = batch["input"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            lm = batch["loss_mask"].to(device, non_blocking=True)
            km = batch["key_mask"].to(device, non_blocking=True)
            slot = batch["slot"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype,
                                enabled=use_amp):
                logits = model(inp, km)
                if lm.any():
                    loss = restoration_loss(logits, tgt, lm, slot, legal,
                                            args.label_smoothing)
                else:
                    loss = logits.sum() * 0.0
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1

            l = loss.item()
            ema_loss = l if ema_loss is None else 0.98 * ema_loss + 0.02 * l
            if is_main and step % 100 == 0:
                dt = time.time() - t0
                print(f"step {step}/{args.steps}  loss {ema_loss:.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  "
                      f"{100/dt:.1f} it/s", flush=True)
                with open(log_path, "a") as fh:
                    fh.write(json.dumps({"step": step, "loss": ema_loss,
                                         "lr": sched.get_last_lr()[0]})
                             + "\n")
                t0 = time.time()

        if is_main:
            m = validate(raw_model, val_loader, tok, device, amp_dtype, legal)
            improved = m["val_cer"] < best_cer
            if improved:
                best_cer = m["val_cer"]
                evals_since_best = 0
                save(best_path, full=False)
            else:
                evals_since_best += 1
            save(last_path, full=True)
            print(f"  eval @ {step}: loss {m['val_loss']:.4f}  "
                  f"acc {m['val_acc']:.4f}  cer {m['val_cer']:.4f}"
                  f"{'  *best*' if improved else ''}", flush=True)
            with open(log_path, "a") as fh:
                fh.write(json.dumps({"step": step, **m,
                                     "best_cer": best_cer}) + "\n")
            if evals_since_best >= args.patience:
                print(f"early stop: no improvement in "
                      f"{args.patience} evals", flush=True)
                stop.fill_(1)
        if world > 1:
            dist.broadcast(stop, src=0)
        if stop.item() == 1:
            break

    if is_main:
        print(f"done at step {step}. best val CER {best_cer:.4f}  "
              f"-> {best_path}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
