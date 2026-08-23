import argparse
import glob
import os
import random
import sys

import torch
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from common import load_split
from byt5_data import (ByT5TrainDataset, ByT5EvalDataset,
                       load_eval)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="buddhist-nlp/byt5-sanskrit")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="runs_byt5")
    ap.add_argument("--max_steps", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=2, help="per GPU")
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--max_length", type=int, default=1024,
                    help="token cap for 512-IAST-char pseudo-paragraphs; "
                         "byte ids stay well under this")
    ap.add_argument("--damage", choices=("field", "tasklen"), default="field",
                    help="training corruption recipe: field = isolated "
                         "single-char gaps; tasklen = eval-shaped gap lengths")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_examples", type=int, default=1150,
                    help="1150 = the full span-only val file (50/len x 23); "
                         "smaller slices made eval_loss noisy enough to fool "
                         "early stopping")
    ap.add_argument("--val_file", default="byt5_spanonly_val.jsonl",
                    help="span-only validation set (Ithaca/Aeneas valid "
                         "mode); built by byt5_eval.py --build_spanonly "
                         "--split val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--early_stopping_patience", type=int, default=10,
                    help="evals without improvement before stopping; "
                         "0 disables early stopping")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--fp16", action="store_true",
                    help="discouraged: T5 is unstable in fp16 on T4")
    return ap.parse_args()


def find_resume_ckpt(out_dir):
    import json

    def weights_intact(d):
        for w in glob.glob(os.path.join(d, "*.safetensors")):
            try:
                from safetensors import safe_open
                with safe_open(w, framework="pt"):
                    pass
            except Exception as e:
                print(f"resume: skipping {d} — corrupt/truncated weights "
                      f"({e})", flush=True)
                return False
        return True

    def candidates(root):
        out = []
        for sp in glob.glob(os.path.join(root, "**", "trainer_state.json"),
                            recursive=True):
            d = os.path.dirname(sp)
            has_weights = (glob.glob(os.path.join(d, "*.safetensors"))
                           or glob.glob(os.path.join(d, "pytorch_model*")))
            has_opt = glob.glob(os.path.join(d, "optimizer*"))
            if not (has_weights and has_opt):
                continue
            try:
                step = json.load(open(sp))["global_step"]
            except Exception:
                continue
            if not weights_intact(d):
                continue
            out.append((step, d))
        return out

    for root in (out_dir, "/kaggle/input"):
        if os.path.isdir(root):
            cands = candidates(root)
            if cands:
                return max(cands)[1]
    return None


def main():
    args = parse_args()
    is_main = os.environ.get("LOCAL_RANK", "0") == "0"

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.set_input_embeddings(model.shared)
    model.config.tie_word_embeddings = False
    model.config.use_cache = False
    if is_main:
        n = sum(p.numel() for p in model.parameters())
        print(f"model {args.model}  params {n/1e6:.0f}M", flush=True)

    train_recs = load_split(args.data_dir, "train")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    size = args.max_steps * args.batch_size * args.grad_accum * world
    train_ds = ByT5TrainDataset(train_recs, tok, size=size, seed=args.seed,
                                max_length=args.max_length, damage=args.damage)
    val_ex = load_eval(os.path.join(args.data_dir, args.val_file))
    random.Random(0).shuffle(val_ex)
    val_ds = ByT5EvalDataset(val_ex[:args.eval_examples], tok,
                             max_length=args.max_length)

    collator = DataCollatorForSeq2Seq(tok, model=model, padding=True)

    targ_kwargs = dict(
        output_dir=args.out_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        lr_scheduler_type="cosine",
        optim="adafactor",
        weight_decay=0.0,
        label_smoothing_factor=args.label_smoothing,
        fp16=args.fp16,
        gradient_checkpointing=True,
        logging_steps=args.log_every,
        eval_strategy="steps",
        eval_steps=args.save_every,
        save_steps=args.save_every,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.num_workers,
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )
    try:
        targs = Seq2SeqTrainingArguments(eval_on_start=True, **targ_kwargs)
    except TypeError:
        targs = Seq2SeqTrainingArguments(**targ_kwargs)

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience))
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_ds,
        eval_dataset=val_ds, data_collator=collator,
        callbacks=callbacks)

    resume = find_resume_ckpt(args.out_dir)
    if is_main:
        print(f"resume from: {resume or 'scratch'}", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    if is_main:
        final = os.path.join(args.out_dir, "final")
        trainer.save_model(final)
        tok.save_pretrained(final)
        print(f"saved final model -> {final}", flush=True)


if __name__ == "__main__":
    main()
