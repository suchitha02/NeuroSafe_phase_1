"""M1: Whisper + LoRA (or full fine-tuning) on TORGO, leave-one-speaker-out (LOSO).

1) Unseen-speaker LOSO: train on every other speaker, test on all of the held-out speaker.
    python train_lora.py --test_speaker F01
    python train_lora.py --test_speaker F01 --mode full          # full fine-tuning comparison

2) Personalisation (RQ2): the held-out speaker is split 50/50 with a fixed seed.
   Training uses ONLY N minutes from the first half; testing always uses the second half,
   so every point on the data-efficiency curve is scored on the same utterances.
    python train_lora.py --test_speaker F01 --adapt_minutes 2 \
        --init_adapter outputs/m1_lora_r32/F01/adapter --epochs 10
   --adapt_minutes 0 trains nothing and just scores the starting model on that half.
"""
import argparse
import json
import math
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (Trainer, TrainingArguments, WhisperForConditionalGeneration,
                          WhisperProcessor, set_seed)

from common import SR, load_audio, save_predictions, transcribe_whisper


class TorgoDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor):
        self.df, self.processor = df.reset_index(drop=True), processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        feats = self.processor.feature_extractor(load_audio(row.path), sampling_rate=SR).input_features[0]
        return {"input_features": feats, "labels": self.processor.tokenizer(row.text).input_ids}


def collate(batch, processor, start_token_id):
    feats = processor.feature_extractor.pad(
        [{"input_features": b["input_features"]} for b in batch], return_tensors="pt")
    labels = processor.tokenizer.pad([{"input_ids": b["labels"]} for b in batch], return_tensors="pt")
    ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
    if (ids[:, 0] == start_token_id).all():  # the model re-adds <|startoftranscript|> itself
        ids = ids[:, 1:]
    return {"input_features": feats.input_features, "labels": ids}


def build_split(df, args):
    target = df[df.speaker == args.test_speaker]
    others = df[df.speaker != args.test_speaker]
    if target.empty:
        raise SystemExit(f"No utterances for speaker {args.test_speaker} in the manifest.")

    if args.adapt_minutes is None:  # unseen-speaker LOSO
        train = others if args.include_controls else others[others.severity != "control"]
        test = target
        if args.drop_test_texts:  # strict text-disjoint training
            train = train[~train.text.isin(set(test.text))]
        if train.empty:
            raise SystemExit("LOSO training set is empty; check the manifest / flags.")
    else:  # personalisation
        shuffled = target.sample(frac=1, random_state=args.seed)
        pool, test = shuffled.iloc[: len(shuffled) // 2], shuffled.iloc[len(shuffled) // 2:]
        train = pool[pool.duration.cumsum() <= args.adapt_minutes * 60]

    # Conservative "seen text" flag: the prompt was spoken by ANY other speaker or is in the
    # training data. report.py uses it to show WER on genuinely unseen sentences/words.
    seen_texts = set(others.text) | set(train.text)
    if args.smoke:
        train, test = train.head(64), test.head(32)
    return train, test, test.text.isin(seen_texts).values


def build_model(args):
    processor = WhisperProcessor.from_pretrained(args.model, language="english", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    if args.mode == "lora":
        if args.init_adapter:
            model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        else:
            config = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                target_modules=["q_proj", "v_proj"], bias="none")
            model = get_peft_model(model, config)
    elif args.init_adapter:
        raise SystemExit("--init_adapter only makes sense with --mode lora")
    return processor, model


def default_name(args):
    name = f"m1_{args.mode}" + (f"_r{args.rank}" if args.mode == "lora" else "")
    if args.adapt_minutes is not None:
        name += "_pers" + ("_fromLOSO" if args.init_adapter else "_fromBase")
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_speaker", required=True)
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--mode", choices=["lora", "full"], default="lora")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--manifest", default="data/torgo_manifest.csv")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--name", default=None, help="system name; derived from the settings if omitted")
    ap.add_argument("--include_controls", type=int, default=1, help="use control speakers for training")
    ap.add_argument("--drop_test_texts", action="store_true",
                    help="LOSO only: remove training utterances whose text appears in the test speaker")
    ap.add_argument("--adapt_minutes", type=float, default=None)
    ap.add_argument("--init_adapter", default=None)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-3 (LoRA) / 1e-5 (full)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny run to test the setup")
    ap.add_argument("--eval_only", action="store_true",
                    help="skip training; decode with --init_adapter (e.g. re-run a failed evaluation)")
    args = ap.parse_args()
    if args.eval_only and not args.init_adapter:
        raise SystemExit("--eval_only needs --init_adapter")
    set_seed(args.seed)

    name = args.name or default_name(args)
    run_dir = Path(args.out_dir) / name / args.test_speaker
    if args.adapt_minutes is not None:
        run_dir = run_dir / f"{args.adapt_minutes:g}min"

    df = pd.read_csv(args.manifest)
    train, test, text_seen = build_split(df, args)
    processor, model = build_model(args)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{name} | test={args.test_speaker} | train={len(train)} utts "
          f"({train.duration.sum() / 60:.1f} min) | test={len(test)} | trainable={trainable:,}/{total:,}")

    if len(train) > 0 and not args.eval_only:
        steps = math.ceil(len(train) / (args.batch_size * args.grad_accum)) * args.epochs
        targs = TrainingArguments(
            output_dir=str(run_dir / "checkpoints"),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr or (1e-3 if args.mode == "lora" else 1e-5),
            num_train_epochs=args.epochs,
            warmup_steps=max(1, int(steps * 0.1)),
            fp16=torch.cuda.is_available(),
            logging_steps=25,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            label_names=["labels"],
            dataloader_num_workers=2,
            seed=args.seed,
        )
        start_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        Trainer(model=model, args=targs, train_dataset=TorgoDataset(train, processor),
                data_collator=partial(collate, processor=processor, start_token_id=start_id)).train()

    if args.mode == "lora":
        if not args.eval_only:
            model.save_pretrained(run_dir / "adapter")  # small: only the LoRA weights
        model = model.merge_and_unload()
    model = model.to("cuda").half() if torch.cuda.is_available() else model

    hyps, confs = transcribe_whisper(model, processor, test.path.tolist(), args.batch_size)
    system = name + (f"@{args.adapt_minutes:g}min" if args.adapt_minutes is not None else "")
    save_predictions(test, hyps, confs, run_dir / "predictions.csv", system,
                     base_system=name, test_speaker=args.test_speaker, text_seen=text_seen,
                     adapt_minutes=args.adapt_minutes)
    info = {"system": system, "test_speaker": args.test_speaker, "trainable_params": trainable,
            "total_params": total, "n_train": len(train), "train_minutes": train.duration.sum() / 60,
            "n_test": len(test), "args": vars(args)}
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
