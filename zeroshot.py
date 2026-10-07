"""M0: zero-shot baselines on TORGO (no training).

    python zeroshot.py --model openai/whisper-small
    python zeroshot.py --model facebook/wav2vec2-base-960h
    python zeroshot.py --model facebook/hubert-large-ls960-ft
"""
import argparse
from pathlib import Path

import pandas as pd
from transformers import (AutoModelForCTC, AutoProcessor, WhisperForConditionalGeneration,
                          WhisperProcessor)

from common import device, save_predictions, transcribe_ctc, transcribe_whisper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--manifest", default="data/torgo_manifest.csv")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--include_controls", action="store_true", help="also decode control speakers")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="10 utterances per speaker, to test the setup")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if not args.include_controls:
        df = df[df.severity != "control"]
    if args.smoke:
        df = df.groupby("speaker").head(10)
    paths = df.path.tolist()

    if "whisper" in args.model.lower():
        processor = WhisperProcessor.from_pretrained(args.model)
        model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device())
        if device() == "cuda":
            model = model.half()
        hyps, confs = transcribe_whisper(model, processor, paths, args.batch_size)
    else:
        processor = AutoProcessor.from_pretrained(args.model)
        model = AutoModelForCTC.from_pretrained(args.model).to(device())
        hyps, confs = transcribe_ctc(model, processor, paths, max(1, args.batch_size // 2))

    system = "m0_" + args.model.split("/")[-1]
    save_predictions(df, hyps, confs, Path(args.out_dir) / system / "predictions.csv", system,
                     test_speaker=df.speaker.values)


if __name__ == "__main__":
    main()
