#!/usr/bin/env bash
# Full Stream 1 pipeline. Run stages one at a time the first time (see README).
# Usage: bash run_all.sh /path/to/TORGO
set -euo pipefail
TORGO_ROOT=${1:?"usage: bash run_all.sh /path/to/TORGO"}
SPEAKERS="F01 F03 F04 M01 M02 M03 M04 M05"

echo "== Stage 0: manifest"
python prepare_torgo.py --torgo_root "$TORGO_ROOT"

echo "== Stage 1: M0 zero-shot baselines"
for MODEL in openai/whisper-tiny openai/whisper-base openai/whisper-small \
             facebook/wav2vec2-base-960h facebook/hubert-large-ls960-ft; do
  python zeroshot.py --model "$MODEL"
done

echo "== Stage 2: M1 LoRA, leave-one-speaker-out"
for S in $SPEAKERS; do
  python train_lora.py --test_speaker "$S" --mode lora --rank 32
done

echo "== Stage 3 (optional, slower): full fine-tuning comparison"
# for S in $SPEAKERS; do python train_lora.py --test_speaker "$S" --mode full --batch_size 8 --grad_accum 2; done

echo "== Stage 4: personalisation curve, starting from each speaker's LOSO adapter"
for S in $SPEAKERS; do
  for MIN in 0 1 2 5; do
    python train_lora.py --test_speaker "$S" --adapt_minutes "$MIN" \
      --init_adapter "outputs/m1_lora_r32/$S/adapter" --epochs 10 --batch_size 8
  done
done

echo "== Stage 5: report"
python report.py --compare m0_whisper-small m1_lora_r32
