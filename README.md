# NeuroSAFE-Voice — Stream 1: dysarthric ASR baselines and LoRA personalisation on TORGO

This covers M0 and M1 of the ablation table: how bad off-the-shelf ASR is on dysarthric speech, and how much Whisper + LoRA recovers. Unseen speakers and small amounts of target-speaker data are tested separately.

| Script | Purpose |
|---|---|
| `prepare_torgo.py` | Scans TORGO and writes `data/torgo_manifest.csv`. Drops non-speech prompts, picks one mic per utterance and records duration and word/sentence type. |
| `eda.py` | Dataset EDA: per-speaker/severity summaries, durations, speaking rate, prompt-overlap (leakage) per fold, audio format and optional audio quality. |
| `zeroshot.py` | **M0.** Zero-shot Whisper (tiny/base/small) and CTC models (wav2vec2, HuBERT). |
| `train_lora.py` | **M1.** Whisper + LoRA or full fine-tuning, leave-one-speaker-out. Also runs the personalisation curve. |
| `report.py` | Per-speaker, per-severity, seen/unseen-text and word/sentence WER with 95% bootstrap CIs. Also produces the paired bootstrap comparison, trainable-parameter table and personalisation curve. |
| `common.py` | Shared helpers: severity map, text normalisation, decoding with confidence. |

## 1. Setup

```bash
pip install -r requirements.txt
```

Download TORGO from the official page (the `F`, `FC`, `M` and `MC` archives) and extract everything under one folder. The scanner finds `<speaker>/Session*/prompts` and `wav_headMic` / `wav_arrayMic` anywhere below that folder.

On Colab, put TORGO on Google Drive and set the runtime to GPU (a T4 is enough for whisper-small + LoRA). Copy the audio to local disk (`/content`) before training, because reading thousands of small files from Drive is slow.

## 2. Smoke test first (about 5 minutes)

```bash
python prepare_torgo.py --torgo_root /path/to/TORGO
python zeroshot.py --model openai/whisper-tiny --smoke
python train_lora.py --test_speaker F01 --model openai/whisper-tiny --smoke --epochs 1
python report.py
```

Check the per-speaker counts printed by `prepare_torgo.py` against the TORGO documentation before running anything long. Then run `python eda.py --audio_stats` and read `outputs/eda/eda.md`. The prompt-overlap table shows how much leakage each LOSO fold has and how much training data `--drop_test_texts` would remove.

## 3. Full runs

`bash run_all.sh /path/to/TORGO` runs every stage. The first time, run the stages one by one:

```bash
# M0 baselines
python zeroshot.py --model openai/whisper-small          # repeat for tiny, base, CTC models

# M1: unseen-speaker LOSO, one fold per dysarthric speaker
python train_lora.py --test_speaker F01                  # repeat for all 8 speakers
python train_lora.py --test_speaker F01 --rank 8         # rank ablation -> system m1_lora_r8
python train_lora.py --test_speaker F01 --mode full      # full fine-tuning comparison

# Personalisation curve (RQ2): N minutes of the target speaker, same fixed test half every time
python train_lora.py --test_speaker F01 --adapt_minutes 2 \
    --init_adapter outputs/m1_lora_r32/F01/adapter --epochs 10 --batch_size 8

python report.py --compare m0_whisper-small m1_lora_r32
```

**Runtime.** With controls included, each LOSO fold trains on roughly all of TORGO. For faster iteration, use `--include_controls 0`, `--epochs 2` or `--model openai/whisper-base`. Keep the final settings identical across folds.

## 4. What the report gives you

`outputs/report/report.md` contains:

- **Per speaker and per severity WER with 95% CIs.** This is the M0 vs M1 table for the slides.
- **Seen vs unseen text.** TORGO repeats many prompts across speakers, so a LOSO model can memorise sentences it heard from other speakers. Always show the unseen-text column next to the overall number. For a strict test, rerun with `--drop_test_texts`, which removes every training utterance whose text appears in the test speaker's set.
- **Words vs sentences.** Whisper often hallucinates on short isolated words, and this split makes that visible.
- **Trainable parameters.** These are the "ΔWER; trainable parameters" numbers for M1.
- **Paired bootstrap ΔWER, overall and per speaker.** Use this to claim "M1 is better than M0" with a CI instead of two raw numbers.
- **Personalisation curve.** WER vs minutes of target-speaker data, from `personalisation_curve_*.png`. This is the direct answer to RQ2.

Every predictions file also stores a per-utterance confidence (Whisper mean token log-prob, CTC mean max-softmax). Stream 3's calibration and risk-gate study can reuse these outputs directly.

## 5. Design decisions worth stating in the presentation

- **Speaker-disjoint evaluation only.** Every fold holds out one dysarthric speaker completely. Control speakers are used for training only.
- **One scoring normaliser for all systems.** Whisper's English normaliser is applied to both references and hypotheses, so CTC and Whisper models are scored identically. Do not mix reports made with and without it (the script warns if it falls back).
- **Corpus-level WER.** Total errors are divided by total reference words, so short and long utterances are weighted correctly.
- **Personalisation scoring.** Personalisation runs are scored on a fixed 50% of each speaker, so they are only compared with each other (the curve), not with the LOSO table.
- **Severity labels.** The labels in `common.py` follow common usage in the TORGO literature. Confirm them against the source you cite.
