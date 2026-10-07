"""Scan a TORGO download and write a manifest CSV (one row per usable utterance).

Expected layout (official distribution, extracted anywhere under --torgo_root):
    <root>/.../F01/Session1/prompts/0001.txt
    <root>/.../F01/Session1/wav_headMic/0001.wav   (or wav_arrayMic)

Usage:
    python prepare_torgo.py --torgo_root /path/to/TORGO
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import soundfile as sf

from common import clean_text, severity

SPEAKER_RE = re.compile(r"^(F|M|FC|MC)\d{2}$")
# Picture-description prompts point to image files; bracketed prompts are non-speech
# instructions (e.g. "[relax your mouth ...]"); "xxx" marks unusable prompts.
NON_SPEECH_RE = re.compile(r"\[|\]|\.jpg|/|xxx", re.IGNORECASE)


def find_wav(session, stem, mics):
    for mic in mics:
        wav = session / f"wav_{mic}" / f"{stem}.wav"
        if wav.exists():
            return wav
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torgo_root", required=True)
    ap.add_argument("--out", default="data/torgo_manifest.csv")
    ap.add_argument("--mics", default="headMic,arrayMic", help="microphone preference order")
    ap.add_argument("--max_seconds", type=float, default=30.0, help="Whisper's input window")
    args = ap.parse_args()

    mics = args.mics.split(",")
    rows, skipped = [], Counter()
    for session in sorted(Path(args.torgo_root).rglob("Session*")):
        if not session.is_dir() or not SPEAKER_RE.match(session.parent.name):
            continue
        speaker = session.parent.name
        for prompt_file in sorted((session / "prompts").glob("*.txt")):
            raw = prompt_file.read_text(errors="ignore").strip()
            text = clean_text(raw)
            if NON_SPEECH_RE.search(raw) or not text:
                skipped["non-speech prompt"] += 1
                continue
            wav = find_wav(session, prompt_file.stem, mics)
            if wav is None:
                skipped["no audio"] += 1
                continue
            try:
                info = sf.info(wav)
                duration = info.frames / info.samplerate
            except Exception:
                skipped["unreadable audio"] += 1
                continue
            if not 0.2 <= duration <= args.max_seconds:
                skipped["bad duration"] += 1
                continue
            rows.append({
                "utt_id": f"{speaker}_{session.name}_{prompt_file.stem}",
                "speaker": speaker,
                "severity": severity(speaker),
                "session": session.name,
                "mic": wav.parent.name.replace("wav_", ""),
                "path": str(wav.resolve()),
                "duration": round(duration, 3),
                "text": text,
                "utt_type": "word" if len(text.split()) == 1 else "sentence",
            })

    if not rows:
        raise SystemExit(f"No utterances found under {args.torgo_root}. Check the folder layout.")
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    summary = df.groupby(["severity", "speaker"]).agg(
        utterances=("utt_id", "size"), minutes=("duration", lambda d: round(d.sum() / 60, 1)),
        words=("utt_type", lambda t: (t == "word").sum()), mics=("mic", lambda m: ",".join(sorted(set(m)))))
    print(summary.to_string())
    print(f"\n{len(df)} utterances written to {args.out}; skipped: {dict(skipped)}")


if __name__ == "__main__":
    main()
