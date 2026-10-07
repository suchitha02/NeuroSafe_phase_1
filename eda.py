"""Exploratory data analysis of the TORGO manifest (run after prepare_torgo.py).

    python eda.py
    python eda.py --audio_stats      # also reads every file: loudness, clipping, silence (slower)

Writes outputs/eda/eda.md plus CSVs and PNG plots.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ORDER = ["severe", "moderate", "mild", "control"]


def speaker_table(df):
    t = df.groupby(["severity", "speaker"]).agg(
        utterances=("utt_id", "size"),
        minutes=("duration", lambda d: d.sum() / 60),
        mean_dur_s=("duration", "mean"),
        pct_single_words=("utt_type", lambda u: 100 * (u == "word").mean()),
        sessions=("session", "nunique"),
        pct_headMic=("mic", lambda m: 100 * (m == "headMic").mean()),
        unique_texts=("text", "nunique"),
    ).reset_index()
    t["severity"] = pd.Categorical(t.severity, ORDER)
    return t.sort_values(["severity", "speaker"]).round(1)


def speaking_rate(df):
    """Words per second on sentences only (single words are dominated by silence padding)."""
    s = df[df.utt_type == "sentence"].copy()
    s["words_per_s"] = s.text.str.split().str.len() / s.duration
    return s


def overlap_table(df):
    """How much each dysarthric speaker's test text is also spoken by other speakers (LOSO leakage),
    and how much training data the strict --drop_test_texts mode would remove."""
    rows = []
    for spk in sorted(df[df.severity != "control"].speaker.unique()):
        test, train = df[df.speaker == spk], df[df.speaker != spk]
        test_texts = set(test.text)
        rows.append({
            "test_speaker": spk,
            "pct_test_utts_text_seen": 100 * test.text.isin(set(train.text)).mean(),
            "pct_test_sentences_seen": 100 * test[test.utt_type == "sentence"].text.isin(set(train.text)).mean(),
            "train_utts_loso": len(train),
            "train_utts_strict": int((~train.text.isin(test_texts)).sum()),
        })
    t = pd.DataFrame(rows)
    t["pct_train_kept_strict"] = 100 * t.train_utts_strict / t.train_utts_loso
    return t.round(1)


def header_stats(df):
    info = [sf.info(p) for p in df.path]
    return pd.DataFrame({"sample_rate": [i.samplerate for i in info],
                         "channels": [i.channels for i in info]}).value_counts().rename("files").reset_index()


def audio_stats(df):
    """Per-file loudness (dBFS), clipping and leading/trailing silence. Reads every file."""
    from tqdm import tqdm
    rows = []
    for path in tqdm(df.path, desc="audio"):
        x, sr = sf.read(path, always_2d=True)
        x = x.mean(1)
        frame = max(1, int(0.02 * sr))
        energy = np.array([np.sqrt(np.mean(x[i:i + frame] ** 2)) for i in range(0, len(x), frame)]) + 1e-10
        voiced = np.where(20 * np.log10(energy) > 20 * np.log10(energy.max()) - 35)[0]
        edge_silence = (len(energy) - (voiced[-1] - voiced[0] + 1)) / len(energy) if len(voiced) else 1.0
        rows.append({"rms_dbfs": 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-10),
                     "pct_clipped": 100 * np.mean(np.abs(x) >= 0.999),
                     "pct_edge_silence": 100 * edge_silence})
    return pd.concat([df[["utt_id", "speaker", "severity"]].reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def save_plots(df, rate, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speakers = speaker_table(df).speaker.tolist()

    fig, ax = plt.subplots(figsize=(9, 4))
    for sev in ORDER:
        d = df[df.severity == sev].duration
        if len(d):
            ax.hist(d, bins=60, range=(0, 12), alpha=0.5, density=True, label=sev)
    ax.set(xlabel="utterance duration (s)", ylabel="density", title="Duration by severity")
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "duration_by_severity.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    data = [rate[rate.speaker == s].words_per_s.dropna() for s in speakers]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(speakers) + 1), speakers, rotation=45)
    ax.set(ylabel="words per second", title="Speaking rate on sentences (lower = slower speech)")
    fig.tight_layout(); fig.savefig(out / "speaking_rate_by_speaker.png", dpi=150); plt.close(fig)

    counts = df.groupby(["speaker", "utt_type"]).size().unstack(fill_value=0).reindex(speakers)
    ax = counts.plot(kind="bar", stacked=True, figsize=(10, 4), title="Utterances per speaker")
    ax.set(xlabel="", ylabel="utterances")
    ax.figure.tight_layout(); ax.figure.savefig(out / "utterances_per_speaker.png", dpi=150); plt.close(ax.figure)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/torgo_manifest.csv")
    ap.add_argument("--out_dir", default="outputs/eda")
    ap.add_argument("--audio_stats", action="store_true", help="read every file (slower)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest)

    spk = speaker_table(df)
    sev = df.groupby("severity").agg(speakers=("speaker", "nunique"), utterances=("utt_id", "size"),
                                     minutes=("duration", lambda d: d.sum() / 60),
                                     median_dur_s=("duration", "median")).reindex(ORDER).dropna().round(1)
    rate = speaking_rate(df)
    rate_table = rate.groupby(["severity", "speaker"]).words_per_s.median().round(2).rename(
        "median_words_per_s").reset_index()
    overlap = overlap_table(df)
    top_prompts = (df.groupby("text").agg(speakers=("speaker", "nunique"), utterances=("utt_id", "size"))
                   .sort_values(["speakers", "utterances"], ascending=False).head(20).reset_index())
    headers = header_stats(df)

    tables = {"speakers": spk, "severity": sev.reset_index(), "speaking_rate": rate_table,
              "prompt_overlap": overlap, "top_prompts": top_prompts, "audio_format": headers}
    if args.audio_stats:
        a = audio_stats(df)
        a.to_csv(out / "audio_stats_per_file.csv", index=False)
        tables["audio_quality"] = a.groupby(["severity", "speaker"])[
            ["rms_dbfs", "pct_clipped", "pct_edge_silence"]].median().round(2).reset_index()
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)

    try:
        save_plots(df, rate, out)
    except ImportError:
        print("matplotlib not installed: plots skipped")

    titles = {"speakers": "Per-speaker summary", "severity": "Per-severity summary",
              "speaking_rate": "Speaking rate (median words/s on sentences)",
              "prompt_overlap": "Prompt overlap per LOSO fold (leakage check)",
              "top_prompts": "Most-repeated prompts", "audio_format": "Audio format (sample rate, channels)",
              "audio_quality": "Audio quality (medians per speaker)"}
    md = [f"# TORGO EDA\n\n{len(df)} utterances, {df.speaker.nunique()} speakers, "
          f"{df.duration.sum() / 3600:.1f} hours.\n"]
    for name, t in tables.items():
        md += [f"\n## {titles[name]}\n", t.to_markdown(index=False), "\n"]
    md += ["\nPlots: duration_by_severity.png, speaking_rate_by_speaker.png, utterances_per_speaker.png\n"]
    (out / "eda.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
