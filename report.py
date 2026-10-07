"""Turn every outputs/**/predictions.csv into the Phase-1 tables and plots.

    python report.py
    python report.py --compare m0_whisper-small m1_lora_r32

Writes outputs/report/report.md plus CSVs and personalisation_curve.png.
WER is corpus-level (total errors / total reference words) with 95% bootstrap CIs over utterances.
"""
import argparse
import json
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd

from common import score_norm


def load_predictions(root):
    files = sorted(Path(root).rglob("predictions.csv"))
    if not files:
        raise SystemExit(f"No predictions.csv found under {root}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for col in ("base_system", "adapt_minutes", "text_seen"):
        if col not in df:
            df[col] = np.nan
    df["base_system"] = df.base_system.fillna(df.system)

    df["ref_n"] = df.text.map(score_norm)
    df["hyp_n"] = df.hyp.fillna("").map(score_norm)
    df = df[df.ref_n.str.len() > 0].copy()
    counts = [jiwer.process_words(r, h) for r, h in zip(df.ref_n, df.hyp_n)]
    df["errors"] = [c.substitutions + c.deletions + c.insertions for c in counts]
    df["ref_words"] = df.ref_n.str.split().str.len()

    # Zero-shot runs have no training set, so borrow the overlap flag from the trained runs.
    seen = df.dropna(subset=["text_seen"]).assign(
        text_seen=lambda d: d.text_seen.astype(str).str.lower() == "true").groupby("utt_id").text_seen.max()
    df["text_seen"] = df.utt_id.map(seen).map({True: "seen text", False: "unseen text"}).fillna("unknown")
    return df


def wer_ci(errors, words, n_boot, rng):
    errors, words = np.asarray(errors), np.asarray(words)
    idx = rng.integers(0, len(errors), (n_boot, len(errors)))
    boots = errors[idx].sum(1) / words[idx].sum(1)
    return errors.sum() / words.sum(), *np.percentile(boots, [2.5, 97.5])


def wer_table(df, by, n_boot, rng):
    rows = []
    for (system, group), g in df.groupby(["system", by]):
        wer, lo, hi = wer_ci(g.errors, g.ref_words, n_boot, rng)
        rows.append({"system": system, by: group, "n_utts": len(g), "WER%": round(100 * wer, 1),
                     "CI95": f"[{100 * lo:.1f}, {100 * hi:.1f}]"})
    long = pd.DataFrame(rows)
    wide = long.assign(cell=long["WER%"].astype(str) + " " + long.CI95).pivot(
        index="system", columns=by, values="cell")
    return long, wide


def paired_compare(df, a, b, n_boot, rng):
    """Paired bootstrap on the utterances both systems decoded. delta < 0 means B is better."""
    A, B = df[df.system == a].set_index("utt_id"), df[df.system == b].set_index("utt_id")
    common = A.index.intersection(B.index)
    if common.empty:
        raise SystemExit(f"{a} and {b} share no utterances")
    A, B = A.loc[common], B.loc[common]
    rows = []
    for speaker in ["ALL"] + sorted(A.speaker.unique()):
        m = np.ones(len(A), bool) if speaker == "ALL" else (A.speaker == speaker).to_numpy()
        ea, eb, n = A.errors.to_numpy()[m], B.errors.to_numpy()[m], A.ref_words.to_numpy()[m]
        idx = rng.integers(0, m.sum(), (n_boot, m.sum()))
        deltas = (eb[idx].sum(1) - ea[idx].sum(1)) / n[idx].sum(1)
        rows.append({"speaker": speaker, "n_utts": int(m.sum()),
                     f"WER% {a}": round(100 * ea.sum() / n.sum(), 1),
                     f"WER% {b}": round(100 * eb.sum() / n.sum(), 1),
                     "delta (pts)": round(100 * (eb.sum() - ea.sum()) / n.sum(), 1),
                     "CI95": f"[{100 * np.percentile(deltas, 2.5):.1f}, {100 * np.percentile(deltas, 97.5):.1f}]",
                     "p(B not better)": round(float((deltas >= 0).mean()), 4)})
    return pd.DataFrame(rows)


def personalisation_curve(df, out_dir):
    curve = df.dropna(subset=["adapt_minutes"])
    if curve.empty:
        return None
    table = (curve.groupby(["base_system", "speaker", "adapt_minutes"])
             .apply(lambda g: 100 * g.errors.sum() / g.ref_words.sum(), include_groups=False)
             .rename("WER%").reset_index())
    table.to_csv(out_dir / "personalisation_curve.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for base, t in table.groupby("base_system"):
            fig, ax = plt.subplots(figsize=(6, 4))
            for speaker, s in t.groupby("speaker"):
                ax.plot(s.adapt_minutes, s["WER%"], marker="o", alpha=0.6, label=speaker)
            mean = t.groupby("adapt_minutes")["WER%"].mean()
            ax.plot(mean.index, mean.values, "k-", lw=3, label="mean")
            ax.set(xlabel="target-speaker adaptation data (minutes)", ylabel="WER (%)", title=base)
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(out_dir / f"personalisation_curve_{base}.png", dpi=150)
            plt.close(fig)
    except ImportError:
        print("matplotlib not installed: curve saved as CSV only")
    return table.pivot_table(index=["base_system", "speaker"], columns="adapt_minutes", values="WER%").round(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--report_dir", default="outputs/report")
    ap.add_argument("--compare", nargs=2, metavar=("SYSTEM_A", "SYSTEM_B"))
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_predictions(args.outputs)
    df[["system", "utt_id", "speaker", "severity", "utt_type", "text_seen",
        "ref_n", "hyp_n", "conf", "errors", "ref_words"]].to_csv(out / "all_scored.csv", index=False)

    # Only LOSO / zero-shot systems are evaluated on full speakers; personalisation runs use a
    # fixed half of each speaker, so they are kept out of the main tables and shown in the curve.
    main_df = df[df.adapt_minutes.isna()]
    sections = ["# TORGO Stream 1 results\n",
                "WER% with 95% bootstrap CI. Personalisation runs are reported separately "
                "because they are scored on a fixed half of each speaker.\n"]
    for by, title in [("speaker", "Per speaker"), ("severity", "Per severity"),
                      ("text_seen", "Prompt overlap (seen vs unseen text)"),
                      ("utt_type", "Single words vs sentences")]:
        long, wide = wer_table(main_df, by, args.n_boot, rng)
        long.to_csv(out / f"wer_by_{by}.csv", index=False)
        sections += [f"\n## {title}\n", wide.to_markdown(), "\n"]
    overall, _ = wer_table(main_df.assign(all="ALL"), "all", args.n_boot, rng)
    sections += ["\n## Overall (dysarthric speakers pooled)\n",
                 overall.drop(columns="all").to_markdown(index=False), "\n"]

    infos = [json.loads(p.read_text()) for p in Path(args.outputs).rglob("run_info.json")]
    if infos:
        params = (pd.DataFrame(infos).groupby("system")
                  .agg(trainable_params=("trainable_params", "first"), total_params=("total_params", "first"),
                       mean_train_minutes=("train_minutes", "mean"), folds=("test_speaker", "nunique")))
        params["trainable_%"] = (100 * params.trainable_params / params.total_params).round(2)
        params = params.round(1)
        for col in ("trainable_params", "total_params"):
            params[col] = params[col].map("{:,}".format)
        sections += ["\n## Trainable parameters\n", params.to_markdown(), "\n"]

    if args.compare:
        cmp = paired_compare(df, *args.compare, args.n_boot, rng)
        cmp.to_csv(out / "paired_comparison.csv", index=False)
        sections += [f"\n## Paired bootstrap: {args.compare[0]} (A) vs {args.compare[1]} (B)\n",
                     cmp.to_markdown(index=False), "\n"]

    curve = personalisation_curve(df, out)
    if curve is not None:
        sections += ["\n## Personalisation curve (WER% vs adaptation minutes)\n", curve.to_markdown(), "\n"]

    (out / "report.md").write_text("\n".join(sections))
    print("\n".join(sections))
    print(f"\nreport written to {out / 'report.md'}")


if __name__ == "__main__":
    main()
