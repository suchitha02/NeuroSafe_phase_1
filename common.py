"""Shared helpers for the TORGO ASR experiments (Stream 1: M0 and M1)."""
import re
from pathlib import Path

from tqdm import tqdm

SR = 16000

# Severity labels commonly used for TORGO in the dysarthric-ASR literature.
# Verify against the source you cite in your report before publishing results.
SEVERITY = {
    "F01": "severe", "M01": "severe", "M02": "severe", "M04": "severe",
    "F03": "moderate", "M05": "moderate",
    "F04": "mild", "M03": "mild",
}
DYSARTHRIC = sorted(SEVERITY)


def severity(speaker):
    return SEVERITY.get(speaker, "control")


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def clean_text(text):
    """Light cleaning for training labels: lowercase, letters/digits/apostrophes only."""
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_scoring_normalizer = None


def score_norm(text):
    """Whisper's English normaliser, applied to refs AND hyps of every system before WER,
    so Whisper and CTC models are scored identically ("one" vs "1", "it's" vs "it is", ...)."""
    global _scoring_normalizer
    if _scoring_normalizer is None:
        try:
            from transformers import WhisperTokenizer
            _scoring_normalizer = WhisperTokenizer.from_pretrained("openai/whisper-small").normalize
        except Exception as err:  # offline: fall back, but say so loudly
            print(f"WARNING: Whisper normaliser unavailable ({err}); using clean_text. "
                  "Do not mix reports produced with different normalisers.")
            _scoring_normalizer = clean_text
    return _scoring_normalizer(str(text))


def load_audio(path):
    """Read a mono 16 kHz file. TORGO is already 16 kHz mono (checked in the EDA), so no
    resampling is needed; this avoids librosa, which pulls in SciPy/scikit-learn."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr} Hz")
    return audio.mean(axis=1)


def save_predictions(df, hyps, confs, path, system, **extra):
    """One CSV per run; report.py reads every predictions.csv under outputs/."""
    out = df[["utt_id", "speaker", "severity", "utt_type", "text"]].copy()
    out["hyp"], out["conf"], out["system"] = hyps, confs, system
    for key, value in extra.items():
        out[key] = value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"saved {len(out)} predictions -> {path}")


def transcribe_whisper(model, processor, paths, batch_size=16):
    """Greedy decoding. Confidence = mean log-prob of the generated text tokens
    (re-scored with one teacher-forced pass, which is robust across transformers versions)."""
    import torch
    model.eval()
    eos = processor.tokenizer.eos_token_id  # Whisper special tokens all have ids >= eos
    lang = {"language": "en", "task": "transcribe"} if getattr(
        model.generation_config, "is_multilingual", False) else {}
    prompt_tokens = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"] if lang \
        else ["<|startoftranscript|>", "<|notimestamps|>"]
    prompt = torch.tensor(processor.tokenizer.convert_tokens_to_ids(prompt_tokens), device=model.device)
    hyps, confs = [], []
    for i in tqdm(range(0, len(paths), batch_size), desc="whisper"):
        audio = [load_audio(p) for p in paths[i:i + batch_size]]
        feats = processor.feature_extractor(audio, sampling_rate=SR, return_tensors="pt").input_features
        feats = feats.to(model.device, dtype=model.dtype)
        with torch.no_grad():
            seq = model.generate(input_features=feats, max_new_tokens=64, **lang)
            hyps += [h.strip() for h in processor.batch_decode(seq, skip_special_tokens=True)]
            # Newer transformers return only the generated tokens (no decoder prompt). Re-add the
            # prompt so rescoring is conditioned correctly and never gets an empty decoder input.
            if not (seq[:, 0] == prompt[0]).all():
                seq = torch.cat([prompt.expand(len(seq), -1), seq], dim=1)
            logits = model(input_features=feats, decoder_input_ids=seq[:, :-1]).logits.float()

        logp = logits.log_softmax(-1).gather(-1, seq[:, 1:, None]).squeeze(-1)
        text_tok = seq[:, 1:] < eos
        conf = (logp * text_tok).sum(1) / text_tok.sum(1).clamp(min=1)
        conf[text_tok.sum(1) == 0] = float("nan")  # empty output: no confidence
        confs += conf.cpu().tolist()
    return hyps, confs


def transcribe_ctc(model, processor, paths, batch_size=8):
    """Greedy CTC decoding. Confidence = mean max-softmax prob over non-blank frames."""
    import torch
    model.eval()
    blank = processor.tokenizer.pad_token_id
    hyps, confs = [], []
    for i in tqdm(range(0, len(paths), batch_size), desc="ctc"):
        audio = [load_audio(p) for p in paths[i:i + batch_size]]
        inputs = processor(audio, sampling_rate=SR, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            probs = model(**inputs).logits.float().softmax(-1)
        best_p, ids = probs.max(-1)
        hyps += [h.strip() for h in processor.batch_decode(ids)]
        keep = ids != blank
        conf = (best_p * keep).sum(1) / keep.sum(1).clamp(min=1)
        conf[keep.sum(1) == 0] = float("nan")
        confs += conf.cpu().tolist()
    return hyps, confs
