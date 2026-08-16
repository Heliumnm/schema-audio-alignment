"""Teacher check — the GPU pass. Produces raw scores only; it decides nothing.

For each recording in the frozen manifest, and for each prompt in `teacher_prompts`, and
for each of the two option orders, read the model's probability mass on the two option
letters at the **first** answer position and record

    p_wet = P(letter meaning wet) / (P(A) + P(B))

Scoring, gating and the permutation control all live in `teacher_score.py`, which needs no
GPU. Keeping them apart means the gate can be recomputed without re-running the model, and
means this script cannot be quietly edited into one that also decides the outcome.

Arms produced here:
  intact    the real audio
  silence   a zero waveform of the same length -- secondary, reveals the model's prior

`audio_shuffled` is NOT here: it is an offline permutation of the intact scores within
annotator x quality x duration tercile, so it costs no forward passes.

    python src/teacher_run.py --manifest results/teacher_split_manifest.csv
"""
import os, json, argparse
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_prompts import PROMPTS, render, positive_letter


def main():
    import torch
    import soundfile as sf
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/teacher_split_manifest.csv")
    ap.add_argument("--config", default="results/teacher_run_config.json")
    ap.add_argument("--out", default="results/teacher_scores.csv")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    M = pd.read_csv(args.manifest)
    recs = sorted(M.uuid.unique())
    print(f"{len(M)} annotator rows over {len(recs)} recordings")
    print(f"model {cfg['model']}")

    proc = AutoProcessor.from_pretrained(cfg["model"])
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        cfg["model"], torch_dtype=torch.float16, device_map="cuda").eval()
    tok = proc.tokenizer

    # the two option letters, as the model would emit them at the first answer position
    ids = {}
    for L in ("A", "B"):
        cand = [tok.encode(v, add_special_tokens=False) for v in (L, " " + L)]
        ids[L] = [c[0] for c in cand if len(c) >= 1]
    print(f"option token ids: {ids}")

    rows = []
    for n, u in enumerate(recs):
        wav = os.path.join(cfg["wav_dir"], u + ".wav")
        y, sr = sf.read(wav, dtype="float32")
        for arm in ("intact", "silence"):
            a = np.zeros_like(y) if arm == "silence" else y
            for pid in PROMPTS:
                for swap in (False, True):
                    text = render(pid, swap)
                    conv = [{"role": "user", "content": [
                        {"type": "audio", "audio_url": wav}, {"type": "text", "text": text}]}]
                    chat = proc.apply_chat_template(conv, add_generation_prompt=True,
                                                   tokenize=False)
                    inp = proc(text=chat, audio=[a], sampling_rate=sr,
                              return_tensors="pt", padding=True)
                    inp = {k: (v.to("cuda") if hasattr(v, "to") else v)
                           for k, v in inp.items()}
                    with torch.no_grad():
                        logits = model(**inp).logits[0, -1].float()
                    p = torch.softmax(logits, -1)
                    pa = float(sum(p[i] for i in ids["A"]))
                    pb = float(sum(p[i] for i in ids["B"]))
                    tot = pa + pb
                    pos = positive_letter(swap)          # letter meaning 'wet'
                    p_wet = (pa if pos == "A" else pb) / tot if tot > 0 else np.nan
                    rows.append({"uuid": u, "arm": arm, "prompt": pid, "swap": swap,
                                 "p_A": pa, "p_B": pb, "mass_on_options": tot,
                                 "p_wet": p_wet})
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(recs)}", flush=True)

    S = pd.DataFrame(rows)
    S.to_csv(args.out, index=False)
    comp = float((S.mass_on_options >= 0.05).mean())
    print(f"\nwrote {args.out}  rows={len(S)}")
    print(f"format compliance (>=5% mass on the two option tokens): {comp:.4f}")
    print("Scoring and the gate are in teacher_score.py; this script decides nothing.")


if __name__ == "__main__":
    main()
