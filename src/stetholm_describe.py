"""
Generate free-form clinical descriptions of respiratory audio  ->  text condition T1.

T1 is the "free-form LLM narration" arm of the text-source comparison: an
audio-language model listens to the cycle and writes a clinical description. It is
maximally *derived* — the text is a function of the audio — which is exactly why it
is in the ablation. Compare against `schema_text.py` conditions (dataset/signal/
model/all).

Backends
--------
  stetholm  nvidia-free default: askyishan/StethoLM (COLA encoder + MedGemma-4B-IT).
            Custom architecture, so it needs trust_remote_code and possibly the
            authors' repo on PYTHONPATH. VERIFY the call signature against
            github.com/yishani/StethoLM before a long run — this loader is written
            defensively but has not been executed against the real checkpoint.
  qwen2     Qwen2-Audio-7B-Instruct, already working in the fidelity-oracle env.
            Use as fallback so the pipeline is never blocked on StethoLM.

Resumable: appends one JSON object per line and skips ids already present, so an
interrupted run continues where it stopped.

    python src/stetholm_describe.py \
        --manifest data/segments/manifest.json \
        --out results/t1_descriptions.jsonl \
        --backend stetholm --limit 20        # eyeball 20 first, then drop --limit
"""
import os, sys, json, time, argparse
import soundfile as sf
import librosa

SR = 16000

PROMPT = (
    "You are listening to a single respiratory cycle recorded with a stethoscope. "
    "Describe what you hear in two or three sentences, as a clinical auscultation "
    "note. Mention the character of the breath sounds and any adventitious sounds "
    "(crackles, wheezes) including their timing, pitch and quality. Describe only "
    "what is audible; do not state a diagnosis."
)


def load_audio(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


# --------------------------------------------------------------- backends
class StethoLM:
    """askyishan/StethoLM — custom COLA + MedGemma architecture."""
    name = "stetholm"

    def __init__(self, model_id="askyishan/StethoLM", device="cuda",
                 dtype="bfloat16", load_4bit=False):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, device_map="auto",
            torch_dtype=getattr(torch, dtype),
        ).eval()
        self.load_4bit = load_4bit

    def describe(self, y, max_new_tokens=160):
        inputs = self.processor(text=PROMPT, audio=y, sampling_rate=SR,
                                return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        n = inputs["input_ids"].shape[1]
        return self.processor.decode(out[0][n:], skip_special_tokens=True).strip()

    def describe_batch(self, ys, max_new_tokens=128):
        return [self.describe(y, max_new_tokens) for y in ys]


class Qwen2Audio:
    """Fallback so W1 never blocks on a third-party checkpoint."""
    name = "qwen2"

    def __init__(self, model_id="Qwen/Qwen2-Audio-7B-Instruct", device="cuda",
                 dtype="bfloat16", load_4bit=False):
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        kw = {"device_map": "auto", "torch_dtype": getattr(torch, dtype)}
        if load_4bit:
            # the lab GPUs are shared and often have <16 GB free; NF4 fits a 7B in ~6 GB
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, **kw).eval()
        # left padding: with mixed prompt lengths, right padding makes generate()
        # continue from pad tokens and truncates short items
        self.processor.tokenizer.padding_side = "left"

    def describe_batch(self, ys, max_new_tokens=128):
        conv = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "x.wav"}, {"type": "text", "text": PROMPT}]}]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True,
                                                  tokenize=False)
        inputs = self.processor(text=[text] * len(ys), audio=list(ys), sampling_rate=SR,
                                return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        n = inputs["input_ids"].shape[1]
        return [self.processor.decode(o[n:], skip_special_tokens=True).strip() for o in out]

    def describe(self, y, max_new_tokens=128):
        return self.describe_batch([y], max_new_tokens)[0]


BACKENDS = {"stetholm": StethoLM, "qwen2": Qwen2Audio}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="stetholm", choices=list(BACKENDS))
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--key", default="path")
    ap.add_argument("--audio_root", default="",
                    help="prefix for relative manifest paths")
    ap.add_argument("--limit", type=int, default=0, help="stop after N (0 = all)")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8,
                    help="batch size; sequential decoding is the bottleneck so this "
                         "is the main speed lever on a contended GPU")
    ap.add_argument("--load_4bit", action="store_true",
                    help="NF4 quantisation; needed when the shared GPUs are busy")
    args = ap.parse_args()

    seg = json.load(open(args.manifest))

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass   # tolerate a torn final line from a killed run
        print(f"resuming: {len(done)} already described")

    todo = [e for e in seg if os.path.basename(e.get(args.key, e["path"])) not in done]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to do"); return

    kw = {"load_4bit": args.load_4bit}
    if args.model_id:
        kw["model_id"] = args.model_id
    print(f"loading backend={args.backend} ...", flush=True)
    model = BACKENDS[args.backend](**kw)

    def resolve(e):
        p = e.get(args.key, e["path"])
        return p if (os.path.isabs(p) or not args.audio_root) else os.path.join(args.audio_root, p)

    t0, n_err, n_done = time.time(), 0, 0
    with open(args.out, "a") as f:
        for s0 in range(0, len(todo), args.batch):
            chunk = todo[s0:s0 + args.batch]
            paths = [resolve(e) for e in chunk]
            try:
                ys = [load_audio(p) for p in paths]
                descs = model.describe_batch(ys, args.max_new_tokens)
            except Exception as ex:
                n_err += len(chunk)
                print(f"  !! batch @{s0}: {type(ex).__name__}: {ex}", file=sys.stderr, flush=True)
                if s0 == 0:
                    # the first batch failing means a broken interface, not bad audio
                    raise
                continue
            for e, p, d in zip(chunk, paths, descs):
                f.write(json.dumps({"id": os.path.basename(p), "split": e.get("split"),
                                    "label": e.get("label"), "backend": model.name,
                                    "text": d}, ensure_ascii=False) + "\n")
            f.flush()
            n_done += len(chunk)
            if (s0 // args.batch) % 5 == 0:
                rate = n_done / (time.time() - t0)
                print(f"  {n_done}/{len(todo)}  {rate:.2f}/s  "
                      f"eta {(len(todo)-n_done)/max(rate,1e-6)/60:.1f} min", flush=True)

    print(f"\nwrote {len(todo) - n_err} descriptions -> {args.out} ({n_err} errors)")
    with open(args.out) as f:
        rows = [json.loads(l) for l in f]
    if rows:
        import statistics
        lens = [len(r["text"].split()) for r in rows]
        print(f"length: mean {statistics.mean(lens):.0f} words, "
              f"min {min(lens)}, max {max(lens)}")
        print("\n--- sample ---")
        for r in rows[:2]:
            print(f"[{r['label']}] {r['text'][:300]}\n")


if __name__ == "__main__":
    main()
