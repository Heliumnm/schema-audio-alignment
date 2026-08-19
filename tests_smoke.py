import sys, json, types
# stub heavy deps: we are testing the provenance/verbalizer logic only
for m in ["torch", "torch.nn", "soundfile", "librosa", "librosa.feature",
          "scipy", "scipy.signal", "scipy.stats", "numpy"]:
    mod = types.ModuleType(m); sys.modules.setdefault(m, mod)
sys.modules["torch"].nn = sys.modules["torch.nn"]
sys.modules["torch.nn"].Module = type("Module", (), {"__init__": lambda self: None})
sys.modules["torch.nn"].Linear = lambda *a, **k: None
sys.modules["scipy.signal"].find_peaks = None
sys.modules["scipy.stats"].kurtosis = None
sys.modules["numpy"].percentile = None; sys.modules["numpy"].array = None
sys.modules["numpy"].isfinite = None

sys.path.insert(0, "src")
from schema_text import build_schema, verbalize, verbalize_explained, CONDITIONS

entry = {"filename": "101_1b1_Al_sc_Meditron.wav", "duration": 2.31, "condition": "clean",
         "crackle": 1, "wheeze": 0, "label": "crackle", "split": "train"}
cues = {"hf_kurtosis": 5.0, "transient_pitch": 3000.0, "wheeze_band": 0.4,
        "wheeze_n_peaks": 2, "wheeze_pitch": 900.0, "breath_intensity": 0.02,
        "crackle_conf": 0.85, "wheeze_conf": 0.5}
thr = {"crackle_conf": (0.3, 0.7), "wheeze_conf": (0.3, 0.7),
       "hf_kurtosis": (1.0, 3.0), "transient_pitch": (1000., 2000.),
       "wheeze_band": (0.1, 0.3), "wheeze_pitch": (400., 800.),
       "breath_intensity": (0.05, 0.15)}

s = build_schema(entry, cues, thr)
for c in CONDITIONS:
    print(f"--- [{c}] ---"); print(verbalize(s, c)); print()

print("=== explained / dataset ===");  print(verbalize_explained(s, "dataset")); print()
print("=== explained / model ===");    print(verbalize_explained(s, "model"));   print()

texts = [verbalize(s, c) for c in CONDITIONS] + [verbalize_explained(s, c) for c in CONDITIONS]
assert all("expert" not in t.lower() for t in texts), "LEAK"
print("LEAK CHECK: pass")
print("bookkeeping block (never rendered):", json.dumps(s["expert_annotation"]))
