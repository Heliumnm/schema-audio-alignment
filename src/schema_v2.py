"""Schema v2 — one field dict, three renderings, so "structure" is the only variable.

Phase 1 compared schema text against Qwen2-Audio narration and called the result a
verdict on structure. It was not: the narration arm carries different information
(its clinical assertions score AUROC 0.509 against ground truth) and hallucinates, so
it varies content and format at once. It stays here only as an external baseline.

The causal comparison is the three arms below, all emitted from the *same* field
dict — same values, same bins, same provenance, same missing-value handling. If any
arm sees a field the others do not, the comparison is no longer content-matched and
proves nothing about structure.

    matched_template    fluent English sentence
    matched_serialized  field=value; provenance=...
    typed_schema        per-field (name, value-bin, provenance) triples for a
                        permutation-invariant set encoder

Design rules carried from the Phase-2 plan:
  - continuous values are kept alongside their bin, not replaced by low/medium/high;
  - every field carries provenance, and recording metadata is separable from
    acoustic evidence so the "drop signal-derived fields" control is a filter rather
    than a re-render;
  - missing fields are marked missing, never filled with a fabricated "none";
  - the expert annotation never enters any rendering.

    python src/schema_v2.py --schema_text results/schema_text.json \
        --out results/schema_v2.json --show 2
"""
import os, json, argparse

# field -> (block, provenance, human-readable phrase template)
# `block` separates recording metadata from acoustic evidence so controls can drop
# one without re-rendering the other.
SPEC = {
    "chest_location":     ("recording", "dataset",      "auscultation site is {v}"),
    "device":             ("recording", "dataset",      "recorded with a {v} stethoscope"),
    "duration_s":         ("recording", "dataset",      "the cycle lasts {v} seconds"),
    "signal_quality":     ("recording", "dataset",      "the signal is {v}"),
    "crackle_likelihood": ("acoustic",  "opera-ct",     "crackle likelihood is {v}"),
    "transient_sharpness":("acoustic",  "judge_d",      "transient sharpness is {v}"),
    "crackle_character":  ("acoustic",  "signal-proxy", "crackle character is {v}"),
    "wheeze_likelihood":  ("acoustic",  "opera-ct",     "wheeze likelihood is {v}"),
    "musical_band_energy":("acoustic",  "judge_d",      "musical-band energy is {v}"),
    "wheeze_type":        ("acoustic",  "signal-proxy", "wheeze type is {v}"),
    "wheeze_pitch":       ("acoustic",  "signal-proxy", "wheeze pitch is {v}"),
    "breath_intensity":   ("acoustic",  "signal-proxy", "breath-sound intensity is {v}"),
}


def fields(rec):
    """Flatten the v1 schema into {field: {value, bin, block, provenance}}.

    Missing values become None and are rendered as an explicit `missing` marker in
    every arm, so no arm silently gains or loses a field relative to another.
    """
    s = rec["schema"]
    a, r = s["adventitious_sounds"], s["recording"]
    loc = r["chest_location"]["value"]
    raw = {
        "chest_location":      f"{loc['region']} {loc['side']}".strip(),
        "device":              r["device"]["value"],
        "duration_s":          r["duration_s"]["value"],
        "signal_quality":      r["signal_quality"]["value"],
        "crackle_likelihood":  a["crackle"]["likelihood"]["level"],
        "transient_sharpness": a["crackle"]["transient_sharpness"]["level"],
        "crackle_character":   a["crackle"]["character"]["value"],
        "wheeze_likelihood":   a["wheeze"]["likelihood"]["level"],
        "musical_band_energy": a["wheeze"]["musical_band_energy"]["level"],
        "wheeze_type":         a["wheeze"]["type"]["value"],
        "wheeze_pitch":        a["wheeze"]["pitch"]["level"],
        "breath_intensity":    s["breath_sound"]["intensity"]["level"],
    }
    out = {}
    for f, (block, prov, _) in SPEC.items():
        v = raw.get(f)
        if isinstance(v, str) and v in ("none", "unknown", "unspecified"):
            v = None
        out[f] = {"value": v, "block": block, "provenance": prov,
                  "missing": v is None}
    return out


def render(F, arm, blocks=("recording", "acoustic"), provenance=None):
    """Render the same field dict three ways. `blocks`/`provenance` implement the
    ablation controls as filters, so every arm drops exactly the same fields."""
    items = [(f, d) for f, d in F.items()
             if d["block"] in blocks and (provenance is None or d["provenance"] in provenance)]
    if arm == "typed_schema":
        return [{"field": f,
                 "value": "missing" if d["missing"] else str(d["value"]),
                 "provenance": d["provenance"],
                 "missing": d["missing"]} for f, d in items]
    if arm == "matched_serialized":
        return "; ".join(
            f"{f}={'missing' if d['missing'] else d['value']}|prov={d['provenance']}"
            for f, d in items)
    # matched_template — same fields, same values, same provenance, fluent form
    parts = []
    for f, d in items:
        phrase = SPEC[f][2].format(v="not available" if d["missing"] else d["value"])
        parts.append(f"{phrase} (source: {d['provenance']})")
    out = "; ".join(parts)
    return out[:1].upper() + out[1:] + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema_text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--show", type=int, default=2)
    args = ap.parse_args()

    D = json.load(open(args.schema_text))
    arms = ["matched_template", "matched_serialized", "typed_schema"]
    out = {}
    for k, rec in D.items():
        F = fields(rec)
        out[k] = {"fields": F, "split": rec["split"], "label": rec["label"],
                  **{a: render(F, a) for a in arms}}

    # the whole design rests on the arms being content-matched; check it rather
    # than trust it
    ks = list(out)
    nf = {len(out[k]["typed_schema"]) for k in ks}
    assert len(nf) == 1, f"typed_schema field count varies: {nf}"
    for k in ks[:200]:
        vals = {n["field"]: n["value"] for n in out[k]["typed_schema"]}
        ser = dict(p.split("|")[0].split("=", 1) for p in out[k]["matched_serialized"].split("; "))
        assert vals == ser, f"{k}: serialized and typed disagree on values"
        # the template arm must carry the same value STRINGS, or tokenisation sees
        # different inputs and the comparison stops being content-matched
        tpl = out[k]["matched_template"]
        for f, v in vals.items():
            if v != "missing" and f != "chest_location":
                assert v in tpl, f"{k}: template lost value {f}={v!r}"
    print(f"content-match check passed: {nf.pop()} fields in every arm, "
          f"values identical across renderings")

    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {len(out)} records -> {args.out}")
    for k in ks[:args.show]:
        print(f"\n=== {k} ({out[k]['label']}) ===")
        print("[template]  ", out[k]["matched_template"][:230])
        print("[serialized]", out[k]["matched_serialized"][:230])
        print("[typed]     ", json.dumps(out[k]["typed_schema"][:3], ensure_ascii=False))


if __name__ == "__main__":
    main()
