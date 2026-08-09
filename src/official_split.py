"""
Build the ICBHI *official* train/test split, and say plainly where it differs from
the id-threshold split this project has been using.

The fidelity-oracle pipeline assigned `train` to patient ids 101-160 and `test` to
the rest, with a docstring claiming that was the official 60/40 partition. It is
not. The challenge ships an explicit per-recording file, and the two agree on only
53.5% of recordings.

Two facts to keep straight:

  * The official partition is 60/40 by RECORDING (540/381), not a patient-id cut.
    The id-threshold split lands at ~50/50 by cycle, so absolute numbers computed on
    it are not comparable to any published ICBHI result.
  * The official partition is NOT strictly patient-independent — a small number of
    patients appear on both sides. `--strict` drops those patients from test so the
    guarantee the rest of this codebase asserts still holds.

Emits {segment_id: "train"|"test"} for every segment in the manifest.

    python src/official_split.py \
        --manifest .../manifest.json \
        --official .../OPERA/datasets/icbhi/ICBHI_challenge_train_test.txt \
        --out results/split_official.json --strict
"""
import os, json, argparse, collections


def recording_of(seg_id):
    """101_1b1_Al_sc_Meditron_cycle000_normal[.wav] -> 101_1b1_Al_sc_Meditron."""
    stem = os.path.splitext(os.path.basename(seg_id))[0]
    return "_".join(stem.split("_")[:5])


def patient_of(seg_id):
    return os.path.basename(seg_id).split("_")[0]


def load_official(path):
    off = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                off[p[0]] = p[1].strip().lower()
    return off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--official", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strict", action="store_true",
                    help="drop patients appearing on both sides (from test) so the "
                         "split is genuinely patient-independent")
    args = ap.parse_args()

    off = load_official(args.official)
    man = json.load(open(args.manifest))
    print(f"official file: {len(off)} recordings "
          f"{dict(collections.Counter(off.values()))}")

    # patients on both sides of the official partition
    pat_side = collections.defaultdict(set)
    for rec, s in off.items():
        pat_side[rec.split("_")[0]].add(s)
    both = sorted(p for p, s in pat_side.items() if len(s) > 1)
    print(f"patients appearing in BOTH official train and test: {len(both)} {both}")

    out, missing, dropped = {}, [], 0
    seg_key = "path" if "path" in man[0] else "filename"
    for e in man:
        sid = os.path.basename(e[seg_key])
        rec = recording_of(sid)
        s = off.get(rec)
        if s is None:
            missing.append(rec)
            continue
        if args.strict and patient_of(sid) in both and s == "test":
            dropped += 1
            continue
        out[sid] = s

    cnt = collections.Counter(out.values())
    pats = collections.defaultdict(set)
    for sid, s in out.items():
        pats[s].add(patient_of(sid))
    overlap = pats["train"] & pats["test"]

    print(f"\nsegments assigned: {len(out)}/{len(man)}  {dict(cnt)}")
    if missing:
        print(f"  {len(missing)} segments had no official entry "
              f"(e.g. {sorted(set(missing))[:3]})")
    if args.strict:
        print(f"  --strict dropped {dropped} test segments from dual-side patients")
    print(f"patients: train {len(pats['train'])}, test {len(pats['test'])}, "
          f"overlap {len(overlap)}")
    if overlap:
        print(f"  WARNING patient overlap remains: {sorted(overlap)[:5]} — "
              f"rerun with --strict")

    # how far the id-threshold split we have been using actually is from official
    old = {os.path.basename(e[seg_key]): e["split"] for e in man}
    comp = [k for k in out if k in old]
    agree = sum(1 for k in comp if out[k] == old[k])
    print(f"\nagreement with the id-threshold split in the manifest: "
          f"{agree}/{len(comp)} ({100*agree/max(len(comp),1):.1f}%)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
