"""Metadata → schema text, with explicit maps and no silent fallback.

The first version had three silent mislabelling paths, all of the same shape — an
unrecognised value quietly became a plausible-looking one:

    AGE.get(v, "NA")                      an unknown age band became the string NA
    "CURRENT" if not Never and not Ex     "Prefer not to say" became CURRENT SMOKER,
                                          which is 505 participants asserted to smoke
    "YES" if v == 1 else "NO"             any unexpected value became NO

Every field now has an enumerated map and **an unknown value raises**. A dataset that
gains a category later fails loudly instead of being silently recoded.
"""
import numpy as np
import pandas as pd

UNIT = "participant_identifier"
AGE_MAP = {"18-44": "18-44", "45-64": "45-64", "65+": "65PLUS"}
SEX_MAP = {"Female": "FEMALE", "Male": "MALE"}
SMOKER_MAP = {"Never smoked": "NEVER", "Ex-smoker": "EX",
              "Current smoker (1 to 10 cigarettes per day)": "CURRENT_1_10",
              "Current smoker (11 or more cigarettes per day)": "CURRENT_11PLUS",
              "Current smoker (e-cigarettes or vapes only)": "CURRENT_ECIG",
              "Prefer not to say": "PREFER_NOT_TO_SAY"}
BINARY_MAP = {0: "NO", 1: "YES", 0.0: "NO", 1.0: "YES", False: "NO", True: "YES"}
MISSING = "[MISSING]"

FIELDS = [("AGE", "age", AGE_MAP), ("SEX", "gender", SEX_MAP),
          ("SMOKER", "smoker_status", SMOKER_MAP),
          ("ASTHMA", "respiratory_condition_asthma", BINARY_MAP),
          ("RESP_OTHER", "respiratory_condition_other", BINARY_MAP)]
EXCLUDED_SYMPTOMS = ("symptom_onset", "symptom_prefer_not_to_say")


def symptom_columns(df):
    return sorted(c for c in df.columns
                  if c.startswith("symptom_") and c not in EXCLUDED_SYMPTOMS)


def audit_values(df, sym_cols):
    """Enumerate what is actually in each column before anything is mapped."""
    report = {}
    for tag, col, m in FIELDS + [(c.replace("symptom_", "").upper(), c, BINARY_MAP)
                                 for c in sym_cols]:
        vals = df[col].dropna().unique()
        unknown = [v for v in vals if v not in m]
        report[tag] = {"column": col, "n_missing": int(df[col].isna().sum()),
                       "values": sorted(map(str, vals)), "unknown": sorted(map(str, unknown))}
    return report


def render(df, sym_cols):
    """Vectorised, and it raises on anything the maps do not cover."""
    cols = FIELDS + [(c.replace("symptom_", "").upper(), c, BINARY_MAP) for c in sym_cols]
    out = None
    for tag, col, m in cols:
        v = df[col]
        bad = v.dropna()[~v.dropna().isin(list(m))]
        if len(bad):
            raise ValueError(f"{col}: unmapped values {sorted(set(map(str, bad)))[:5]} — "
                             f"extend the map deliberately rather than defaulting")
        s = v.map(m).fillna(MISSING)
        piece = "[" + tag + "=" + s.astype(str) + "]"
        out = piece if out is None else out + " " + piece
    return out


def build(participant_csv, cohort_csv, splits_csv):
    p = pd.read_csv(participant_csv, low_memory=False)
    C = pd.read_csv(cohort_csv)
    s = pd.read_csv(splits_csv, low_memory=False)
    sym = symptom_columns(p)
    keep = [UNIT] + [c for _, c, _ in FIELDS] + sym
    d = C[[UNIT]].merge(p[keep], on=UNIT).merge(s[[UNIT, "splits"]], on=UNIT)
    order = {q: i for i, q in enumerate(C[UNIT])}
    d = d.sort_values(UNIT, key=lambda c: c.map(order)).reset_index(drop=True)
    audit = audit_values(d, sym)
    d["text"] = render(d, sym)
    return d, sym, audit


def expected_log_K(text_series, batch_size, n_draws=20000, seed=0):
    """The InfoNCE floor a duplicate-heavy corpus imposes, computed the way the sampler
    actually behaves: the average of log K over anchors, where K counts the identical
    texts in the anchor's own batch including itself.

    An earlier version reported log(E[K]), which is not the same quantity and is larger by
    Jensen. Those numbers are withdrawn.
    """
    counts = text_series.value_counts()
    n = len(text_series)
    freq = (counts / n).to_numpy()
    rs = np.random.RandomState(seed)
    anchors = rs.choice(len(freq), n_draws, p=freq)         # anchors drawn as they occur
    K = 1 + rs.binomial(batch_size - 1, freq[anchors])
    return float(np.mean(np.log(K))), float(np.mean(K))
