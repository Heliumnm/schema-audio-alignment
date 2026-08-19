#!/usr/bin/env python3
"""Generate the frozen UKCOVID figures used by the ICASSP draft.

The script reads only the committed formal result JSON.  It does not load participant
predictions, refit a model, inspect Coswara, or recompute confidence intervals.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "metadata_alignment_results__raw_disease_primary.json"
OUT = ROOT / "figures"
SPLITS = ("standard", "matched", "matched_long")
SPLIT_LABELS = {
    "standard": "Standard",
    "matched": "Matched",
    "matched_long": "Matched-long",
}


def esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def line(x1: float, y1: float, x2: float, y2: float, **attrs: object) -> str:
    values = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, **attrs}
    rendered = " ".join(f'{key}="{esc(value)}"' for key, value in values.items())
    return f"<line {rendered}/>"


def text(x: float, y: float, value: object, **attrs: object) -> str:
    values = {"x": x, "y": y, **attrs}
    rendered = " ".join(f'{key}="{esc(item)}"' for key, item in values.items())
    return f"<text {rendered}>{esc(value)}</text>"


def rect(x: float, y: float, width: float, height: float, **attrs: object) -> str:
    values = {"x": x, "y": y, "width": width, "height": height, **attrs}
    rendered = " ".join(f'{key}="{esc(item)}"' for key, item in values.items())
    return f"<rect {rendered}/>"


def svg_document(width: int, height: int, body: list[str], description: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{esc(description)}">',
            "<style>",
            "text{font-family:Helvetica,Arial,sans-serif;fill:#1f2933}",
            ".small{font-size:14px}.label{font-size:14px}.title{font-size:16px;font-weight:700}",
            ".axis{stroke:#5f6b76;stroke-width:1}.grid{stroke:#d7dde3;stroke-width:1}",
            ".zero{stroke:#27313a;stroke-width:1.2;stroke-dasharray:4 3}",
            ".ci{stroke:#2463a6;stroke-width:2}.point{fill:#2463a6;stroke:#163f6b;stroke-width:1}",
            ".box{fill:#f5f7f9;stroke:#687682;stroke-width:1}",
            ".correct{stroke:#2463a6;stroke-width:2.4}.within{stroke:#c26d15;stroke-width:2.4;stroke-dasharray:6 3}",
            ".global{stroke:#687682;stroke-width:2.4;stroke-dasharray:2 3}",
            "</style>",
            f"<desc>{esc(description)}</desc>",
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
            *body,
            "</svg>",
            "",
        ]
    )


def make_pairing_design() -> None:
    width, height = 780, 330
    body: list[str] = []
    body.append(text(20, 25, "Pairing-controlled audit under cohort shift", **{"class": "title"}))

    boxes = [
        (22, 50, 110, 42, "Cough audio", ""),
        (160, 50, 110, 42, "AST-6L", "frozen"),
        (298, 50, 128, 42, "Projector", "trainable"),
        (454, 50, 72, 42, "z_audio", ""),
        (22, 116, 110, 42, "Metadata", "schema text"),
        (160, 116, 110, 42, "Phi-2", "frozen"),
        (454, 116, 72, 42, "z_text", ""),
    ]
    for x, y, w, h, first, second in boxes:
        body.append(rect(x, y, w, h, rx=4, **{"class": "box"}))
        body.append(text(x + w / 2, y + (18 if second else 26), first, **{"class": "label", "text-anchor": "middle"}))
        if second:
            body.append(text(x + w / 2, y + 34, second, **{"class": "small", "text-anchor": "middle"}))
    for y, starts in ((71, ((132, 160), (270, 298), (426, 454))), (137, ((132, 160), (270, 454)))):
        for start, end in starts:
            body.append(line(start, y, end, y, **{"class": "axis"}))

    body.append(text(22, 190, "Training pair construction", **{"class": "title"}))

    rows = [
        (218, "Correct", "z_audio(i) <-> z_text(i)", "correct"),
        (252, "Within-label shuffle", "z_audio(i) <-> z_text(j),  y_i = y_j", "within"),
        (286, "Global shuffle", "z_audio(i) <-> z_text(k), unrestricted", "global"),
    ]
    for y, label, explanation, style in rows:
        body.append(line(30, y, 116, y, **{"class": style}))
        body.append(text(128, y + 4, label, **{"class": "label"}))
        body.append(text(274, y + 4, explanation, **{"class": "small"}))

    body.append(rect(590, 48, 168, 82, rx=4, **{"class": "box"}))
    body.append(text(674, 71, "Standard source domain", **{"class": "label", "text-anchor": "middle"}))
    body.append(text(674, 90, "recruitment -> COVID", **{"class": "small", "text-anchor": "middle"}))
    body.append(text(674, 112, "AUROC 0.9966", **{"class": "title", "text-anchor": "middle"}))

    body.append(rect(590, 196, 168, 82, rx=4, **{"class": "box"}))
    body.append(text(674, 219, "Matched domain", **{"class": "label", "text-anchor": "middle"}))
    body.append(text(674, 238, "recruitment -> COVID", **{"class": "small", "text-anchor": "middle"}))
    body.append(text(674, 260, "AUROC 0.5000", **{"class": "title", "text-anchor": "middle"}))
    body.append(line(674, 130, 674, 196, **{"class": "zero"}))

    body.append(text(22, 318, "Correct - within-label isolates participant correspondence while retaining label-level co-occurrence.", **{"class": "small"}))
    output = OUT / "icassp_pairing_design.svg"
    output.write_text(
        svg_document(width, height, body, "Correct, within-label-shuffled, and globally-shuffled audio-metadata alignment evaluated under source and matched cohorts."),
        encoding="utf-8",
    )


def panel(
    x0: int,
    title_value: str,
    x_label: str,
    values: dict[str, dict[str, object]],
    domain: tuple[float, float],
) -> list[str]:
    body: list[str] = []
    left, right = x0 + 112, x0 + 344
    top, bottom = 58, 245
    low, high = domain

    def sx(value: float) -> float:
        return left + (value - low) / (high - low) * (right - left)

    body.append(text(x0 + 8, 25, title_value, **{"class": "title"}))
    ticks = [low + index * (high - low) / 4 for index in range(5)]
    for tick in ticks:
        x = sx(tick)
        body.append(line(x, top, x, bottom, **{"class": "grid"}))
        body.append(text(x, bottom + 20, f"{tick:+.02f}", **{"class": "small", "text-anchor": "middle"}))
    body.append(line(sx(0), top, sx(0), bottom, **{"class": "zero"}))
    body.append(line(left, bottom, right, bottom, **{"class": "axis"}))
    body.append(text((left + right) / 2, bottom + 42, x_label, **{"class": "small", "text-anchor": "middle"}))

    for index, split in enumerate(SPLITS):
        y = 88 + index * 58
        item = values[split]
        observed = float(item["observed"])
        ci_low, ci_high = (float(value) for value in item["ci"])
        body.append(text(left - 10, y + 4, SPLIT_LABELS[split], **{"class": "label", "text-anchor": "end"}))
        body.append(line(sx(ci_low), y, sx(ci_high), y, **{"class": "ci"}))
        body.append(line(sx(ci_low), y - 5, sx(ci_low), y + 5, **{"class": "ci"}))
        body.append(line(sx(ci_high), y - 5, sx(ci_high), y + 5, **{"class": "ci"}))
        body.append(f'<circle cx="{sx(observed):.2f}" cy="{y}" r="4.5" class="point"/>')
        anchor = "start" if observed <= 0 else "end"
        offset = 7 if observed <= 0 else -7
        body.append(text(sx(observed) + offset, y - 10, f"{observed:+.03f}", **{"class": "small", "text-anchor": anchor}))
    return body


def make_forest(result: dict[str, object]) -> None:
    comparison = result["disease"]["comparisons"]["raw"]["correct_minus_within_label"]
    auc = {split: comparison[split]["delta_auroc"] for split in SPLITS}
    nll = {split: comparison[split]["delta_neg_nll"] for split in SPLITS}
    body = [
        text(20, 300, "Positive favors correct; negative Delta(-NLL) means worse calibrated transfer.", **{"class": "small"})
    ]
    body.extend(panel(0, "A  Ranking", "Delta AUROC: correct - within-label", auc, (-0.05, 0.05)))
    body.extend(panel(390, "B  Calibration", "Delta(-NLL): correct - within-label", nll, (-0.06, 0.03)))
    output = OUT / "icassp_pairing_forest.svg"
    output.write_text(
        svg_document(780, 320, body, "Forest plot of correct minus within-label alignment for AUROC and calibrated negative log-likelihood across Standard, matched, and matched-long cohorts."),
        encoding="utf-8",
    )


def write_absolute_table(result: dict[str, object]) -> None:
    arms = result["disease"]["arms"]["raw"]
    rows = []
    for arm in ("raw_ast", "correct", "within_label", "global"):
        for split in SPLITS:
            item = arms[arm][split]
            rows.append(
                {
                    "arm": arm,
                    "split": split,
                    "n": item["n"],
                    "auroc": item["auroc"],
                    "auroc_ci_low": item["auroc_ci"][0],
                    "auroc_ci_high": item["auroc_ci"][1],
                    "nll": -float(item["neg_nll"]),
                    "nll_ci_low": -float(item["neg_nll_ci"][1]),
                    "nll_ci_high": -float(item["neg_nll_ci"][0]),
                }
            )
    output = OUT / "icassp_absolute_results.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    result = json.loads(RESULT.read_text())
    expected = "comparisons.raw.correct_minus_within_label.matched.delta_neg_nll"
    if result["disease"].get("primary_result_path") != expected:
        raise ValueError("formal primary-result pointer changed; refusing to redraw figures")
    OUT.mkdir(exist_ok=True)
    make_pairing_design()
    make_forest(result)
    write_absolute_table(result)
    print(f"wrote frozen ICASSP assets to {OUT}")


if __name__ == "__main__":
    main()
