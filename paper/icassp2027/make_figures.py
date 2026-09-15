#!/usr/bin/env python3
"""Build the two vector figures used by the ICASSP manuscript.

Only the Python standard library and ReportLab are required. This keeps the
Overleaf source package independent of plotting-library versions.
"""

from __future__ import annotations

import csv
from pathlib import Path

import reportlab
from reportlab.lib.colors import Color, HexColor, black, white
from reportlab.lib.pagesizes import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

REPORTLAB_FONTS = Path(reportlab.__file__).resolve().parent / "fonts"
FONT = "Vera"
FONT_BOLD = "VeraBd"
pdfmetrics.registerFont(TTFont(FONT, str(REPORTLAB_FONTS / "Vera.ttf")))
pdfmetrics.registerFont(TTFont(FONT_BOLD, str(REPORTLAB_FONTS / "VeraBd.ttf")))

BLUE = HexColor("#2864A4")
ORANGE = HexColor("#D47A16")
GREEN = HexColor("#2F7D55")
GREY = HexColor("#666666")
LIGHT = HexColor("#F2F2F2")


def box(c: canvas.Canvas, x: float, y: float, w: float, h: float,
        color: Color, title: str, lines: list[str]) -> None:
    c.setStrokeColor(color)
    c.setFillColor(white)
    c.setLineWidth(1.0)
    c.roundRect(x, y, w, h, 6, stroke=1, fill=1)
    c.setFillColor(color)
    c.setFont(FONT_BOLD, 7.8)
    c.drawCentredString(x + w / 2, y + h - 14, title)
    c.setFillColor(black)
    c.setFont(FONT, 6.9)
    line_y = y + h - 29
    for line in lines:
        c.drawCentredString(x + w / 2, line_y, line)
        line_y -= 10


def arrow(c: canvas.Canvas, x1: float, y: float, x2: float) -> None:
    c.setStrokeColor(GREY)
    c.setFillColor(GREY)
    c.setLineWidth(1.0)
    c.line(x1, y, x2, y)
    c.line(x2, y, x2 - 5, y + 3)
    c.line(x2, y, x2 - 5, y - 3)


def delta_term(c: canvas.Canvas, center_x: float, y: float,
               suffix: str, contrast: str) -> None:
    """Draw a vector Delta followed by a readable, compact contrast label."""
    size = 6.9
    tail = f"{suffix} = {contrast}"
    delta_w = 7.0
    tail_w = stringWidth(tail, FONT_BOLD, size)
    gap = 0.7
    x = center_x - (delta_w + gap + tail_w) / 2
    c.setFillColor(black)
    c.setStrokeColor(black)
    c.setLineWidth(0.7)
    path = c.beginPath()
    path.moveTo(x, y)
    path.lineTo(x + delta_w / 2, y + size)
    path.lineTo(x + delta_w, y)
    path.close()
    c.drawPath(path, stroke=1, fill=0)
    c.setFont(FONT_BOLD, size)
    c.drawString(x + delta_w + gap, y, tail)


def pairing_figure() -> None:
    width, height = 7.05 * inch, 1.42 * inch
    c = canvas.Canvas(
        str(OUT / "pairing_audit.pdf"),
        pagesize=(width, height),
        initialFontName=FONT,
        invariant=1,
    )
    margin = 6
    gap = 5
    arm_w = 78
    eval_w = 120
    y = 29
    h = 58
    xs = [margin + i * (arm_w + gap) for i in range(4)]
    box(c, xs[0], y, arm_w, h, BLUE, "CORRECT", ["same person", "same label + sex"])
    box(c, xs[1], y, arm_w, h, GREEN, "LABEL+SEX", ["different person", "same label + sex"])
    box(c, xs[2], y, arm_w, h, ORANGE, "WITHIN-LABEL", ["different person", "same label"])
    box(c, xs[3], y, arm_w, h, GREY, "GLOBAL", ["different person", "no restriction"])
    arrow(c, xs[3] + arm_w + 5, y + h / 2, width - eval_w - 11)
    box(
        c,
        width - eval_w - 7,
        18,
        eval_w,
        76,
        GREEN,
        "THREE QUESTIONS",
        ["1. Pairing?  Retrieval", "2. Retained?  Probes", "3. Transfer?  Matched disease"],
    )
    delta_term(c, (xs[0] + xs[1] + arm_w) / 2, 12, "profile", "C-Wys")
    delta_term(c, (xs[1] + xs[2] + arm_w) / 2, 12, "sex", "Wys-W")
    delta_term(c, (xs[2] + xs[3] + arm_w) / 2, 12, "label", "W-G")
    c.setFillColor(black)
    c.setFont(FONT, 5.8)
    c.drawCentredString((xs[0] + xs[1] + arm_w) / 2, 3, "Residual pairing")
    c.drawCentredString((xs[1] + xs[2] + arm_w) / 2, 3, "Sex consistency")
    c.drawCentredString((xs[2] + xs[3] + arm_w) / 2, 3, "Label association")
    c.showPage()
    c.save()


def draw_marker(c: canvas.Canvas, x: float, y: float, marker: str, color: Color) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color)
    size = 3.1
    if marker == "circle":
        c.circle(x, y, size, stroke=1, fill=1)
    elif marker == "square":
        c.rect(x - size, y - size, 2 * size, 2 * size, stroke=1, fill=1)
    elif marker == "diamond":
        path = c.beginPath()
        path.moveTo(x, y + size + 0.5)
        path.lineTo(x + size + 0.5, y)
        path.lineTo(x, y - size - 0.5)
        path.lineTo(x - size - 0.5, y)
        path.close()
        c.drawPath(path, stroke=1, fill=1)
    else:
        path = c.beginPath()
        path.moveTo(x, y + size + 0.7)
        path.lineTo(x + size + 0.7, y - size)
        path.lineTo(x - size - 0.7, y - size)
        path.close()
        c.drawPath(path, stroke=1, fill=1)


def forest_figure() -> None:
    with (ROOT / "figure_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    width, height = 7.05 * inch, 3.03 * inch
    c = canvas.Canvas(
        str(OUT / "cross_dataset_forest.pdf"),
        pagesize=(width, height),
        initialFontName=FONT,
        invariant=1,
    )
    left = 103
    right = 7
    top = 29
    bottom = 24
    panel_gap = 21
    panel_w = (width - left - right - 2 * panel_gap) / 3
    row_h = (height - top - bottom) / len(rows)
    colors = {"AST-6L": BLUE, "OPERA-CT": ORANGE, "HeAR": GREEN}
    markers = {"discovery": "circle", "secondary": "square", "reconstructed": "diamond", "post-hoc": "triangle"}
    panels = [
        ("retrieval", "Profile retrieval", "Delta MRR (C-W)", -0.012, 0.112),
        ("sex", "Sex probe", "Delta AUROC (C-W)", -0.018, 0.245),
        ("disease", "Matched disease", "Delta AUROC (C-Wys)", -0.165, 0.125),
    ]

    for idx, row in enumerate(rows):
        if row["dataset"] in {"CODA TB", "Coswara"}:
            y0 = height - top - (idx + 1) * row_h
            c.setFillColor(LIGHT)
            c.rect(0, y0, width, row_h, stroke=0, fill=1)

    c.setFillColor(black)
    c.setFont(FONT, 6.3)
    for idx, row in enumerate(rows):
        y = height - top - (idx + 0.5) * row_h
        label = f'{row["dataset"]} / {row["backbone"]}'
        c.drawRightString(left - 6, y - 2.2, label)

    for pidx, (metric, title, subtitle, xmin, xmax) in enumerate(panels):
        x0 = left + pidx * (panel_w + panel_gap)
        x1 = x0 + panel_w

        c.setFillColor(black)
        c.setFont(FONT_BOLD, 7.4)
        c.drawCentredString((x0 + x1) / 2, height - 10, title)
        c.setFont(FONT, 6.4)
        c.drawCentredString((x0 + x1) / 2, height - 19, subtitle)

        def scale(value: float) -> float:
            return x0 + (value - xmin) / (xmax - xmin) * panel_w

        zero_x = scale(0.0)
        c.setStrokeColor(HexColor("#777777"))
        c.setDash(2, 2)
        c.line(zero_x, bottom - 1, zero_x, height - top + 1)
        c.setDash()

        for tick in (xmin, 0.0, xmax):
            tx = scale(tick)
            c.setStrokeColor(HexColor("#DDDDDD"))
            c.line(tx, bottom, tx, height - top)
            c.setFillColor(black)
            c.setFont(FONT, 5.4)
            label = "0" if abs(tick) < 1e-12 else f"{tick:.2f}"
            c.drawCentredString(tx, 7, label)

        for idx, row in enumerate(rows):
            y = height - top - (idx + 0.5) * row_h
            value = float(row[metric])
            low = float(row[f"{metric}_low"])
            high = float(row[f"{metric}_high"])
            color = colors[row["backbone"]]
            c.setStrokeColor(color)
            c.setLineWidth(0.8)
            c.line(scale(low), y, scale(high), y)
            c.line(scale(low), y - 2, scale(low), y + 2)
            c.line(scale(high), y - 2, scale(high), y + 2)
            draw_marker(c, scale(value), y, markers[row["status"]], color)

    legend_y = 15
    legend_x = left + 27
    c.setFont(FONT, 6.1)
    for name in ("AST-6L", "OPERA-CT", "HeAR"):
        draw_marker(c, legend_x, legend_y, "circle", colors[name])
        c.setFillColor(black)
        c.drawString(legend_x + 6, legend_y - 2.2, name)
        legend_x += 58

    c.showPage()
    c.save()


if __name__ == "__main__":
    pairing_figure()
    forest_figure()
