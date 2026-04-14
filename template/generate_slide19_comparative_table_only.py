#!/usr/bin/env python3
"""
Generate Slide 19 comparative table only (no chart).

Outputs:
- template/slide19_comparative_table_only.png
- template/slide19_comparative_table_only.svg
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from textwrap import fill


def main() -> None:
    bg = "#0B2235"
    panel = "#102F45"
    edge = "#00C2C7"
    txt = "#EAF4FF"

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=bg)
    ax.set_facecolor(panel)
    ax.axis("off")

    columns = ["Method Family", "Strength", "Limitation", "Our Advantage"]
    raw_rows = [
        ["Rule-based", "Fast, simple", "Brittle under traffic variation", "Better pattern learning"],
        ["Handcrafted ML", "Explainable", "Heavy feature engineering", "Less manual engineering"],
        ["Prior image-CNN", "Strong concept", "Weaker deployment/reproducibility", "MindSpore + full pipeline"],
    ]

    # Wrap long fields and auto-fit column widths by content length.
    wrap_widths = [16, 18, 32, 32]
    rows = [
        [fill(cell, width=wrap_widths[i]) for i, cell in enumerate(row)]
        for row in raw_rows
    ]

    col_len = [len(columns[i]) for i in range(len(columns))]
    for row in raw_rows:
        for i, cell in enumerate(row):
            col_len[i] = max(col_len[i], len(cell))
    total = float(sum(col_len))
    col_widths = [max(0.14, l / total) for l in col_len]
    # Normalize to exactly 1.0 so table uses full width.
    s = sum(col_widths)
    col_widths = [w / s for w in col_widths]

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.3)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(edge)
        cell.set_linewidth(1.2)
        if r == 0:
            cell.set_facecolor("#0E2A3D")
            cell.get_text().set_color(txt)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(panel)
            cell.get_text().set_color(txt)

    ax.set_title("Comparative Analysis by Method Family", color=txt, fontsize=18, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig("template/slide19_comparative_table_only.png", dpi=260, facecolor=bg)
    fig.savefig("template/slide19_comparative_table_only.svg", facecolor=bg)
    print("Created template/slide19_comparative_table_only.png and .svg")


if __name__ == "__main__":
    main()
