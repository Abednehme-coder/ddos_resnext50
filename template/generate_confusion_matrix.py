#!/usr/bin/env python3
"""
Generate a styled confusion matrix image for slides.

Output:
  - template/slide18_confusion_matrix.png
  - template/slide18_confusion_matrix.svg

Usage:
  python3 template/generate_confusion_matrix.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def main() -> None:
    # Test confusion matrix from reports/ICT_Innovation_Regional_Report.md
    # rows=true class, columns=predicted class
    cm = np.array(
        [
            [135, 0],     # true ddos   -> pred ddos, pred normal
            [35, 1675],   # true normal -> pred ddos, pred normal
        ]
    )

    row_labels = ["Actual: DDoS", "Actual: Normal"]
    col_labels = ["Predicted: DDoS", "Predicted: Normal"]

    # Theme colors matching your slides
    bg = "#0B2235"      # slide background
    panel = "#102F45"   # panel fill
    edge = "#00C2C7"    # cyan accent
    txt = "#EAF4FF"     # light text

    # Custom colormap aligned with presentation tones (navy -> cyan).
    synshield_cmap = LinearSegmentedColormap.from_list(
        "synshield",
        ["#13364F", "#1E5B75", "#00AFC1", "#6FE7EA"],
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.6), facecolor=bg)
    ax.set_facecolor(panel)

    im = ax.imshow(cm, cmap=synshield_cmap, vmin=0, vmax=float(cm.max()))

    # Axis labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, color=txt, fontsize=11)
    ax.set_yticklabels(row_labels, color=txt, fontsize=11)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True, colors=txt)

    # Cell text
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            color = "#0B2235" if value > 55 else txt
            ax.text(j, i, f"{int(value)}", ha="center", va="center", color=color, fontsize=13, weight="bold")

    # Title and frame
    ax.set_title("Confusion Matrix", color=txt, fontsize=16, weight="bold", pad=12)
    for spine in ax.spines.values():
        spine.set_edgecolor(edge)
        spine.set_linewidth(1.2)

    # Grid lines between cells
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color=edge, linestyle="-", linewidth=1.0, alpha=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count", color=txt)
    cbar.ax.yaxis.set_tick_params(color=txt)
    plt.setp(cbar.ax.get_yticklabels(), color=txt)
    cbar.outline.set_edgecolor(edge)

    fig.tight_layout()
    fig.savefig("template/slide18_confusion_matrix.png", dpi=240, facecolor=bg)
    fig.savefig("template/slide18_confusion_matrix.svg", facecolor=bg)
    print("Created template/slide18_confusion_matrix.png and .svg")


if __name__ == "__main__":
    main()
