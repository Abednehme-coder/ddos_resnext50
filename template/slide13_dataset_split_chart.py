#!/usr/bin/env python3
"""
Generate Slide 13 dataset split chart as PNG/SVG.

Usage:
  python template/slide13_dataset_split_chart.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt


def main() -> None:
    # Update these values to your exact counts if needed.
    counts = {
        "Train": 80,
        "Validation": 10,
        "Test": 10,
        "Blind": 270,
    }

    labels = list(counts.keys())
    values = list(counts.values())
    colors = ["#1f6feb", "#00bcd4", "#5e60ce", "#ff8f00"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="#0b1f44", linewidth=1.0)

    ax.set_title("Dataset Partition Overview", fontsize=14, weight="bold")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    # Add ratio note from splitting script defaults.
    ax.text(
        0.02,
        0.96,
        "Train/Val/Test default ratio: 80/10/10",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#0b1f44",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#eef6ff", "edgecolor": "#bfd7ff"},
    )

    fig.tight_layout()
    fig.savefig("template/slide13_dataset_split_chart.png", dpi=220)
    fig.savefig("template/slide13_dataset_split_chart.svg")
    print("Created template/slide13_dataset_split_chart.png and .svg")


if __name__ == "__main__":
    main()
