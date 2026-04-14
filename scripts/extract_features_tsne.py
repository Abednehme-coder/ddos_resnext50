#!/usr/bin/env python3
"""
Optional: t-SNE on penultimate-layer embeddings (not run by default).

Requires saving features from ResNeXt before the final classifier. The current
``mindcv`` forward path is not wrapped here; integrate a forward hook or a
custom Cell in ``train_resnext.py`` to export (N, D) features, then:

  sklearn.manifold.TSNE(n_components=2).fit_transform(X)

This script is a placeholder so the repository documents the intended workflow
without committing a heavy dependency on internal model hooks.
"""
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Placeholder for future t-SNE on embeddings.")
    parser.parse_args()
    print(
        "Not implemented: export penultimate features from ResNeXt in MindSpore, "
        "then run sklearn t-SNE / UMAP offline. See research/main.tex Appendix."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
