from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay updated math labels onto the single-column perturbation figure.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--out-pdf", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_img = Image.open(args.input).convert("RGB")
    width, height = base_img.size

    # Re-layout the figure on a fresh white canvas:
    # 1) shrink the plotted area slightly;
    # 2) reserve a clean left margin for two compact text columns.
    plot_scale = 0.90
    scaled_w = int(width * plot_scale)
    scaled_h = int(height * plot_scale)
    scaled_img = base_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (width, height), "white")
    paste_x = 220
    paste_y = (height - scaled_h) // 2
    canvas.paste(scaled_img, (paste_x, paste_y))

    dpi = 300
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas)
    ax.axis("off")

    fig.text(
        0.022,
        0.73,
        "Static Base",
        rotation=90,
        ha="center",
        va="center",
        fontsize=24,
    )
    fig.text(
        0.070,
        0.73,
        "$\\hat{F}_0(x)$",
        rotation=90,
        ha="center",
        va="center",
        fontsize=23,
    )
    fig.text(
        0.022,
        0.25,
        "Perturbation",
        rotation=90,
        ha="center",
        va="center",
        fontsize=24,
    )
    fig.text(
        0.070,
        0.25,
        "$M(x,\\mathbf{t})$",
        rotation=90,
        ha="center",
        va="center",
        fontsize=23,
    )

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=dpi, pad_inches=0)
    fig.savefig(args.out_pdf, dpi=dpi, pad_inches=0)
    plt.close(fig)
    print(f"Saved updated PNG to {args.out_png}")
    print(f"Saved updated PDF to {args.out_pdf}")


if __name__ == "__main__":
    main()
