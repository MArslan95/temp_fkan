from __future__ import annotations

import re
from pathlib import Path

import matplotlib

# Non-interactive backend: suitable for saving figures from a script.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ============================================================
# Configuration
# ============================================================

ROOT = Path(r"D:\FKAN_Mamba\HC")

# The final files will be saved in:
# D:\FKAN_Mamba\HC\Composite_Figure
OUTPUT_DIR = ROOT / "Composite_Figure"
OUTPUT_NAME = "HC_classification_maps_2x5"

PANEL_FOLDERS = [
    ("GT", "GT"),
    ("HViT", "HViT"),
    ("SSFTTKAN", "SSFTTKAN"),
    ("HybridKAN", "HybridKAN"),
    ("SSViT", "SSViT"),
    ("DF", "DF"),
    ("SSMM", "SSMM"),
    ("MHM", "MHM"),
    ("WM", "WM"),
    ("FKANMamba", "FKANMamba"),
]

# IEEE-style font settings.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ============================================================
# File discovery
# ============================================================

def normalize_name(value: str) -> str:
    """Remove spaces and punctuation and convert to lowercase."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def resolve_folder(root: Path, requested_name: str) -> Path:
    """
    Find a method directory while ignoring capitalization,
    spaces, underscores, and hyphens.
    """
    if not root.exists():
        raise FileNotFoundError(
            f"Root directory does not exist:\n{root}"
        )

    requested_normalized = normalize_name(requested_name)

    matches = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and normalize_name(path.name) == requested_normalized
    ]

    if not matches:
        available = "\n".join(
            f"  - {path.name}"
            for path in sorted(root.iterdir())
            if path.is_dir()
        )

        raise FileNotFoundError(
            f"Method directory '{requested_name}' was not found in:\n"
            f"{root}\n\n"
            f"Available directories:\n{available}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple directories matched '{requested_name}':\n"
            + "\n".join(str(path) for path in matches)
        )

    return matches[0]


def candidate_score(path: Path, method_name: str) -> int:
    """
    Score candidate PNG files.

    Higher scores indicate that a file is more likely to be the
    required classification map.
    """
    filename = path.name.lower()
    compact = normalize_name(path.stem)
    method_compact = normalize_name(method_name)

    score = 0

    # Desired filename components.
    if "hc" in compact:
        score += 30

    if "groundtruths" in compact:
        score += 150
    elif "groundtruth" in compact:
        score += 140

    if "predictedgt" in compact:
        score += 130

    if method_compact in compact:
        score += 40

    # Give priority to the real GT map in the GT folder.
    if method_name == "GT":
        if compact.startswith("hcgroundtruth"):
            score += 250

        if "predicted" in compact:
            score -= 300

    # Reject irrelevant plots.
    excluded_tokens = (
        "accuracy",
        "loss",
        "curve",
        "confusion",
        "matrix",
        "history",
        "legend",
        "training",
        "validation",
        "comparison",
        "composite",
        "classificationmaps2x5",
    )

    if any(token in filename for token in excluded_tokens):
        score -= 1000

    return score


def resolve_classification_map(
    root: Path,
    method_name: str,
    folder_name: str,
) -> Path:
    """Locate the most likely classification-map PNG."""
    folder = resolve_folder(root, folder_name)

    all_pngs = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() == ".png"
    )

    if not all_pngs:
        raise FileNotFoundError(
            f"No PNG files were found inside:\n{folder}"
        )

    ranked = sorted(
        all_pngs,
        key=lambda path: (
            candidate_score(path, method_name),
            path.stat().st_size,
        ),
        reverse=True,
    )

    selected = ranked[0]
    selected_score = candidate_score(selected, method_name)

    if selected_score <= 0:
        available = "\n".join(
            f"  - {path.name}"
            for path in all_pngs
        )

        raise FileNotFoundError(
            f"No valid classification map was identified for "
            f"'{method_name}'.\n\n"
            f"PNG files found in {folder}:\n{available}"
        )

    print(f"{method_name:12s} -> {selected}")
    return selected


def resolve_all_panels() -> list[tuple[str, Path]]:
    """Resolve all ten panel images."""
    return [
        (
            method_name,
            resolve_classification_map(
                ROOT,
                method_name,
                folder_name,
            ),
        )
        for method_name, folder_name in PANEL_FOLDERS
    ]


# ============================================================
# Image loading
# ============================================================

def load_image(path: Path) -> np.ndarray:
    """
    Load an existing PNG without modifying or interpolating its
    classification labels.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image does not exist: {path}")

    try:
        with Image.open(path) as image:
            # RGB prevents unexpected palette or grayscale handling.
            return np.asarray(image.convert("RGB"))
    except Exception as error:
        raise RuntimeError(
            f"Could not read image:\n{path}\n\nReason: {error}"
        ) from error


# ============================================================
# Composite figure
# ============================================================

def create_figure(panels: list[tuple[str, Path]]) -> tuple[Path, Path]:
    """
    Create a two-row by five-column composite figure.

    The classification maps remain raster images, while the method
    names remain vector text in the exported PDF.
    """
    expected_panels = 10

    if len(panels) != expected_panels:
        raise ValueError(
            f"Expected {expected_panels} panels, got {len(panels)}."
        )

    images = [
        (method_name, load_image(path))
        for method_name, path in panels
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Large internal canvas is intentional:
    # it preserves the extreme vertical aspect ratio of the HC maps.
    fig, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(16.0, 19.0),
        squeeze=False,
    )

    for index, (ax, (method_name, image)) in enumerate(
        zip(axes.flat, images)
    ):
        ax.imshow(
            image,
            interpolation="nearest",
            aspect="equal",
        )

        ax.set_xticks([])
        ax.set_yticks([])

        # Remove every axis border.
        for spine in ax.spines.values():
            spine.set_visible(False)

        panel_letter = chr(ord("a") + index)

        # Use large vector labels because the complete figure will
        # later be scaled down in the IEEE document.
        ax.text(
            0.5,
            -0.025,
            f"({panel_letter}) {method_name}",
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=28,
            fontweight="bold",
            clip_on=False,
        )

    # Explicit spacing avoids caption/image overlap.
    fig.subplots_adjust(
        left=0.015,
        right=0.985,
        top=0.995,
        bottom=0.035,
        wspace=0.055,
        hspace=0.105,
    )

    pdf_path = OUTPUT_DIR / f"{OUTPUT_NAME}.pdf"
    png_path = OUTPUT_DIR / f"{OUTPUT_NAME}.png"

    # PDF: vector labels plus embedded raster maps.
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )

    # PNG fallback. 300 dpi is sufficient because the original
    # source maps are already raster images.
    fig.savefig(
        png_path,
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )

    plt.close(fig)

    if not pdf_path.exists():
        raise RuntimeError(f"PDF was not created: {pdf_path}")

    if not png_path.exists():
        raise RuntimeError(f"PNG was not created: {png_path}")

    return pdf_path, png_path


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    print("=" * 80)
    print("Resolving classification maps")
    print("=" * 80)

    panels = resolve_all_panels()

    print("\n" + "=" * 80)
    print("Creating 2 x 5 composite figure")
    print("=" * 80)

    pdf_path, png_path = create_figure(panels)

    print("\n" + "=" * 80)
    print("Figure creation completed")
    print("=" * 80)
    print(f"Saved PDF : {pdf_path.resolve()}")
    print(f"Saved PNG : {png_path.resolve()}")
    print(f"PDF size  : {pdf_path.stat().st_size / 1024:.1f} KB")
    print(f"PNG size  : {png_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()