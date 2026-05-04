from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SUMMARY_PATH = ROOT / "outputs" / "section6_real" / "section6_summary.json"
OUTPUT_PATH = ROOT / "outputs" / "section6_real" / "section6_resolution_example_panel.png"


def load_examples():
    dataset = torchvision.datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    target_digits = [7, 3, 0]
    chosen = []
    for digit in target_digits:
        for image, label in dataset:
            if int(label) == digit:
                chosen.append((image, label))
                break
    return chosen


def resize_image(image: torch.Tensor, size: int) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    resized = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0)


def resolution_accuracy_map(summary: dict) -> tuple[dict[int, float], dict[int, float]]:
    lif = {int(item["image_size"]): float(item["accuracy"]) for item in summary["resolution_sweep"]["surrogate"]["lif"]}
    circuit = {int(item["image_size"]): float(item["accuracy"]) for item in summary["resolution_sweep"]["surrogate"]["circuit"]}
    return lif, circuit


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    lif_acc, circuit_acc = resolution_accuracy_map(summary)
    examples = load_examples()
    resolutions = [4, 7, 14, 28]

    fig, axes = plt.subplots(len(examples), len(resolutions), figsize=(10, 7))
    if len(examples) == 1:
        axes = [axes]

    for row, (image, label) in enumerate(examples):
        for col, size in enumerate(resolutions):
            ax = axes[row][col]
            resized = resize_image(image, size)
            ax.imshow(resized.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(
                    f"{size}x{size}\nLIF {lif_acc[size]*100:.1f}% | Circuit {circuit_acc[size]*100:.1f}%",
                    fontsize=10,
                )
            if col == 0:
                ax.set_ylabel(f"Digit {int(label)}", fontsize=10)

    fig.suptitle("Section 6: Example MNIST digits across input resolutions", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
