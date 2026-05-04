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

from section6_compare_workflow import (
    Section6Config,
    fit_section2_params,
    make_mnist_loaders,
    run_surrogate_experiment,
    set_seed,
)


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "outputs" / "section6_real_v4" / "section6_summary.json"
OUTPUT_PATH = ROOT / "outputs" / "section6_real_v4" / "section6_resolution_model_grid_8x3.png"
DATA_DIR = ROOT / "data"


def load_fixed_digits() -> dict[int, torch.Tensor]:
    dataset = torchvision.datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    chosen: dict[int, torch.Tensor] = {}
    for image, label in dataset:
        digit = int(label)
        if digit in (7, 3, 0) and digit not in chosen:
            chosen[digit] = image
        if len(chosen) == 3:
            break
    return chosen


def resize_image(image: torch.Tensor, size: int) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    resized = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)
    return resized.squeeze(0)


def run_prediction(model, image: torch.Tensor, time_steps: int, device: torch.device) -> int:
    model.eval()
    with torch.inference_mode():
        output = model(image.unsqueeze(0).to(device), time_window=time_steps)
        return int(output.argmax(dim=1).item())


def get_surrogate_resolution_accuracy(summary: dict, model_kind: str, size: int) -> float:
    return next(
        item["accuracy"]
        for item in summary["resolution_sweep"]["surrogate"][model_kind]
        if int(item["image_size"]) == size
    )


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    cfg_dict = summary["config"]
    cfg = Section6Config(
        surrogate_epochs=cfg_dict["surrogate_epochs"],
        stdp_epochs=cfg_dict["stdp_epochs"],
        classifier_epochs=cfg_dict["classifier_epochs"],
        batch_size=cfg_dict["batch_size"],
        learning_rate=cfg_dict["learning_rate"],
        stdp_learning_rate=cfg_dict["stdp_learning_rate"],
        trace_decay=cfg_dict["trace_decay"],
        hidden_features=cfg_dict["hidden_features"],
        current_scale_a=cfg_dict["current_scale_a"],
        dt_s=cfg_dict["dt_s"],
        resolution_baseline=cfg_dict["resolution_baseline"],
        time_steps_baseline=cfg_dict["time_steps_baseline"],
        train_subset=cfg_dict["train_subset"],
        test_subset=cfg_dict["test_subset"],
        num_workers=cfg_dict["num_workers"],
        seed=cfg_dict["seed"],
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neuron_params = fit_section2_params()
    fixed_digits = load_fixed_digits()
    resolutions = [4, 7, 14, 28]

    models: dict[tuple[int, str], object] = {}
    for size in resolutions:
        lif_model, _ = run_surrogate_experiment("lif", size, cfg.time_steps_baseline, cfg, neuron_params, device)
        circuit_model, _ = run_surrogate_experiment("circuit", size, cfg.time_steps_baseline, cfg, neuron_params, device)
        models[(size, "lif")] = lif_model.to(device)
        models[(size, "circuit")] = circuit_model.to(device)

    fig, axes = plt.subplots(8, 3, figsize=(9, 18))
    digit_order = [7, 3, 0]

    for block_idx, size in enumerate(resolutions):
        for model_offset, model_kind in enumerate(("lif", "circuit")):
            row = block_idx * 2 + model_offset
            acc = get_surrogate_resolution_accuracy(summary, model_kind, size)
            model = models[(size, model_kind)]

            for col, digit in enumerate(digit_order):
                image28 = fixed_digits[digit]
                resized = resize_image(image28, size)
                pred = run_prediction(model, resized, cfg.time_steps_baseline, device)
                display = F.interpolate(resized.unsqueeze(0), size=(28, 28), mode="nearest").squeeze(0).squeeze(0)

                ax = axes[row, col]
                ax.imshow(display.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
                ax.set_xticks([])
                ax.set_yticks([])

                color = "tab:green" if pred == digit else "tab:red"
                ax.set_title(f"pred {pred}", fontsize=10, color=color)

                if col == 0:
                    label = "LIF" if model_kind == "lif" else "Circuit"
                    ax.set_ylabel(f"{size}x{size}\n{label}\n{acc*100:.1f}%", fontsize=10)

    for col, digit in enumerate(digit_order):
        axes[0, col].set_xlabel(f"Digit {digit}", fontsize=11)
        axes[0, col].xaxis.set_label_position("top")

    fig.suptitle(
        "Section 6: Surrogate examples across resolutions\nRows alternate LIF and circuit-informed models; columns are fixed example digits",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.988))
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
