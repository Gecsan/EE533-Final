from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from section6_compare_workflow import (
    Section6Config,
    fit_section2_params,
    make_mnist_loaders,
    run_surrogate_experiment,
    set_seed,
)


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "outputs" / "section6_real_v4" / "section6_summary.json"
OUT_PATH = ROOT / "outputs" / "section6_real_v4" / "section6_7x7_lif_vs_circuit_examples.png"


def upsample_for_display(image: torch.Tensor, out_size: int = 28) -> torch.Tensor:
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image = image.unsqueeze(0)
    up = F.interpolate(image.float(), size=(out_size, out_size), mode="nearest")
    return up.squeeze().cpu()


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    lif_acc = next(item["accuracy"] for item in summary["resolution_sweep"]["surrogate"]["lif"] if item["image_size"] == 7)
    circuit_acc = next(item["accuracy"] for item in summary["resolution_sweep"]["surrogate"]["circuit"] if item["image_size"] == 7)

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

    # Recreate the exact 7x7 surrogate models used for the comparison.
    lif_model, _ = run_surrogate_experiment("lif", 7, cfg.time_steps_baseline, cfg, neuron_params, device)
    circuit_model, _ = run_surrogate_experiment("circuit", 7, cfg.time_steps_baseline, cfg, neuron_params, device)
    lif_model.to(device).eval()
    circuit_model.to(device).eval()

    _, test_loader = make_mnist_loaders(
        image_size=7,
        batch_size=32,
        train_subset=cfg.train_subset,
        test_subset=cfg.test_subset,
        num_workers=cfg.num_workers,
    )

    # Pick samples where circuit is right and LIF is wrong to illustrate the gap.
    selected = []
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            lif_out = lif_model(images, time_window=cfg.time_steps_baseline)
            circuit_out = circuit_model(images, time_window=cfg.time_steps_baseline)
            lif_pred = lif_out.argmax(dim=1)
            circuit_pred = circuit_out.argmax(dim=1)
            for i in range(images.size(0)):
                if int(circuit_pred[i]) == int(labels[i]) and int(lif_pred[i]) != int(labels[i]):
                    selected.append(
                        {
                            "image": images[i].detach().cpu(),
                            "label": int(labels[i]),
                            "lif_pred": int(lif_pred[i]),
                            "circuit_pred": int(circuit_pred[i]),
                        }
                    )
                if len(selected) >= 6:
                    break
            if len(selected) >= 6:
                break

    if not selected:
        raise RuntimeError("Could not find illustrative samples where circuit outperforms LIF.")

    fig, axes = plt.subplots(len(selected), 3, figsize=(8, 2.2 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for row, item in enumerate(selected):
        img7 = item["image"].squeeze(0)
        img28 = upsample_for_display(img7, out_size=28)

        axes[row][0].imshow(img28.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[row][0].set_title(f"Target: {item['label']}", fontsize=10)
        axes[row][1].imshow(img28.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[row][1].set_title(f"LIF: {item['lif_pred']}", fontsize=10, color="tab:blue")
        axes[row][2].imshow(img28.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[row][2].set_title(f"Circuit: {item['circuit_pred']}", fontsize=10, color="tab:orange")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"7x7 surrogate comparison: LIF {lif_acc*100:.1f}% vs Circuit {circuit_acc*100:.1f}%\n"
        "Examples where the circuit-informed model predicts correctly and LIF does not",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
