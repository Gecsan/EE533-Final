from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

import matplotlib.pyplot as plt
import torch
from torch import nn

from section3_mnist_workflow import quantize_tensor
from section4_training_workflow import (
    CircuitMNISTSNN,
    Section4Config,
    evaluate_model,
    load_circuit_params,
    make_loaders,
    set_seed,
    train_model,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "section7"


@dataclass(frozen=True)
class Section7Config:
    epochs: int = 2
    batch_size: int = 64
    learning_rate: float = 1e-3
    current_scale_a: float = 2.5e-6
    dt_s: float = 5e-6
    train_subset: int | None = 4000
    test_subset: int | None = 2000
    num_workers: int = 0
    seed: int = 0
    target_accuracy: float = 0.75
    hidden_sizes: tuple[int, ...] = (128, 64, 32)
    time_steps: tuple[int, ...] = (20, 10, 5)
    quant_bits: tuple[int, ...] = (8, 4, 2)
    image_sizes: tuple[int, ...] = (28, 14, 7, 4)


def simplify_score(result: dict[str, float]) -> tuple[int, int, int, int]:
    # Lower is simpler.
    return (
        int(result["hidden_size"]),
        int(result["time_steps"]),
        int(result["quant_bits"]),
        int(result["image_size"]),
    )


def candidate_search_order(candidate: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    image_size, hidden_size, time_steps, quant_bits = candidate
    # Search likely-to-work simplifications first:
    # keep resolution and time steps high enough to preserve accuracy,
    # while still preferring smaller hidden layers and lower precision.
    return (
        -image_size,
        -time_steps,
        hidden_size,
        quant_bits,
    )


def quantize_model_inplace(model: nn.Module, bits: int) -> dict[str, torch.Tensor]:
    backup = {name: param.detach().clone() for name, param in model.named_parameters()}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                param.copy_(quantize_tensor(param, bits))
    return backup


def restore_model_parameters(model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(backup[name])


def run_candidate(
    image_size: int,
    hidden_size: int,
    time_steps: int,
    quant_bits: int,
    cfg: Section7Config,
    device: torch.device,
) -> dict[str, float]:
    neuron_params = load_circuit_params()
    train_loader, test_loader = make_loaders(
        image_size=image_size,
        batch_size=cfg.batch_size,
        train_subset=cfg.train_subset,
        test_subset=cfg.test_subset,
        num_workers=cfg.num_workers,
    )
    train_cfg = Section4Config(
        image_size=image_size,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        current_scale_a=cfg.current_scale_a,
        dt_s=cfg.dt_s,
        time_steps=time_steps,
        hidden_sizes=(hidden_size,),
        train_subset=cfg.train_subset,
        test_subset=cfg.test_subset,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    model = CircuitMNISTSNN(
        input_size=image_size * image_size,
        hidden_features=hidden_size,
        neuron_params=neuron_params,
        current_scale_a=cfg.current_scale_a,
        dt_s=cfg.dt_s,
    )
    train_result = train_model(model, train_loader, test_loader, train_cfg, device)
    backup = quantize_model_inplace(model, quant_bits)
    try:
        quant_metrics = evaluate_model(model, test_loader, time_steps, device)
    finally:
        restore_model_parameters(model, backup)

    return {
        "image_size": image_size,
        "hidden_size": hidden_size,
        "time_steps": time_steps,
        "quant_bits": quant_bits,
        "train_accuracy": float(train_result["final_test_metrics"]["train_accuracy"]),
        "test_accuracy_pre_quant": float(train_result["final_test_metrics"]["test_accuracy"]),
        "test_accuracy": float(quant_metrics["accuracy"]),
    }


def format_candidate_label(item: dict[str, float]) -> str:
    return (
        f"{item['image_size']}x{item['image_size']}\n"
        f"h{item['hidden_size']}  T{item['time_steps']}\n"
        f"{item['quant_bits']}b"
    )


def save_scatter(
    results: list[dict[str, float]],
    target_accuracy: float,
    title: str = "Section 7: Simplified circuit-informed model candidates",
    output_path: Path | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    x = list(range(len(results)))
    y = [item["test_accuracy"] for item in results]
    labels = [format_candidate_label(item) for item in results]
    colors = ["tab:green" if item["test_accuracy"] >= target_accuracy else "tab:red" for item in results]
    ax.bar(x, y, color=colors)
    ax.axhline(target_accuracy, color="black", linestyle="--", linewidth=1, label="75% target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8, linespacing=1.15)
    ax.set_ylabel("Test Accuracy")
    ax.set_xlabel("Candidate configuration\nresolution / hidden size / time steps / quantization bits")
    ax.set_title(title)
    ax.set_ylim(0.0, 0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout(pad=1.6)
    out = output_path or (OUTPUT_DIR / "section7_candidate_accuracy.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 7 workflow: simplify the circuit-informed SNN while staying above 75% accuracy.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-subset", type=int, default=4000)
    parser.add_argument("--test-subset", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--current-scale-na", type=float, default=2500.0)
    parser.add_argument("--dt-us", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-accuracy", type=float, default=0.75)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 64, 32])
    parser.add_argument("--time-steps", type=int, nargs="*", default=[20, 10, 5])
    parser.add_argument("--quant-bits", type=int, nargs="*", default=[8, 4, 2])
    parser.add_argument("--image-sizes", type=int, nargs="*", default=[28, 14, 7, 4])
    parser.add_argument("--max-candidates", type=int, default=12)
    args = parser.parse_args()

    cfg = Section7Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        current_scale_a=args.current_scale_na * 1e-9,
        dt_s=args.dt_us * 1e-6,
        train_subset=args.train_subset if args.train_subset > 0 else None,
        test_subset=args.test_subset if args.test_subset > 0 else None,
        num_workers=args.num_workers,
        seed=args.seed,
        target_accuracy=args.target_accuracy,
        hidden_sizes=tuple(args.hidden_sizes),
        time_steps=tuple(args.time_steps),
        quant_bits=tuple(args.quant_bits),
        image_sizes=tuple(args.image_sizes),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates: list[tuple[int, int, int, int]] = []
    for image_size in cfg.image_sizes:
        for hidden_size in cfg.hidden_sizes:
            for time_steps in cfg.time_steps:
                for quant_bits in cfg.quant_bits:
                    candidates.append((image_size, hidden_size, time_steps, quant_bits))
    candidates.sort(key=candidate_search_order)
    candidates = candidates[: args.max_candidates]

    results = []
    for image_size, hidden_size, time_steps, quant_bits in candidates:
        print(
            f"Testing candidate: image={image_size}, hidden={hidden_size}, "
            f"T={time_steps}, quant={quant_bits}-bit"
        )
        result = run_candidate(image_size, hidden_size, time_steps, quant_bits, cfg, device)
        results.append(result)

    passing = [result for result in results if result["test_accuracy"] >= cfg.target_accuracy]
    best = None
    if passing:
        best = min(passing, key=simplify_score)

    plot_path = save_scatter(results, cfg.target_accuracy)
    summary = {
        "config": {
            **asdict(cfg),
            "hidden_sizes": list(cfg.hidden_sizes),
            "time_steps": list(cfg.time_steps),
            "quant_bits": list(cfg.quant_bits),
            "image_sizes": list(cfg.image_sizes),
        },
        "device": str(device),
        "candidate_results": results,
        "passing_candidates": passing,
        "best_simplified_model": best,
        "artifacts": {
            "candidate_accuracy": str(plot_path),
        },
        "notes": [
            "All candidates use the circuit-informed surrogate model only.",
            "A candidate counts as valid when its post-quantization test accuracy is at least 75%.",
            "The reported best simplified model is the smallest passing candidate according to hidden size, time steps, quantization bits, and image size.",
        ],
    }
    summary_path = OUTPUT_DIR / "section7_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Section 7 workflow complete.")
    print(f"Summary: {summary_path}")
    print(f"candidate_accuracy: {plot_path}")
    if best is not None:
        print(
            "Best simplified model: "
            f"image={best['image_size']} hidden={best['hidden_size']} "
            f"T={best['time_steps']} quant={best['quant_bits']} "
            f"acc={best['test_accuracy'] * 100:.2f}%"
        )
    else:
        print("No candidate reached the target accuracy.")


if __name__ == "__main__":
    main()
