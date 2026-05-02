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
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader, Subset

from circuit_if_neuron import CadenceNeuronParameters, CircuitIFNode


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "section4"
SECTION2_SUMMARY = ROOT / "outputs" / "section2" / "section2_summary.json"
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class Section4Config:
    image_size: int = 28
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 1e-3
    current_scale_a: float = 2500e-9
    dt_s: float = 5e-6
    time_steps: int = 20
    hidden_sizes: tuple[int, ...] = (512, 256, 128, 64, 32)
    train_subset: int | None = 10000
    test_subset: int | None = 2000
    num_workers: int = 0
    seed: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_circuit_params() -> CadenceNeuronParameters:
    summary = json.loads(SECTION2_SUMMARY.read_text())
    fitted = summary["fitted_params"]
    return CadenceNeuronParameters(
        vdd=fitted["vdd_V"],
        c_mem=fitted["c_mem_F"],
        c_fb=fitted["c_fb_F"],
        v_reset=fitted["v_reset_V"],
        v_threshold=fitted["v_threshold_V"],
        refractory_s=fitted["refractory_s"],
        pulse_width_s=fitted["pulse_width_s"],
        leak_conductance_s=fitted["leak_conductance_s"],
        feedback_cap_scale=fitted["feedback_cap_scale"],
        current_gain=fitted["current_gain"],
    )


def make_loaders(
    image_size: int,
    batch_size: int,
    train_subset: int | None,
    test_subset: int | None,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform,
    )

    if train_subset is not None:
        train_dataset = Subset(train_dataset, range(min(train_subset, len(train_dataset))))
    if test_subset is not None:
        test_dataset = Subset(test_dataset, range(min(test_subset, len(test_dataset))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def reset_module_state(module: nn.Module) -> None:
    for child in module.modules():
        if hasattr(child, "reset"):
            child.reset()


class CircuitMNISTSNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_features: int,
        neuron_params: CadenceNeuronParameters,
        current_scale_a: float,
        dt_s: float,
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_features)
        self.sn1 = CircuitIFNode(neuron_params, dt_s=dt_s)
        self.fc2 = nn.Linear(hidden_features, 10)
        self.sn2 = CircuitIFNode(neuron_params, dt_s=dt_s)
        self.current_scale_a = current_scale_a

    def forward(self, x: torch.Tensor, time_steps: int) -> torch.Tensor:
        x = self.flatten(x).clamp(0.0, 1.0)
        out_spike = 0.0
        for _ in range(time_steps):
            encoded = torch.bernoulli(x)
            current_1 = torch.relu(self.fc1(encoded)) * self.current_scale_a
            spike_1 = self.sn1(current_1)
            current_2 = torch.relu(self.fc2(spike_1)) * self.current_scale_a
            out_spike = out_spike + self.sn2(current_2)
        return out_spike / time_steps


def evaluate_model(
    model: CircuitMNISTSNN,
    loader: DataLoader,
    time_steps: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            reset_module_state(model)
            outputs = model(images, time_steps=time_steps)
            loss = loss_fn(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_loss += float(loss.item()) * labels.size(0)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}


def train_model(
    model: CircuitMNISTSNN,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: Section4Config,
    device: torch.device,
) -> dict[str, object]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            reset_module_state(model)
            outputs = model(images, time_steps=cfg.time_steps)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            running_loss += float(loss.item()) * labels.size(0)
            running_correct += int((preds == labels).sum().item())
            running_total += int(labels.size(0))

        test_metrics = evaluate_model(model, test_loader, cfg.time_steps, device)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(running_total, 1),
            "train_accuracy": running_correct / max(running_total, 1),
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
        }
        history.append(epoch_metrics)
        print(
            f"epoch {epoch + 1}/{cfg.epochs}: "
            f"train_acc={epoch_metrics['train_accuracy'] * 100:.2f}% "
            f"test_acc={epoch_metrics['test_accuracy'] * 100:.2f}%"
        )

    return {"history": history, "final_test_metrics": history[-1]}


def save_hidden_size_plot(results: list[dict[str, float]]) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        [item["hidden_size"] for item in results],
        [item["test_accuracy"] for item in results],
        marker="o",
        linewidth=2,
    )
    ax.set_title("Section 4: Circuit-informed SNN accuracy vs hidden-layer size")
    ax.set_xlabel("Hidden-layer neurons")
    ax.set_ylabel("Classification accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "section4_hidden_size_accuracy.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4 workflow: train circuit-informed MNIST SNN.")
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--current-scale-na", type=float, default=2500.0)
    parser.add_argument("--dt-us", type=float, default=5.0)
    parser.add_argument("--time-steps", type=int, default=20)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[512, 256, 128, 64, 32])
    parser.add_argument("--train-subset", type=int, default=10000)
    parser.add_argument("--test-subset", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    cfg = Section4Config(
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        current_scale_a=args.current_scale_na * 1e-9,
        dt_s=args.dt_us * 1e-6,
        time_steps=args.time_steps,
        hidden_sizes=tuple(args.hidden_sizes),
        train_subset=args.train_subset if args.train_subset > 0 else None,
        test_subset=args.test_subset if args.test_subset > 0 else None,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neuron_params = load_circuit_params()
    train_loader, test_loader = make_loaders(
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        train_subset=cfg.train_subset,
        test_subset=cfg.test_subset,
        num_workers=cfg.num_workers,
    )

    results: list[dict[str, float]] = []
    input_size = cfg.image_size * cfg.image_size
    best_model_summary = None

    for hidden_size in cfg.hidden_sizes:
        print(f"Running circuit-informed SNN with hidden size {hidden_size}...")
        model = CircuitMNISTSNN(
            input_size=input_size,
            hidden_features=hidden_size,
            neuron_params=neuron_params,
            current_scale_a=cfg.current_scale_a,
            dt_s=cfg.dt_s,
        )
        experiment = train_model(model, train_loader, test_loader, cfg, device)
        final_metrics = experiment["final_test_metrics"]
        result = {
            "hidden_size": hidden_size,
            "train_accuracy": final_metrics["train_accuracy"],
            "test_accuracy": final_metrics["test_accuracy"],
            "train_loss": final_metrics["train_loss"],
            "test_loss": final_metrics["test_loss"],
        }
        results.append(result)
        if best_model_summary is None or result["test_accuracy"] > best_model_summary["test_accuracy"]:
            best_model_summary = {
                **result,
                "history": experiment["history"],
            }

    plot_path = save_hidden_size_plot(results)
    summary = {
        "config": {
            **asdict(cfg),
            "hidden_sizes": list(cfg.hidden_sizes),
        },
        "device": str(device),
        "circuit_params": {
            "v_reset_V": neuron_params.v_reset,
            "v_threshold_V": neuron_params.v_threshold,
            "refractory_s": neuron_params.refractory_s,
            "pulse_width_s": neuron_params.pulse_width_s,
            "current_gain": neuron_params.current_gain,
            "c_mem_F": neuron_params.c_mem,
        },
        "hidden_size_results": results,
        "best_model": best_model_summary,
        "artifacts": {
            "hidden_size_accuracy": str(plot_path),
        },
    }

    summary_path = OUTPUT_DIR / "section4_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Section 4 workflow complete.")
    print(f"Summary: {summary_path}")
    print(f"hidden_size_accuracy: {plot_path}")
    if best_model_summary is not None:
        print(
            "Best circuit-informed model: "
            f"hidden={best_model_summary['hidden_size']} "
            f"test_acc={best_model_summary['test_accuracy'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
