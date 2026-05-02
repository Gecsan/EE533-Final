from __future__ import annotations

import argparse
import copy
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

from circuit_if_neuron import (
    CadenceNeuronParameters,
    CircuitIFNode,
    fit_cadence_parameters_from_sweep,
    load_cadence_fi_sweep,
)


ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT / "documents"
OUTPUT_DIR = ROOT / "outputs" / "section3"
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class SweepConfig:
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_features: int = 128
    current_scale_a: float = 500e-9
    dt_s: float = 5e-6
    resolution_baseline: int = 14
    time_steps_baseline: int = 10
    train_subset: int | None = 2048
    test_subset: int | None = 1024
    num_workers: int = 0
    seed: int = 0


def build_base_params() -> CadenceNeuronParameters:
    return CadenceNeuronParameters(
        vdd=3.2,
        c_mem=10e-12,
        c_fb=3.757e-12,
        v_reset=1.12,
        v_threshold=1.83,
        refractory_s=2e-6,
        pulse_width_s=2e-6,
        leak_conductance_s=0.0,
        feedback_cap_scale=1.0,
        current_gain=1.0,
    )


def fit_section2_params() -> CadenceNeuronParameters:
    summary_path = ROOT / "outputs" / "section2" / "section2_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
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

    measurements = load_cadence_fi_sweep(DOCUMENTS_DIR / "EE533Final1C.csv")
    return fit_cadence_parameters_from_sweep(build_base_params(), measurements).params


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_mnist_loaders(
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


class BaselineMNISTSNN(nn.Module):
    def __init__(self, input_size: int, hidden_features: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_features)
        self.sn1 = DefaultLIFNode(tau=2.0, v_threshold=0.5, v_reset=0.0)
        self.fc2 = nn.Linear(hidden_features, 10)
        self.sn2 = DefaultLIFNode(tau=2.0, v_threshold=0.5, v_reset=0.0)

    def forward(self, x: torch.Tensor, time_window: int) -> torch.Tensor:
        x = self.flatten(x).clamp(0.0, 1.0)
        out_spike = 0.0
        for _ in range(time_window):
            encoded = torch.bernoulli(x)
            hidden = self.fc1(encoded)
            spike_1 = self.sn1(hidden)
            logits = self.fc2(spike_1)
            out_spike = out_spike + self.sn2(logits)
        return out_spike / time_window


class CircuitMNISTSNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_features: int,
        current_scale_a: float,
        neuron_params: CadenceNeuronParameters,
        dt_s: float,
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_features)
        self.sn1 = CircuitIFNode(neuron_params, dt_s=dt_s)
        self.fc2 = nn.Linear(hidden_features, 10)
        self.sn2 = CircuitIFNode(neuron_params, dt_s=dt_s)
        self.current_scale_a = current_scale_a

    def forward(self, x: torch.Tensor, time_window: int) -> torch.Tensor:
        x = self.flatten(x).clamp(0.0, 1.0)
        out_spike = 0.0
        for _ in range(time_window):
            encoded = torch.bernoulli(x)
            current_1 = torch.relu(self.fc1(encoded)) * self.current_scale_a
            spike_1 = self.sn1(current_1)
            current_2 = torch.relu(self.fc2(spike_1)) * self.current_scale_a
            out_spike = out_spike + self.sn2(current_2)
        return out_spike / time_window


class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane_minus_threshold: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(membrane_minus_threshold)
        return (membrane_minus_threshold >= 0).to(membrane_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (membrane_minus_threshold,) = ctx.saved_tensors
        surrogate = 1.0 / (1.0 + membrane_minus_threshold.abs()).pow(2)
        return grad_output * surrogate


class DefaultLIFNode(nn.Module):
    def __init__(self, tau: float = 2.0, v_threshold: float = 1.0, v_reset: float = 0.0):
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.register_buffer("_vmem", torch.tensor(float(v_reset)))

    def reset(self) -> None:
        self._vmem = torch.tensor(float(self.v_reset), device=self._vmem.device, dtype=self._vmem.dtype)

    def _ensure_state_shape(self, x: torch.Tensor) -> None:
        if (
            self._vmem.ndim == 0
            or self._vmem.shape != x.shape
            or self._vmem.device != x.device
            or self._vmem.dtype != x.dtype
        ):
            self._vmem = torch.full_like(x, self.v_reset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state_shape(x)
        updated_v = self._vmem + (x - (self._vmem - self.v_reset)) / self.tau
        spikes = _SurrogateSpike.apply(updated_v - self.v_threshold)
        self._vmem = torch.where(spikes > 0, torch.full_like(updated_v, self.v_reset), updated_v)
        return spikes


def reset_module_state(module: nn.Module) -> None:
    for child in module.modules():
        if hasattr(child, "reset"):
            child.reset()


def build_model(
    neuron_kind: str,
    input_size: int,
    cfg: SweepConfig,
    neuron_params: CadenceNeuronParameters,
) -> nn.Module:
    if neuron_kind == "lif":
        return BaselineMNISTSNN(
            input_size=input_size,
            hidden_features=cfg.hidden_features,
        )
    if neuron_kind == "circuit":
        return CircuitMNISTSNN(
            input_size=input_size,
            hidden_features=cfg.hidden_features,
            current_scale_a=cfg.current_scale_a,
            neuron_params=neuron_params,
            dt_s=cfg.dt_s,
        )
    raise ValueError(f"Unsupported neuron kind: {neuron_kind}")


def evaluate_model(
    model: nn.Module,
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
            outputs = model(images, time_window=time_steps)
            loss = loss_fn(outputs, labels)
            preds = outputs.argmax(dim=1)

            total_loss += float(loss.item()) * labels.size(0)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    time_steps: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, object]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            reset_module_state(model)
            outputs = model(images, time_window=time_steps)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            running_loss += float(loss.item()) * labels.size(0)
            running_correct += int((preds == labels).sum().item())
            running_total += int(labels.size(0))

        train_metrics = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(running_total, 1),
            "train_accuracy": running_correct / max(running_total, 1),
        }
        test_metrics = evaluate_model(model, test_loader, time_steps, device)
        history.append(
            {
                **train_metrics,
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
            }
        )
        print(
            f"epoch {epoch + 1}/{epochs}: "
            f"train_acc={train_metrics['train_accuracy'] * 100:.2f}% "
            f"test_acc={test_metrics['accuracy'] * 100:.2f}%"
        )

    return {
        "history": history,
        "final_test_metrics": history[-1] if history else {},
    }


def quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2:
        raise ValueError("bits must be at least 2")
    levels = 2**bits
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    if torch.isclose(tensor_min, tensor_max):
        return tensor.clone()
    scale = (tensor_max - tensor_min) / (levels - 1)
    return torch.round((tensor - tensor_min) / scale) * scale + tensor_min


def quantize_model_weights(model: nn.Module, bits: int) -> nn.Module:
    model_copy = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in model_copy.named_parameters():
            if "weight" in name:
                param.copy_(quantize_tensor(param, bits))
    return model_copy


def run_single_experiment(
    neuron_kind: str,
    image_size: int,
    time_steps: int,
    cfg: SweepConfig,
    neuron_params: CadenceNeuronParameters,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    input_size = image_size * image_size
    train_loader, test_loader = make_mnist_loaders(
        image_size=image_size,
        batch_size=cfg.batch_size,
        train_subset=cfg.train_subset,
        test_subset=cfg.test_subset,
        num_workers=cfg.num_workers,
    )
    model = build_model(
        neuron_kind=neuron_kind,
        input_size=input_size,
        cfg=cfg,
        neuron_params=neuron_params,
    )
    results = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        time_steps=time_steps,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        device=device,
    )
    return model, results


def save_plot(x_values, lif_values, circuit_values, title: str, xlabel: str, ylabel: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x_values, lif_values, marker="o", label="Default LIF")
    ax.plot(x_values, circuit_values, marker="s", label="Circuit IF")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_baseline_bar_plot(lif_accuracy: float, circuit_accuracy: float) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(["Default LIF", "Circuit IF"], [lif_accuracy, circuit_accuracy], color=["tab:blue", "tab:orange"])
    ax.set_title("Section 3: Baseline MNIST accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "section3_baseline_accuracy.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 3 MNIST workflow for default vs circuit-informed SNNs.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-subset", type=int, default=2048)
    parser.add_argument("--test-subset", type=int, default=1024)
    parser.add_argument("--hidden-features", type=int, default=128)
    parser.add_argument("--baseline-resolution", type=int, default=14)
    parser.add_argument("--baseline-time-steps", type=int, default=10)
    parser.add_argument("--resolution-sweep", type=int, nargs="*", default=[4, 7, 14, 28])
    parser.add_argument("--time-step-sweep", type=int, nargs="*", default=[5, 10, 20])
    parser.add_argument("--quant-bits", type=int, nargs="*", default=[2, 3, 4, 5])
    parser.add_argument("--current-scale-na", type=float, default=500.0)
    parser.add_argument("--dt-us", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = SweepConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_features=args.hidden_features,
        current_scale_a=args.current_scale_na * 1e-9,
        dt_s=args.dt_us * 1e-6,
        resolution_baseline=args.baseline_resolution,
        time_steps_baseline=args.baseline_time_steps,
        train_subset=args.train_subset if args.train_subset > 0 else None,
        test_subset=args.test_subset if args.test_subset > 0 else None,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neuron_params = fit_section2_params()

    print("Running baseline LIF experiment...")
    lif_model, lif_baseline = run_single_experiment(
        neuron_kind="lif",
        image_size=cfg.resolution_baseline,
        time_steps=cfg.time_steps_baseline,
        cfg=cfg,
        neuron_params=neuron_params,
        device=device,
    )

    print("Running circuit-informed experiment...")
    circuit_model, circuit_baseline = run_single_experiment(
        neuron_kind="circuit",
        image_size=cfg.resolution_baseline,
        time_steps=cfg.time_steps_baseline,
        cfg=cfg,
        neuron_params=neuron_params,
        device=device,
    )

    resolution_results = {"lif": [], "circuit": []}
    for size in args.resolution_sweep:
        print(f"Resolution sweep at {size}x{size} with default LIF...")
        _, results = run_single_experiment("lif", size, cfg.time_steps_baseline, cfg, neuron_params, device)
        resolution_results["lif"].append(
            {"image_size": size, "accuracy": results["final_test_metrics"]["test_accuracy"]}
        )

        print(f"Resolution sweep at {size}x{size} with circuit IF...")
        _, results = run_single_experiment("circuit", size, cfg.time_steps_baseline, cfg, neuron_params, device)
        resolution_results["circuit"].append(
            {"image_size": size, "accuracy": results["final_test_metrics"]["test_accuracy"]}
        )

    timestep_results = {"lif": [], "circuit": []}
    for time_steps in args.time_step_sweep:
        print(f"Time-step sweep at T={time_steps} with default LIF...")
        _, results = run_single_experiment("lif", cfg.resolution_baseline, time_steps, cfg, neuron_params, device)
        timestep_results["lif"].append(
            {"time_steps": time_steps, "accuracy": results["final_test_metrics"]["test_accuracy"]}
        )

        print(f"Time-step sweep at T={time_steps} with circuit IF...")
        _, results = run_single_experiment("circuit", cfg.resolution_baseline, time_steps, cfg, neuron_params, device)
        timestep_results["circuit"].append(
            {"time_steps": time_steps, "accuracy": results["final_test_metrics"]["test_accuracy"]}
        )

    quant_results = {"lif": [], "circuit": []}
    quant_test_loader = make_mnist_loaders(
        image_size=cfg.resolution_baseline,
        batch_size=cfg.batch_size,
        train_subset=cfg.train_subset,
        test_subset=cfg.test_subset,
        num_workers=cfg.num_workers,
    )[1]
    for bits in args.quant_bits:
        print(f"Quantization sweep at {bits} bits...")
        lif_quant_model = quantize_model_weights(lif_model, bits).to(device)
        circuit_quant_model = quantize_model_weights(circuit_model, bits).to(device)

        lif_quant_metrics = evaluate_model(lif_quant_model, quant_test_loader, cfg.time_steps_baseline, device)
        circuit_quant_metrics = evaluate_model(circuit_quant_model, quant_test_loader, cfg.time_steps_baseline, device)
        quant_results["lif"].append({"bits": bits, "accuracy": lif_quant_metrics["accuracy"]})
        quant_results["circuit"].append({"bits": bits, "accuracy": circuit_quant_metrics["accuracy"]})

    baseline_plot_path = save_baseline_bar_plot(
        lif_accuracy=lif_baseline["final_test_metrics"]["test_accuracy"],
        circuit_accuracy=circuit_baseline["final_test_metrics"]["test_accuracy"],
    )
    resolution_plot_path = save_plot(
        [item["image_size"] for item in resolution_results["lif"]],
        [item["accuracy"] for item in resolution_results["lif"]],
        [item["accuracy"] for item in resolution_results["circuit"]],
        title="Section 3: Resolution sweep",
        xlabel="MNIST resolution",
        ylabel="Accuracy",
        filename="section3_resolution_sweep.png",
    )
    timestep_plot_path = save_plot(
        [item["time_steps"] for item in timestep_results["lif"]],
        [item["accuracy"] for item in timestep_results["lif"]],
        [item["accuracy"] for item in timestep_results["circuit"]],
        title="Section 3: Time-step sweep",
        xlabel="Time steps",
        ylabel="Accuracy",
        filename="section3_timestep_sweep.png",
    )
    quant_plot_path = save_plot(
        [item["bits"] for item in quant_results["lif"]],
        [item["accuracy"] for item in quant_results["lif"]],
        [item["accuracy"] for item in quant_results["circuit"]],
        title="Section 3: Post-training weight quantization sweep",
        xlabel="Quantization bits",
        ylabel="Accuracy",
        filename="section3_quantization_sweep.png",
    )

    summary = {
        "config": asdict(cfg),
        "device": str(device),
        "circuit_params": {
            "v_reset_V": neuron_params.v_reset,
            "v_threshold_V": neuron_params.v_threshold,
            "refractory_s": neuron_params.refractory_s,
            "pulse_width_s": neuron_params.pulse_width_s,
            "current_gain": neuron_params.current_gain,
            "c_mem_F": neuron_params.c_mem,
        },
        "baseline_accuracy": {
            "lif": lif_baseline["final_test_metrics"]["test_accuracy"],
            "circuit": circuit_baseline["final_test_metrics"]["test_accuracy"],
        },
        "baseline_histories": {
            "lif": lif_baseline["history"],
            "circuit": circuit_baseline["history"],
        },
        "resolution_sweep": resolution_results,
        "time_step_sweep": timestep_results,
        "quantization_sweep": quant_results,
        "artifacts": {
            "baseline_accuracy": str(baseline_plot_path),
            "resolution_sweep": str(resolution_plot_path),
            "time_step_sweep": str(timestep_plot_path),
            "quantization_sweep": str(quant_plot_path),
        },
    }

    summary_path = OUTPUT_DIR / "section3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Section 3 workflow complete.")
    print(f"Summary: {summary_path}")
    for name, path in summary["artifacts"].items():
        print(f"{name}: {path}")
    print(
        "Baseline accuracy: "
        f"LIF={summary['baseline_accuracy']['lif'] * 100:.2f}% "
        f"Circuit={summary['baseline_accuracy']['circuit'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
