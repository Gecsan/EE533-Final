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
from section3_mnist_workflow import DefaultLIFNode, fit_section2_params, quantize_tensor, reset_module_state


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "section6"
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class Section6Config:
    surrogate_epochs: int = 3
    stdp_epochs: int = 2
    classifier_epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 1e-3
    stdp_learning_rate: float = 0.01
    trace_decay: float = 0.9
    hidden_features: int = 128
    current_scale_a: float = 2.5e-6
    dt_s: float = 5e-6
    resolution_baseline: int = 28
    time_steps_baseline: int = 20
    train_subset: int | None = 4000
    test_subset: int | None = 2000
    num_workers: int = 0
    seed: int = 0


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


def evaluate_surrogate_model(
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

    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}


def train_surrogate_model(
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

        test_metrics = evaluate_surrogate_model(model, test_loader, time_steps, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / max(running_total, 1),
                "train_accuracy": running_correct / max(running_total, 1),
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
            }
        )

    return {"history": history, "final_test_metrics": history[-1] if history else {}}


class STDPFeatureExtractor(nn.Module):
    def __init__(
        self,
        neuron_kind: str,
        input_size: int,
        hidden_features: int,
        neuron_params: CadenceNeuronParameters,
        current_scale_a: float,
        dt_s: float,
    ):
        super().__init__()
        self.neuron_kind = neuron_kind
        self.input_size = input_size
        self.hidden_features = hidden_features
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size, hidden_features, bias=False)
        self.current_scale_a = current_scale_a
        self.neuron_params = neuron_params
        self.dt_s = dt_s
        if neuron_kind == "lif":
            self.neuron = DefaultLIFNode(tau=2.0, v_threshold=0.5, v_reset=0.0)
        elif neuron_kind == "circuit":
            self.neuron = CircuitIFNode(neuron_params, dt_s=dt_s)
        else:
            raise ValueError(f"Unsupported neuron kind: {neuron_kind}")

        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)

    def reset(self) -> None:
        if hasattr(self.neuron, "reset"):
            self.neuron.reset()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bernoulli(x.clamp(0.0, 1.0))

    def forward_step(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = self.flatten(encoded).float()
        hidden = self.fc(flat)
        if self.neuron_kind == "circuit":
            spikes = self.neuron(torch.relu(hidden) * self.current_scale_a)
        else:
            spikes = self.neuron(hidden)
        return flat, spikes

    def extract_features(self, x: torch.Tensor, time_steps: int) -> torch.Tensor:
        self.reset()
        feature_sum = None
        for _ in range(time_steps):
            encoded = self.encode(x)
            _, spikes = self.forward_step(encoded)
            feature_sum = spikes if feature_sum is None else feature_sum + spikes
        return feature_sum / time_steps

    def extract_classifier_features(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.flatten(x).float().clamp(0.0, 1.0)
        return torch.relu(self.fc(flat))


class STDPClassifier(nn.Module):
    def __init__(self, feature_extractor: STDPFeatureExtractor, hidden_features: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(hidden_features, 10)

    def forward(self, x: torch.Tensor, time_steps: int) -> torch.Tensor:
        with torch.no_grad():
            features = self.feature_extractor.extract_classifier_features(x)
        return self.classifier(features)


def normalize_feature_weights(feature_extractor: STDPFeatureExtractor) -> None:
    with torch.no_grad():
        weights = feature_extractor.fc.weight.data
        weights.clamp_(min=0.0)
        norms = weights.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
        weights.div_(norms)


def train_stdp_feature_extractor(
    feature_extractor: STDPFeatureExtractor,
    train_loader: DataLoader,
    time_steps: int,
    stdp_epochs: int,
    stdp_lr: float,
    trace_decay: float,
    device: torch.device,
) -> None:
    feature_extractor.to(device)
    feature_extractor.train()

    for _ in range(stdp_epochs):
        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            for sample_idx in range(images.size(0)):
                image = images[sample_idx : sample_idx + 1]
                feature_extractor.reset()
                pre_trace = torch.zeros(feature_extractor.input_size, device=device)
                for _ in range(time_steps):
                    encoded = feature_extractor.encode(image)
                    pre_spike, post_spike = feature_extractor.forward_step(encoded)
                    pre_trace = trace_decay * pre_trace + pre_spike.squeeze(0)
                    post_binary = (post_spike > 0).float().view(-1)
                    if post_binary.sum() > 0:
                        winner = int(torch.argmax(post_binary).item())
                        with torch.no_grad():
                            winner_weights = feature_extractor.fc.weight.data[winner]
                            winner_weights.add_(stdp_lr * (pre_trace - winner_weights))
                normalize_feature_weights(feature_extractor)


def evaluate_stdp_classifier(
    model: STDPClassifier,
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
            outputs = model(images, time_steps=time_steps)
            loss = loss_fn(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_loss += float(loss.item()) * labels.size(0)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}


def train_stdp_classifier(
    feature_extractor: STDPFeatureExtractor,
    train_loader: DataLoader,
    test_loader: DataLoader,
    time_steps: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, object]:
    model = STDPClassifier(feature_extractor, feature_extractor.hidden_features).to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
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
            outputs = model(images, time_steps=time_steps)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            running_loss += float(loss.item()) * labels.size(0)
            running_correct += int((preds == labels).sum().item())
            running_total += int(labels.size(0))

        test_metrics = evaluate_stdp_classifier(model, test_loader, time_steps, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / max(running_total, 1),
                "train_accuracy": running_correct / max(running_total, 1),
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
            }
        )

    return {"model": model, "history": history, "final_test_metrics": history[-1] if history else {}}


def run_surrogate_experiment(
    neuron_kind: str,
    image_size: int,
    time_steps: int,
    cfg: Section6Config,
    neuron_params: CadenceNeuronParameters,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    train_loader, test_loader = make_mnist_loaders(
        image_size=image_size,
        batch_size=cfg.batch_size,
        train_subset=cfg.train_subset,
        test_subset=cfg.test_subset,
        num_workers=cfg.num_workers,
    )
    input_size = image_size * image_size
    if neuron_kind == "lif":
        model = BaselineMNISTSNN(input_size, cfg.hidden_features)
    elif neuron_kind == "circuit":
        model = CircuitMNISTSNN(input_size, cfg.hidden_features, cfg.current_scale_a, neuron_params, cfg.dt_s)
    else:
        raise ValueError(f"Unsupported neuron kind: {neuron_kind}")
    results = train_surrogate_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        time_steps=time_steps,
        epochs=cfg.surrogate_epochs,
        learning_rate=cfg.learning_rate,
        device=device,
    )
    return model, results


def run_stdp_experiment(
    neuron_kind: str,
    image_size: int,
    time_steps: int,
    cfg: Section6Config,
    neuron_params: CadenceNeuronParameters,
    device: torch.device,
) -> tuple[STDPClassifier, dict[str, object]]:
    train_loader, test_loader = make_mnist_loaders(
        image_size=image_size,
        batch_size=cfg.batch_size,
        train_subset=cfg.train_subset,
        test_subset=cfg.test_subset,
        num_workers=cfg.num_workers,
    )
    input_size = image_size * image_size
    feature_extractor = STDPFeatureExtractor(
        neuron_kind=neuron_kind,
        input_size=input_size,
        hidden_features=cfg.hidden_features,
        neuron_params=neuron_params,
        current_scale_a=cfg.current_scale_a,
        dt_s=cfg.dt_s,
    )
    train_stdp_feature_extractor(
        feature_extractor=feature_extractor,
        train_loader=train_loader,
        time_steps=time_steps,
        stdp_epochs=cfg.stdp_epochs,
        stdp_lr=cfg.stdp_learning_rate,
        trace_decay=cfg.trace_decay,
        device=device,
    )
    classifier_results = train_stdp_classifier(
        feature_extractor=feature_extractor,
        train_loader=train_loader,
        test_loader=test_loader,
        time_steps=time_steps,
        epochs=cfg.classifier_epochs,
        learning_rate=cfg.learning_rate,
        device=device,
    )
    return classifier_results["model"], classifier_results


def evaluate_quantized(
    model: nn.Module,
    training_method: str,
    loader: DataLoader,
    time_steps: int,
    device: torch.device,
) -> dict[str, float]:
    if training_method == "surrogate":
        return evaluate_surrogate_model(model, loader, time_steps, device)
    return evaluate_stdp_classifier(model, loader, time_steps, device)


def evaluate_quantized_inplace(
    model: nn.Module,
    bits: int,
    training_method: str,
    loader: DataLoader,
    time_steps: int,
    device: torch.device,
) -> dict[str, float]:
    backup = {name: param.detach().clone() for name, param in model.named_parameters()}
    try:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "weight" in name:
                    param.copy_(quantize_tensor(param, bits))
        return evaluate_quantized(model, training_method, loader, time_steps, device)
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(backup[name])


def save_factor_plot(results: dict[str, dict[str, list[dict[str, float]]]], factor_key: str, xlabel: str, title: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    style_map = {
        ("surrogate", "lif"): ("tab:blue", "o", "LIF surrogate"),
        ("surrogate", "circuit"): ("tab:orange", "s", "Circuit surrogate"),
        ("stdp", "lif"): ("tab:green", "^", "LIF STDP"),
        ("stdp", "circuit"): ("tab:red", "D", "Circuit STDP"),
    }
    for training_method, neuron_map in results.items():
        for neuron_kind, values in neuron_map.items():
            x = [item[factor_key] for item in values]
            y = [item["accuracy"] for item in values]
            color, marker, label = style_map[(training_method, neuron_kind)]
            ax.plot(x, y, color=color, marker=marker, label=label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def save_training_method_bar_plot(summary: dict[str, dict[str, float]]) -> Path:
    labels = ["LIF\nsurrogate", "Circuit\nsurrogate", "LIF\nSTDP", "Circuit\nSTDP"]
    values = [
        summary["surrogate"]["lif"],
        summary["surrogate"]["circuit"],
        summary["stdp"]["lif"],
        summary["stdp"]["circuit"],
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green", "tab:red"])
    ax.set_title("Section 6: Training method comparison")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / "section6_training_method_accuracy.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Section 6 workflow: default vs circuit-informed neurons across surrogate and STDP training.")
    parser.add_argument("--surrogate-epochs", type=int, default=3)
    parser.add_argument("--stdp-epochs", type=int, default=2)
    parser.add_argument("--classifier-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-subset", type=int, default=4000)
    parser.add_argument("--test-subset", type=int, default=2000)
    parser.add_argument("--hidden-features", type=int, default=128)
    parser.add_argument("--baseline-resolution", type=int, default=28)
    parser.add_argument("--baseline-time-steps", type=int, default=20)
    parser.add_argument("--resolution-sweep", type=int, nargs="*", default=[4, 7, 14, 28])
    parser.add_argument("--time-step-sweep", type=int, nargs="*", default=[5, 10, 20])
    parser.add_argument("--quant-bits", type=int, nargs="*", default=[2, 4, 8])
    parser.add_argument("--current-scale-na", type=float, default=2500.0)
    parser.add_argument("--dt-us", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = Section6Config(
        surrogate_epochs=args.surrogate_epochs,
        stdp_epochs=args.stdp_epochs,
        classifier_epochs=args.classifier_epochs,
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

    baseline_accuracy: dict[str, dict[str, float]] = {"surrogate": {}, "stdp": {}}
    trained_models: dict[tuple[str, str], nn.Module] = {}

    for training_method in ("surrogate", "stdp"):
        for neuron_kind in ("lif", "circuit"):
            print(f"Running baseline {training_method} / {neuron_kind} experiment...")
            if training_method == "surrogate":
                model, results = run_surrogate_experiment(
                    neuron_kind=neuron_kind,
                    image_size=cfg.resolution_baseline,
                    time_steps=cfg.time_steps_baseline,
                    cfg=cfg,
                    neuron_params=neuron_params,
                    device=device,
                )
            else:
                model, results = run_stdp_experiment(
                    neuron_kind=neuron_kind,
                    image_size=cfg.resolution_baseline,
                    time_steps=cfg.time_steps_baseline,
                    cfg=cfg,
                    neuron_params=neuron_params,
                    device=device,
                )
            baseline_accuracy[training_method][neuron_kind] = results["final_test_metrics"]["test_accuracy"]
            trained_models[(training_method, neuron_kind)] = model

    resolution_results = {tm: {"lif": [], "circuit": []} for tm in ("surrogate", "stdp")}
    for size in args.resolution_sweep:
        for training_method in ("surrogate", "stdp"):
            for neuron_kind in ("lif", "circuit"):
                print(f"Resolution sweep {size}x{size}: {training_method} / {neuron_kind}...")
                if training_method == "surrogate":
                    _, results = run_surrogate_experiment(neuron_kind, size, cfg.time_steps_baseline, cfg, neuron_params, device)
                else:
                    _, results = run_stdp_experiment(neuron_kind, size, cfg.time_steps_baseline, cfg, neuron_params, device)
                resolution_results[training_method][neuron_kind].append(
                    {"image_size": size, "accuracy": results["final_test_metrics"]["test_accuracy"]}
                )

    timestep_results = {tm: {"lif": [], "circuit": []} for tm in ("surrogate", "stdp")}
    for time_steps in args.time_step_sweep:
        for training_method in ("surrogate", "stdp"):
            for neuron_kind in ("lif", "circuit"):
                print(f"Time-step sweep T={time_steps}: {training_method} / {neuron_kind}...")
                if training_method == "surrogate":
                    _, results = run_surrogate_experiment(neuron_kind, cfg.resolution_baseline, time_steps, cfg, neuron_params, device)
                else:
                    _, results = run_stdp_experiment(neuron_kind, cfg.resolution_baseline, time_steps, cfg, neuron_params, device)
                timestep_results[training_method][neuron_kind].append(
                    {"time_steps": time_steps, "accuracy": results["final_test_metrics"]["test_accuracy"]}
                )

    quant_results = {tm: {"lif": [], "circuit": []} for tm in ("surrogate", "stdp")}
    _, quant_test_loader = make_mnist_loaders(
        image_size=cfg.resolution_baseline,
        batch_size=cfg.batch_size,
        train_subset=cfg.train_subset,
        test_subset=cfg.test_subset,
        num_workers=cfg.num_workers,
    )
    for bits in args.quant_bits:
        print(f"Quantization sweep {bits} bits...")
        for training_method in ("surrogate", "stdp"):
            for neuron_kind in ("lif", "circuit"):
                metrics = evaluate_quantized_inplace(
                    trained_models[(training_method, neuron_kind)],
                    bits,
                    training_method,
                    quant_test_loader,
                    cfg.time_steps_baseline,
                    device,
                )
                quant_results[training_method][neuron_kind].append({"bits": bits, "accuracy": metrics["accuracy"]})

    training_method_plot = save_training_method_bar_plot(baseline_accuracy)
    resolution_plot = save_factor_plot(
        resolution_results,
        factor_key="image_size",
        xlabel="Image resolution",
        title="Section 6: Resolution sweep",
        filename="section6_resolution_sweep.png",
    )
    timestep_plot = save_factor_plot(
        timestep_results,
        factor_key="time_steps",
        xlabel="Time steps",
        title="Section 6: Temporal encoding sweep",
        filename="section6_timestep_sweep.png",
    )
    quant_plot = save_factor_plot(
        quant_results,
        factor_key="bits",
        xlabel="Quantization bits",
        title="Section 6: Post-training quantization sweep",
        filename="section6_quantization_sweep.png",
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
            "c_fb_F": neuron_params.c_fb,
        },
        "baseline_accuracy_by_training_method": baseline_accuracy,
        "resolution_sweep": resolution_results,
        "time_step_sweep": timestep_results,
        "quantization_sweep": quant_results,
        "artifacts": {
            "training_method_accuracy": str(training_method_plot),
            "resolution_sweep": str(resolution_plot),
            "time_step_sweep": str(timestep_plot),
            "quantization_sweep": str(quant_plot),
        },
        "analysis_notes": [
            "Surrogate training is supervised end-to-end with gradient descent.",
            "The STDP branch uses unsupervised competitive spike-based feature learning followed by a supervised linear classifier on frozen features.",
            "The same hidden-layer width is used across the compared models in each run.",
        ],
    }

    summary_path = OUTPUT_DIR / "section6_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Section 6 workflow complete.")
    print(f"Summary: {summary_path}")
    for name, path in summary["artifacts"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
