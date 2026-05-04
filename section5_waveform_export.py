from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

import matplotlib.pyplot as plt
import torch

from section4_training_workflow import (
    CircuitMNISTSNN,
    Section4Config,
    load_circuit_params,
    make_loaders,
    reset_module_state,
    set_seed,
    train_model,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "section5"
SECTION4_SUMMARY = ROOT / "outputs" / "section4" / "section4_summary.json"


def load_section4_best_config() -> Section4Config:
    summary = json.loads(SECTION4_SUMMARY.read_text())
    cfg = summary["config"]
    best_hidden = summary["best_model"]["hidden_size"]
    return Section4Config(
        image_size=cfg["image_size"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        current_scale_a=cfg["current_scale_a"],
        dt_s=cfg["dt_s"],
        time_steps=cfg["time_steps"],
        hidden_sizes=(best_hidden,),
        train_subset=cfg["train_subset"],
        test_subset=cfg["test_subset"],
        num_workers=cfg["num_workers"],
        seed=cfg["seed"],
    )


def select_reference_sample(
    model: CircuitMNISTSNN,
    test_loader,
    time_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    model.eval()
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            reset_module_state(model)
            outputs = model(images, time_steps=time_steps)
            preds = outputs.argmax(dim=1)
            correct = preds == labels
            if correct.any():
                idx = int(correct.nonzero(as_tuple=False)[0].item())
                return images[idx : idx + 1].cpu(), int(labels[idx].item()), int(preds[idx].item())
    raise RuntimeError("Could not find a correctly classified sample in the test loader")


def run_hidden_trace(
    model: CircuitMNISTSNN,
    image: torch.Tensor,
    time_steps: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model.eval()
    x = image.to(device).view(1, -1).clamp(0.0, 1.0)
    reset_module_state(model)

    encoded_steps = []
    hidden_currents = []
    hidden_spikes = []
    hidden_vmem = []
    output_currents = []
    output_spikes = []

    with torch.inference_mode():
        for _ in range(time_steps):
            encoded = torch.bernoulli(x)
            current_1 = torch.relu(model.fc1(encoded)) * model.current_scale_a
            spike_1 = model.sn1(current_1)
            current_2 = torch.relu(model.fc2(spike_1)) * model.current_scale_a
            spike_2 = model.sn2(current_2)

            encoded_steps.append(encoded.squeeze(0).cpu())
            hidden_currents.append(current_1.squeeze(0).cpu())
            hidden_spikes.append(spike_1.squeeze(0).cpu())
            hidden_vmem.append(model.sn1.v.squeeze(0).cpu())
            output_currents.append(current_2.squeeze(0).cpu())
            output_spikes.append(spike_2.squeeze(0).cpu())

    return {
        "encoded_steps": torch.stack(encoded_steps, dim=0),
        "hidden_currents": torch.stack(hidden_currents, dim=0),
        "hidden_spikes": torch.stack(hidden_spikes, dim=0),
        "hidden_vmem": torch.stack(hidden_vmem, dim=0),
        "output_currents": torch.stack(output_currents, dim=0),
        "output_spikes": torch.stack(output_spikes, dim=0),
    }


def choose_four_neurons(hidden_currents: torch.Tensor, hidden_spikes: torch.Tensor) -> list[int]:
    scores = hidden_currents.sum(dim=0) + 10.0 * hidden_spikes.sum(dim=0)
    top = torch.topk(scores, k=4).indices.tolist()
    return sorted(int(i) for i in top)


def waveform_to_pwlf_points(currents_a: torch.Tensor, dt_s: float, epsilon_s: float) -> list[tuple[float, float]]:
    currents = [float(v) for v in currents_a.tolist()]
    points: list[tuple[float, float]] = []
    for idx, current in enumerate(currents):
        t_start = idx * dt_s
        t_end = (idx + 1) * dt_s
        if idx == 0:
            points.append((t_start, current))
        points.append((max(t_end - epsilon_s, t_start), current))
        if idx < len(currents) - 1:
            points.append((t_end, float(currents[idx + 1])))
        else:
            points.append((t_end, current))
    deduped: list[tuple[float, float]] = []
    for time_s, current_a in points:
        if deduped and abs(deduped[-1][0] - time_s) < 1e-18 and abs(deduped[-1][1] - current_a) < 1e-18:
            continue
        deduped.append((time_s, current_a))
    return deduped


def write_pwlf_file(path: Path, points: list[tuple[float, float]]) -> None:
    lines = ["# time_s current_A"]
    for time_s, current_a in points:
        lines.append(f"{time_s:.12e} {current_a:.12e}")
    path.write_text("\n".join(lines) + "\n")


def write_spectre_include(path: Path, instance_name: str, points: list[tuple[float, float]]) -> None:
    wave_tokens = " ".join(f"{time_s:.12e} {current_a:.12e}" for time_s, current_a in points)
    text = (
        f"// Spectre isource PWL for {instance_name}\n"
        f"{instance_name} (iin 0) isource type=pwl wave=[ {wave_tokens} ]\n"
    )
    path.write_text(text)


def plot_four_waveforms(time_s: torch.Tensor, hidden_currents: torch.Tensor, selected_neurons: list[int]) -> Path:
    fig, axes = plt.subplots(len(selected_neurons), 1, figsize=(9, 7), sharex=True)
    for ax, neuron_idx in zip(axes, selected_neurons):
        ax.step(time_s * 1e6, hidden_currents[:, neuron_idx] * 1e9, where="post")
        ax.set_ylabel(f"N{neuron_idx}\nIin (nA)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (us)")
    fig.suptitle("Section 5: Hidden-neuron input current waveforms for Cadence")
    fig.tight_layout()
    out = OUTPUT_DIR / "section5_four_neuron_waveforms.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate four hidden-neuron PWLF waveforms for Section 5.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_section4_best_config()
    if args.epochs is not None:
        cfg = Section4Config(**{**cfg.__dict__, "epochs": args.epochs})
    if args.train_subset is not None:
        cfg = Section4Config(**{**cfg.__dict__, "train_subset": args.train_subset})
    if args.test_subset is not None:
        cfg = Section4Config(**{**cfg.__dict__, "test_subset": args.test_subset})
    if args.seed is not None:
        cfg = Section4Config(**{**cfg.__dict__, "seed": args.seed})

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

    hidden_size = cfg.hidden_sizes[0]
    model = CircuitMNISTSNN(
        input_size=cfg.image_size * cfg.image_size,
        hidden_features=hidden_size,
        neuron_params=neuron_params,
        current_scale_a=cfg.current_scale_a,
        dt_s=cfg.dt_s,
    )
    print(f"Training best Section 4 model again for Section 5 export (hidden={hidden_size})...")
    training = train_model(model, train_loader, test_loader, cfg, device)

    image, label, pred = select_reference_sample(model, test_loader, cfg.time_steps, device)
    trace = run_hidden_trace(model, image, cfg.time_steps, device)
    selected_neurons = choose_four_neurons(trace["hidden_currents"], trace["hidden_spikes"])

    dt_s = cfg.dt_s
    epsilon_s = min(dt_s * 0.1, 1e-9)
    time_s = torch.arange(cfg.time_steps + 1, dtype=torch.float64) * dt_s

    waveform_paths = []
    neuron_summaries = []
    for neuron_idx in selected_neurons:
        currents = trace["hidden_currents"][:, neuron_idx]
        spikes = trace["hidden_spikes"][:, neuron_idx]
        vmem = trace["hidden_vmem"][:, neuron_idx]
        points = waveform_to_pwlf_points(currents, dt_s=dt_s, epsilon_s=epsilon_s)

        pwlf_path = OUTPUT_DIR / f"section5_neuron_{neuron_idx}_input.pwlf"
        spectre_path = OUTPUT_DIR / f"section5_neuron_{neuron_idx}_source.scs"
        write_pwlf_file(pwlf_path, points)
        write_spectre_include(spectre_path, f"I_NEURON_{neuron_idx}", points)
        waveform_paths.append({"neuron_index": neuron_idx, "pwlf": str(pwlf_path), "spectre": str(spectre_path)})

        spike_steps = [int(i) for i in (spikes > 0).nonzero(as_tuple=False).view(-1).tolist()]
        neuron_summaries.append(
            {
                "neuron_index": neuron_idx,
                "spike_steps": spike_steps,
                "spike_times_us": [round((step + 1) * dt_s * 1e6, 6) for step in spike_steps],
                "mean_current_nA": float(currents.mean().item() * 1e9),
                "max_current_nA": float(currents.max().item() * 1e9),
                "final_vmem_V": float(vmem[-1].item()),
            }
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "selected_neurons": selected_neurons,
        },
        OUTPUT_DIR / "section5_model_checkpoint.pt",
    )

    image_path = OUTPUT_DIR / "section5_reference_image.pt"
    torch.save({"image": image, "label": label, "prediction": pred}, image_path)

    waveform_plot = plot_four_waveforms(time_s[:-1], trace["hidden_currents"], selected_neurons)

    summary = {
        "section4_retrain_final_metrics": training["final_test_metrics"],
        "reference_image": {
            "label": label,
            "prediction": pred,
            "tensor_path": str(image_path),
        },
        "selected_neurons": selected_neurons,
        "time_step_dt_us": cfg.dt_s * 1e6,
        "time_steps": cfg.time_steps,
        "waveforms": waveform_paths,
        "python_reference": neuron_summaries,
        "artifacts": {
            "waveform_plot": str(waveform_plot),
            "checkpoint": str(OUTPUT_DIR / "section5_model_checkpoint.pt"),
        },
        "cadence_notes": [
            "Each .pwlf file is a two-column time/current waveform in SI units.",
            "Each .scs file is a Spectre isource snippet using the same waveform points.",
            "Hook each waveform to one of the four replicated neuron instances and export vmem/vout for comparison.",
        ],
    }
    summary_path = OUTPUT_DIR / "section5_waveform_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Section 5 waveform export complete.")
    print(f"Summary: {summary_path}")
    print(f"Selected neurons: {selected_neurons}")
    for item in waveform_paths:
        print(f"neuron_{item['neuron_index']}_pwlf: {item['pwlf']}")
        print(f"neuron_{item['neuron_index']}_spectre: {item['spectre']}")


if __name__ == "__main__":
    main()
