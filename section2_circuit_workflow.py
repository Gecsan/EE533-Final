from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

import matplotlib.pyplot as plt
import numpy as np

from circuit_if_neuron import (
    CadenceNeuronParameters,
    CircuitIFNode,
    constant_current_waveform,
    fit_cadence_parameters_from_sweep,
    load_cadence_fi_sweep,
    load_cadence_trace_csv,
    pulsed_current_waveform,
    simulate_neuron,
)

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional for non-torch use
    torch = None
    nn = None


ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT / "documents"
OUTPUT_DIR = ROOT / "outputs" / "section2"


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


def reset_module_state(module: nn.Module) -> None:
    for child in module.modules():
        if hasattr(child, "reset"):
            child.reset()


class CircuitMNISTSNN(nn.Module):
    """Minimal SNN that replaces LIF nodes with the fitted circuit IF node."""

    def __init__(
        self,
        neuron_params: CadenceNeuronParameters,
        dt_s: float,
        current_scale_a: float = 100e-9,
        hidden_features: int = 128,
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_features)
        self.sn1 = CircuitIFNode(neuron_params, dt_s=dt_s)
        self.fc2 = nn.Linear(hidden_features, 10)
        self.sn2 = CircuitIFNode(neuron_params, dt_s=dt_s)
        self.current_scale_a = current_scale_a

    def reset(self) -> None:
        self.sn1.reset()
        self.sn2.reset()

    def forward(self, x: torch.Tensor, time_window: int = 10) -> torch.Tensor:
        x = self.flatten(x).clamp(0.0, 1.0)
        out_spike = 0.0
        for _ in range(time_window):
            encoded = torch.bernoulli(x)
            current_1 = torch.relu(self.fc1(encoded)) * self.current_scale_a
            spike_1 = self.sn1(current_1)
            current_2 = torch.relu(self.fc2(spike_1)) * self.current_scale_a
            out_spike = out_spike + self.sn2(current_2)
        return out_spike / time_window


def plot_fi_curve(currents_a: np.ndarray, measured_rates_hz: np.ndarray, predicted_rates_hz: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(currents_a * 1e9, measured_rates_hz, "o", label="Cadence measured")
    ax.semilogx(currents_a * 1e9, predicted_rates_hz, linewidth=2, label="Python model")
    ax.set_xlabel("Input current (nA)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Section 2a: Measured vs modeled f-I curve")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    output_path = OUTPUT_DIR / "section2_fi_curve.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_constant_current_overlay(
    params: CadenceNeuronParameters,
    current_a: float,
    measurement,
    window_s: float = 2e-3,
) -> tuple[Path, float]:
    dt_s = float(np.median(np.diff(measurement.time_s)))
    metric_window_s = max(window_s, 10e-3)
    sim = simulate_neuron(
        params=params,
        duration_s=metric_window_s,
        dt_s=dt_s,
        input_current=constant_current_waveform(current_a),
        vmem0=params.v_reset,
    )

    measured_mask = measurement.time_s <= window_s
    model_mask = sim["time_s"] <= window_s

    measured_spikes = measurement.spike_times_s[measurement.spike_times_s <= metric_window_s]
    model_spikes = sim["spike_times_s"][sim["spike_times_s"] <= metric_window_s]
    measured_isi_s = np.diff(measured_spikes)
    model_isi_s = np.diff(model_spikes)
    isi_count = min(measured_isi_s.size, model_isi_s.size)
    isi_error_s = float(np.mean(np.abs(measured_isi_s[:isi_count] - model_isi_s[:isi_count]))) if isi_count > 0 else float("inf")

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(sim["time_s"][model_mask] * 1e3, sim["input_current_a"][model_mask] * 1e9, color="tab:blue")
    axes[0].set_ylabel("Iin (nA)")
    axes[0].set_title(f"Section 2b: {current_a * 1e9:.1f} nA constant-current dynamics")

    axes[1].plot(measurement.time_s[measured_mask] * 1e3, measurement.vmem_v[measured_mask], label="Cadence vmem")
    axes[1].plot(sim["time_s"][model_mask] * 1e3, sim["vmem_v"][model_mask], "--", label="Model vmem")
    axes[1].axhline(params.v_threshold, linestyle=":", color="black", linewidth=1, label="threshold")
    axes[1].axhline(params.v_reset, linestyle="--", color="gray", linewidth=1, label="reset")
    axes[1].set_ylabel("Vmem (V)")
    axes[1].legend(loc="upper right")

    axes[2].plot(measurement.time_s[measured_mask] * 1e3, measurement.vout_v[measured_mask], label="Cadence vout")
    axes[2].plot(sim["time_s"][model_mask] * 1e3, sim["vout_v"][model_mask], "--", label="Model vout")
    axes[2].set_ylabel("Vout (V)")
    axes[2].set_xlabel("Time (ms)")
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    output_path = OUTPUT_DIR / "section2_constant_current_overlay.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path, isi_error_s


def plot_measured_pulse_trace() -> Path:
    pulse_measurement = load_cadence_trace_csv(DOCUMENTS_DIR / "533FINALPulseCurrentData.csv")
    window_s = 3e-3
    mask = pulse_measurement.time_s <= window_s

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    axes[0].plot(pulse_measurement.time_s[mask] * 1e3, pulse_measurement.vmem_v[mask], color="tab:green")
    axes[0].axhline(pulse_measurement.threshold_v, linestyle=":", color="black", linewidth=1, label="estimated threshold")
    axes[0].axhline(pulse_measurement.reset_v, linestyle="--", color="gray", linewidth=1, label="estimated reset")
    axes[0].set_ylabel("Vmem (V)")
    axes[0].set_title("Measured pulse-response trace from Cadence")
    axes[0].legend(loc="upper right")

    axes[1].plot(pulse_measurement.time_s[mask] * 1e3, pulse_measurement.vout_v[mask], color="tab:red")
    axes[1].set_ylabel("Vout (V)")
    axes[1].set_xlabel("Time (ms)")

    fig.tight_layout()
    output_path = OUTPUT_DIR / "section2_measured_pulse_trace.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_synthetic_pulse_demo(params: CadenceNeuronParameters) -> Path:
    pulse_demo_params = params.with_updates(
        leak_conductance_s=500.0,
        pulse_width_s=40e-6,
    )
    input_current_a = 10e-9
    sim = simulate_neuron(
        params=pulse_demo_params,
        duration_s=3e-3,
        dt_s=0.1e-6,
        reset_fall_s=30e-6,
        spike_peak_v=2.15,
        input_current=constant_current_waveform(input_current_a),
        vmem0=pulse_demo_params.v_reset,
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(sim["time_s"] * 1e3, sim["input_current_a"] * 1e9, color="tab:blue")
    axes[0].set_ylabel("Ichg (nA)")
    axes[0].set_title("Synthetic charge-spike-reset demo")

    axes[1].plot(sim["time_s"] * 1e3, sim["vmem_v"], color="tab:green")
    axes[1].axhline(pulse_demo_params.v_threshold, linestyle=":", color="black", linewidth=1)
    axes[1].axhline(pulse_demo_params.v_reset, linestyle="--", color="gray", linewidth=1)
    axes[1].set_ylabel("Vmem (V)")

    axes[2].plot(sim["time_s"] * 1e3, sim["vout_v"], color="tab:red")
    axes[2].set_ylabel("Vout (V)")
    axes[2].set_xlabel("Time (ms)")

    fig.tight_layout()
    output_path = OUTPUT_DIR / "section2_synthetic_pulse_demo.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def run_snn_smoke_test(params: CadenceNeuronParameters) -> dict[str, object]:
    if torch is None or nn is None or CircuitIFNode is None:
        return {"status": "skipped", "reason": "torch is not available"}

    torch.manual_seed(0)
    model = CircuitMNISTSNN(
        neuron_params=params,
        dt_s=5e-6,
        current_scale_a=500e-9,
        hidden_features=128,
    )
    with torch.no_grad():
        nn.init.uniform_(model.fc1.weight, a=0.0, b=0.02)
        nn.init.uniform_(model.fc2.weight, a=0.0, b=0.05)
        model.fc1.bias.fill_(0.15)
        model.fc2.bias.fill_(0.1)

    images = torch.full((8, 1, 28, 28), 0.7)
    reset_module_state(model)
    with torch.no_grad():
        outputs = model(images, time_window=10)

    return {
        "status": "ok",
        "output_shape": list(outputs.shape),
        "mean_output_spike_rate": float(outputs.mean().item()),
        "max_output_spike_rate": float(outputs.max().item()),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_params = build_base_params()
    sweep_measurements = load_cadence_fi_sweep(DOCUMENTS_DIR / "EE533Final1C.csv")
    fit = fit_cadence_parameters_from_sweep(base_params, sweep_measurements)

    measurement_by_current = {round(m.current_a, 15): m for m in sweep_measurements}
    representative_measurement = measurement_by_current[round(10e-9, 15)]

    fi_path = plot_fi_curve(fit.currents_a, fit.measured_rates_hz, fit.predicted_rates_hz)
    overlay_path, isi_error_s = plot_constant_current_overlay(
        params=fit.params,
        current_a=10e-9,
        measurement=representative_measurement,
    )
    measured_pulse_path = plot_measured_pulse_trace()
    synthetic_pulse_path = plot_synthetic_pulse_demo(fit.params)
    snn_smoke = run_snn_smoke_test(fit.params)

    summary = {
        "fitted_params": {
            "vdd_V": fit.params.vdd,
            "c_mem_F": fit.params.c_mem,
            "c_fb_F": fit.params.c_fb,
            "v_reset_V": fit.params.v_reset,
            "v_threshold_V": fit.params.v_threshold,
            "refractory_s": fit.params.refractory_s,
            "pulse_width_s": fit.params.pulse_width_s,
            "leak_conductance_s": fit.params.leak_conductance_s,
            "feedback_cap_scale": fit.params.feedback_cap_scale,
            "current_gain": fit.params.current_gain,
            "total_capacitance_F": fit.params.total_capacitance,
            "delta_v_V": fit.params.delta_v,
        },
        "fit_metrics": {
            "rmse_hz": fit.rmse_hz,
            "mape_percent": fit.mape_percent,
            "mean_threshold_v": fit.mean_threshold_v,
            "mean_reset_v": fit.mean_reset_v,
            "mean_pulse_width_s": fit.mean_pulse_width_s,
            "constant_current_isi_error_us": isi_error_s * 1e6,
        },
        "measured_fi_curve": [
            {"current_nA": float(current * 1e9), "rate_hz": float(rate)}
            for current, rate in zip(fit.currents_a, fit.measured_rates_hz)
        ],
        "modeled_fi_curve": [
            {"current_nA": float(current * 1e9), "rate_hz": float(rate)}
            for current, rate in zip(fit.currents_a, fit.predicted_rates_hz)
        ],
        "snn_smoke_test": snn_smoke,
        "artifacts": {
            "fi_curve": str(fi_path),
            "constant_overlay": str(overlay_path),
            "measured_pulse_trace": str(measured_pulse_path),
            "synthetic_pulse_demo": str(synthetic_pulse_path),
        },
    }

    summary_path = OUTPUT_DIR / "section2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Section 2 workflow complete.")
    print(f"Summary: {summary_path}")
    for name, path in summary["artifacts"].items():
        print(f"{name}: {path}")
    print(
        "Fitted parameters: "
        f"v_reset={fit.params.v_reset:.4f} V, "
        f"v_threshold={fit.params.v_threshold:.4f} V, "
        f"refractory={fit.params.refractory_s * 1e6:.3f} us, "
        f"pulse_width={fit.params.pulse_width_s * 1e6:.3f} us, "
        f"current_gain={fit.params.current_gain:.4f}"
    )
    print(
        "Fit quality: "
        f"RMSE={fit.rmse_hz:.3f} Hz, "
        f"MAPE={fit.mape_percent:.3f} %, "
        f"10 nA ISI error={isi_error_s * 1e6:.3f} us"
    )
    print(f"SNN smoke test: {snn_smoke}")


if __name__ == "__main__":
    main()
