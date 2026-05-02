from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

import matplotlib.pyplot as plt
import numpy as np

from circuit_if_neuron import (
    CadenceNeuronParameters,
    extract_spike_edges_from_vout,
    load_cadence_trace_csv,
    nearest_neighbor_timing_error_s,
    simulate_neuron,
)


ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT / "documents"
OUTPUT_DIR = ROOT / "outputs" / "section3_validation"
SECTION2_SUMMARY = ROOT / "outputs" / "section2" / "section2_summary.json"


def load_section2_params() -> CadenceNeuronParameters:
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


def build_measurement(path: Path):
    return load_cadence_trace_csv(path)


def slice_measurement(measurement, duration_s: float):
    mask = measurement.time_s <= duration_s
    spike_mask = measurement.spike_times_s <= duration_s
    data = {
        "current_a": measurement.current_a,
        "time_s": measurement.time_s[mask],
        "vmem_v": measurement.vmem_v[mask],
        "vout_v": measurement.vout_v[mask],
        "spike_times_s": measurement.spike_times_s[spike_mask],
        "firing_rate_hz": measurement.firing_rate_hz,
        "pulse_width_s": measurement.pulse_width_s,
        "threshold_v": measurement.threshold_v,
        "reset_v": measurement.reset_v,
    }
    return type("MeasurementSlice", (), data)()


def infer_effective_current_waveform(
    params: CadenceNeuronParameters,
    time_s: np.ndarray,
    vmem_v: np.ndarray,
    vout_v: np.ndarray,
) -> np.ndarray:
    if time_s.size < 2:
        raise ValueError("Trace must contain at least two samples")

    dt = np.diff(time_s)
    dv = np.diff(vmem_v)
    prev_v = vmem_v[:-1]
    leak_term = params.leak_conductance_s * (prev_v - params.v_reset) * dt
    denom = params.current_gain * dt
    inferred = np.zeros_like(vmem_v)

    valid = denom > 0
    inferred_steps = np.zeros_like(dv)
    inferred_steps[valid] = (dv[valid] + leak_term[valid]) * params.total_capacitance / denom[valid]

    rising_edges_s, falling_edges_s = extract_spike_edges_from_vout(time_s, vout_v)
    reset_mask = np.zeros_like(vmem_v, dtype=bool)
    for rise_s, fall_s in zip(rising_edges_s, falling_edges_s):
        reset_mask |= (time_s >= rise_s) & (time_s <= fall_s)

    valid_points = np.isfinite(inferred_steps)
    inferred[:-1] = inferred_steps
    inferred[-1] = inferred[-2] if inferred.size > 1 else 0.0

    masked = ~reset_mask
    if np.count_nonzero(masked) > 2:
        valid_idx = np.flatnonzero(masked)
        inferred = np.interp(np.arange(time_s.size), valid_idx, inferred[valid_idx])

    return inferred


def waveform_from_samples(time_s: np.ndarray, current_a: np.ndarray):
    def _waveform(t_query: np.ndarray) -> np.ndarray:
        return np.interp(t_query, time_s, current_a)

    return _waveform


def compute_metrics(reference_spikes_s: np.ndarray, candidate_spikes_s: np.ndarray) -> dict[str, float]:
    ref_rate_hz = 0.0
    cand_rate_hz = 0.0
    if reference_spikes_s.size >= 2:
        ref_rate_hz = float((reference_spikes_s.size - 1) / (reference_spikes_s[-1] - reference_spikes_s[0]))
    if candidate_spikes_s.size >= 2:
        cand_rate_hz = float((candidate_spikes_s.size - 1) / (candidate_spikes_s[-1] - candidate_spikes_s[0]))
    timing_error_s = nearest_neighbor_timing_error_s(reference_spikes_s, candidate_spikes_s)
    return {
        "reference_rate_hz": ref_rate_hz,
        "model_rate_hz": cand_rate_hz,
        "rate_error_hz": abs(cand_rate_hz - ref_rate_hz),
        "mean_spike_timing_error_us": timing_error_s * 1e6,
    }


def plot_validation_case(
    label: str,
    measurement,
    inferred_current_a: np.ndarray,
    sim: dict[str, np.ndarray],
) -> Path:
    window_s = min(3e-3, measurement.time_s[-1] - measurement.time_s[0])
    measured_mask = measurement.time_s <= window_s
    sim_mask = sim["time_s"] <= window_s

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(measurement.time_s[measured_mask] * 1e3, inferred_current_a[measured_mask] * 1e9, color="tab:blue")
    axes[0].set_ylabel("Iin (nA)")
    axes[0].set_title(f"Section 3: {label} validation with Section 2 fitted neuron")

    axes[1].plot(measurement.time_s[measured_mask] * 1e3, measurement.vmem_v[measured_mask], label="Cadence vmem")
    axes[1].plot(sim["time_s"][sim_mask] * 1e3, sim["vmem_v"][sim_mask], "--", label="Python vmem")
    axes[1].set_ylabel("Vmem (V)")
    axes[1].axhline(sim["params"].v_threshold, linestyle=":", color="black", linewidth=1, label="threshold")
    axes[1].axhline(sim["params"].v_reset, linestyle="--", color="gray", linewidth=1, label="reset")
    axes[1].legend(loc="upper right")

    axes[2].plot(measurement.time_s[measured_mask] * 1e3, measurement.vout_v[measured_mask], label="Cadence vout")
    axes[2].plot(sim["time_s"][sim_mask] * 1e3, sim["vout_v"][sim_mask], "--", label="Python vout")
    axes[2].set_ylabel("Vout (V)")
    axes[2].set_xlabel("Time (ms)")
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    output_path = OUTPUT_DIR / f"section3_{label.lower().replace(' ', '_')}_overlay.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def run_case(
    label: str,
    csv_name: str,
    params: CadenceNeuronParameters,
    duration_s: float = 50e-3,
) -> tuple[dict[str, object], Path]:
    measurement = build_measurement(DOCUMENTS_DIR / csv_name)
    measurement = slice_measurement(measurement, min(duration_s, float(measurement.time_s[-1])))
    inferred_current_a = infer_effective_current_waveform(
        params=params,
        time_s=measurement.time_s,
        vmem_v=measurement.vmem_v,
        vout_v=measurement.vout_v,
    )
    sim = simulate_neuron(
        params=params,
        duration_s=float(measurement.time_s[-1]),
        dt_s=float(np.median(np.diff(measurement.time_s))),
        input_current=waveform_from_samples(measurement.time_s, inferred_current_a),
        vmem0=float(measurement.vmem_v[0]),
    )
    sim["params"] = params
    metrics = compute_metrics(measurement.spike_times_s, sim["spike_times_s"])
    current_stats = {
        "mean_current_nA": float(np.mean(inferred_current_a) * 1e9),
        "std_current_nA": float(np.std(inferred_current_a) * 1e9),
        "min_current_nA": float(np.min(inferred_current_a) * 1e9),
        "max_current_nA": float(np.max(inferred_current_a) * 1e9),
    }
    overlay_path = plot_validation_case(label, measurement, inferred_current_a, sim)
    summary = {
        "label": label,
        "csv": csv_name,
        "validation_window_ms": float(measurement.time_s[-1] * 1e3),
        "metrics": metrics,
        "current_stats": current_stats,
        "spike_count_cadence": int(measurement.spike_times_s.size),
        "spike_count_python": int(sim["spike_times_s"].size),
    }
    return summary, overlay_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    params = load_section2_params()

    const_summary, const_plot = run_case("Constant Current", "533FINALConstCurrentData.csv", params)
    pulse_summary, pulse_plot = run_case("Pulsed Current", "533FINALPulseCurrentData.csv", params)

    summary = {
        "section2_params": asdict(params),
        "constant_current": const_summary,
        "pulsed_current": pulse_summary,
        "artifacts": {
            "constant_overlay": str(const_plot),
            "pulse_overlay": str(pulse_plot),
        },
        "notes": [
            "Section 3 reuses the exact fitted neuron parameters from Section 2.",
            "The input-current waveform is inferred directly from the measured Cadence trace because the exported CSVs do not include an explicit input-current column.",
            "If a dedicated Cadence pulse-current source export becomes available, replace the inferred waveform with the exact exported source waveform.",
        ],
    }

    summary_path = OUTPUT_DIR / "section3_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Section 3 validation workflow complete.")
    print(f"Summary: {summary_path}")
    print(f"constant_overlay: {const_plot}")
    print(f"pulse_overlay: {pulse_plot}")
    print(
        "Constant timing/rate: "
        f"{const_summary['metrics']['mean_spike_timing_error_us']:.3f} us, "
        f"{const_summary['metrics']['rate_error_hz']:.3f} Hz"
    )
    print(
        "Pulse timing/rate: "
        f"{pulse_summary['metrics']['mean_spike_timing_error_us']:.3f} us, "
        f"{pulse_summary['metrics']['rate_error_hz']:.3f} Hz"
    )


if __name__ == "__main__":
    main()
