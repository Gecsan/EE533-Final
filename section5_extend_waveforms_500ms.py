from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from circuit_if_neuron import simulate_neuron
from section4_training_workflow import load_circuit_params
from section5_compare_cadence import read_pwlf, waveform_from_points


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "outputs" / "section5_scaled_4na"
SRC_SUMMARY = SRC_DIR / "section5_scaled_4na_summary.json"
OUT_DIR = ROOT / "outputs" / "section5_scaled_4na_500ms_hold"


def write_pwlf(path: Path, points: list[tuple[float, float]]) -> None:
    lines = ["# time_s current_A"]
    for time_s, current_a in points:
        lines.append(f"{time_s:.12e} {current_a:.12e}")
    path.write_text("\n".join(lines) + "\n")


def write_spectre(path: Path, instance_name: str, points: list[tuple[float, float]]) -> None:
    wave_tokens = " ".join(f"{t:.12e} {i:.12e}" for t, i in points)
    path.write_text(
        f"// 500 ms hold-current Spectre isource PWL for {instance_name}\n"
        f"{instance_name} (iin 0) isource type=pwl wave=[ {wave_tokens} ]\n"
    )


def measure_rate_window(spike_times_s: np.ndarray, start_s: float, stop_s: float) -> float:
    spikes = spike_times_s[(spike_times_s >= start_s) & (spike_times_s <= stop_s)]
    if len(spikes) >= 2:
        interval = spikes[-1] - spikes[0]
        if interval > 0:
            return float((len(spikes) - 1) / interval)
    window = stop_s - start_s
    return float(len(spikes) / window) if window > 0 else 0.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SRC_SUMMARY.read_text())

    target_stop_s = 0.5
    current_gain_scale = 0.92
    refractory_scale = 0.8
    initial_vmem_v = 0.0
    params_base = load_circuit_params()
    params = params_base.with_updates(
        current_gain=params_base.current_gain * current_gain_scale,
        refractory_s=params_base.refractory_s * refractory_scale,
    )

    entries = []
    fig, axes = plt.subplots(len(summary["scaled_waveforms"]), 1, figsize=(10, 2.6 * len(summary["scaled_waveforms"])))
    if len(summary["scaled_waveforms"]) == 1:
        axes = [axes]

    for ax, entry in zip(axes, summary["scaled_waveforms"]):
        neuron_idx = int(entry["neuron_index"])
        src_pwlf = Path(entry["pwlf"])
        points = read_pwlf(src_pwlf)
        last_t, last_i = points[-1]
        if target_stop_s > last_t:
            extended_points = points + [(target_stop_s, last_i)]
        else:
            extended_points = points

        pwlf_out = OUT_DIR / f"section5_neuron_{neuron_idx}_input_500ms_hold.pwlf"
        scs_out = OUT_DIR / f"section5_neuron_{neuron_idx}_source_500ms_hold.scs"
        write_pwlf(pwlf_out, extended_points)
        write_spectre(scs_out, f"I_NEURON_{neuron_idx}_500MS", extended_points)

        sim = simulate_neuron(
            params=params,
            duration_s=target_stop_s,
            dt_s=100e-9,
            input_current=waveform_from_points(extended_points),
            vmem0=initial_vmem_v,
        )
        spike_times_s = sim["spike_times_s"]
        entries.append(
            {
                "neuron_index": neuron_idx,
                "pwlf": str(pwlf_out),
                "spectre": str(scs_out),
                "initial_vmem_v": initial_vmem_v,
                "hold_current_nA": float(last_i * 1e9),
                "spike_count_0_500ms": int(len(spike_times_s)),
                "spike_times_first_20ms_us": [float(x * 1e6) for x in spike_times_s[spike_times_s <= 0.02]],
                "rate_0_20ms_hz": measure_rate_window(spike_times_s, 0.0, 0.02),
                "rate_200_500ms_hz": measure_rate_window(spike_times_s, 0.2, 0.5),
            }
        )

        plot_mask = sim["time_s"] <= 0.02
        ax.plot(sim["time_s"][plot_mask] * 1e3, sim["input_current_a"][plot_mask] * 1e9, color="tab:blue")
        ax.set_ylabel(f"n{neuron_idx}\nIin (nA)")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Section 5 500 ms hold-current waveforms (first 20 ms shown)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "section5_500ms_hold_waveforms_first20ms.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    out_summary = {
        "source_summary": str(SRC_SUMMARY),
        "target_stop_s": target_stop_s,
        "hold_behavior": "The last PWL current value is explicitly held constant through 500 ms.",
        "python_replay": {
            "current_gain_scale": current_gain_scale,
            "refractory_scale": refractory_scale,
            "dt_ns": 100.0,
            "initial_vmem_v": initial_vmem_v,
        },
        "waveforms": entries,
    }
    summary_path = OUT_DIR / "section5_scaled_4na_500ms_hold_summary.json"
    summary_path.write_text(json.dumps(out_summary, indent=2))

    print(f"500 ms hold-current waveforms complete: {summary_path}")
    for entry in entries:
        print(
            f"neuron_{entry['neuron_index']}: hold={entry['hold_current_nA']:.3f} nA "
            f"spikes_0_500ms={entry['spike_count_0_500ms']} rate_200_500ms={entry['rate_200_500ms_hz']:.2f} Hz"
        )


if __name__ == "__main__":
    main()
