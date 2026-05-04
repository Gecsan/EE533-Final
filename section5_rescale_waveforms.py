from __future__ import annotations

import json
from pathlib import Path

from section4_training_workflow import load_circuit_params
from circuit_if_neuron import simulate_neuron


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "outputs" / "section5"
OUT_DIR = ROOT / "outputs" / "section5_scaled_4na"
SUMMARY_PATH = SRC_DIR / "section5_waveform_summary.json"


def read_pwlf(path: Path) -> list[tuple[float, float]]:
    points = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        t_str, i_str = line.split()
        points.append((float(t_str), float(i_str)))
    return points


def write_pwlf(path: Path, points: list[tuple[float, float]]) -> None:
    lines = ["# time_s current_A"]
    for time_s, current_a in points:
        lines.append(f"{time_s:.12e} {current_a:.12e}")
    path.write_text("\n".join(lines) + "\n")


def write_spectre(path: Path, instance_name: str, points: list[tuple[float, float]]) -> None:
    wave_tokens = " ".join(f"{t:.12e} {i:.12e}" for t, i in points)
    path.write_text(
        f"// Scaled Spectre isource PWL for {instance_name}\n"
        f"{instance_name} (iin 0) isource type=pwl wave=[ {wave_tokens} ]\n"
    )


def waveform_from_points(points: list[tuple[float, float]]):
    import numpy as np

    times = np.array([p[0] for p in points], dtype=float)
    currents = np.array([p[1] for p in points], dtype=float)

    def _waveform(t_query):
        return np.interp(t_query, times, currents)

    return _waveform


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text())
    params = load_circuit_params()
    target_mean_na = 4.0

    scaled_entries = []
    for entry, ref in zip(summary["waveforms"], summary["python_reference"]):
        neuron_idx = entry["neuron_index"]
        pwlf_src = Path(entry["pwlf"])
        points = read_pwlf(pwlf_src)
        source_mean_na = ref["mean_current_nA"]
        scale = target_mean_na / source_mean_na if source_mean_na != 0 else 1.0
        scaled_points = [(t, i * scale) for t, i in points]

        pwlf_out = OUT_DIR / f"section5_neuron_{neuron_idx}_input_4nAavg.pwlf"
        scs_out = OUT_DIR / f"section5_neuron_{neuron_idx}_source_4nAavg.scs"
        write_pwlf(pwlf_out, scaled_points)
        write_spectre(scs_out, f"I_NEURON_{neuron_idx}_SCALED", scaled_points)

        duration_s = scaled_points[-1][0]
        dt_s = summary["time_step_dt_us"] * 1e-6
        sim = simulate_neuron(
            params=params,
            duration_s=duration_s,
            dt_s=dt_s,
            input_current=waveform_from_points(scaled_points),
            vmem0=params.v_reset,
        )
        scaled_entries.append(
            {
                "neuron_index": neuron_idx,
                "scale_factor": scale,
                "pwlf": str(pwlf_out),
                "spectre": str(scs_out),
                "mean_current_nA": target_mean_na,
                "max_current_nA": max(i for _, i in scaled_points) * 1e9,
                "python_spike_times_us": [float(x * 1e6) for x in sim["spike_times_s"]],
            }
        )

    out_summary = {
        "source_summary": str(SUMMARY_PATH),
        "target_mean_current_nA": target_mean_na,
        "notes": [
            "These waveforms are amplitude-scaled versions of the original Section 5 hidden-neuron currents.",
            "They were generated to reduce saturation in Cadence.",
            "At this lower current range, the single-neuron Python replay may produce few or zero spikes over the original 100 us window.",
        ],
        "scaled_waveforms": scaled_entries,
    }
    out_path = OUT_DIR / "section5_scaled_4na_summary.json"
    out_path.write_text(json.dumps(out_summary, indent=2))

    print("Scaled Section 5 waveforms complete.")
    print(f"Summary: {out_path}")
    for entry in scaled_entries:
        print(f"neuron_{entry['neuron_index']}: {entry['pwlf']}")


if __name__ == "__main__":
    main()
