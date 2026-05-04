from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from circuit_if_neuron import simulate_neuron
from section4_training_workflow import load_circuit_params


ROOT = Path(__file__).resolve().parent
DEFAULT_CADENCE_CSV = Path.home() / "Downloads" / "533FinalPart5Cadence.csv"
DEFAULT_REFERENCE = ROOT / "outputs" / "section5_scaled_4na" / "section5_scaled_4na_summary.json"
OUTPUT_DIR = ROOT / "outputs" / "section5_compare"


@dataclass
class NeuronTrace:
    neuron_slot: int
    vmem_time_s: np.ndarray
    vmem_v: np.ndarray
    vout_time_s: np.ndarray
    vout_v: np.ndarray


def sanitize_name(name: str) -> str:
    return name.strip().lstrip("/")


def parse_cadence_csv(path: Path) -> list[NeuronTrace]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = np.array([[float(x) for x in row] for row in reader], dtype=float)

    col_map: dict[str, np.ndarray] = {}
    for idx, name in enumerate(header):
        col_map[sanitize_name(name)] = data[:, idx]

    traces = []
    slot = 1
    while f"vmem{slot} X" in col_map and f"vout{slot} X" in col_map:
        traces.append(
            NeuronTrace(
                neuron_slot=slot,
                vmem_time_s=col_map[f"vmem{slot} X"],
                vmem_v=col_map[f"vmem{slot} Y"],
                vout_time_s=col_map[f"vout{slot} X"],
                vout_v=col_map[f"vout{slot} Y"],
            )
        )
        slot += 1
    return traces


def detect_spike_times(time_s: np.ndarray, vout_v: np.ndarray, threshold_v: float = 1.6) -> np.ndarray:
    above = vout_v >= threshold_v
    rising_edges = np.flatnonzero((~above[:-1]) & above[1:]) + 1
    return time_s[rising_edges]


def clip_trace(trace: NeuronTrace, stop_s: float | None) -> NeuronTrace:
    if stop_s is None:
        return trace
    vmem_mask = trace.vmem_time_s <= stop_s
    vout_mask = trace.vout_time_s <= stop_s
    return NeuronTrace(
        neuron_slot=trace.neuron_slot,
        vmem_time_s=trace.vmem_time_s[vmem_mask],
        vmem_v=trace.vmem_v[vmem_mask],
        vout_time_s=trace.vout_time_s[vout_mask],
        vout_v=trace.vout_v[vout_mask],
    )


def load_reference(summary_path: Path) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    summary = json.loads(summary_path.read_text())
    if "scaled_waveforms" in summary:
        source_summary_path = Path(summary["source_summary"])
        source_time_step_us = None
        if source_summary_path.exists():
            source_summary = json.loads(source_summary_path.read_text())
            source_time_step_us = source_summary.get("time_step_dt_us")
        return (
            {
                int(entry["neuron_index"]): {
                    "reference_spike_times_us": entry["python_spike_times_us"],
                    "waveform_path": entry["pwlf"],
                }
                for entry in summary["scaled_waveforms"]
            },
            {
                "time_step_dt_us": source_time_step_us,
            }
        )
    if "python_reference" in summary:
        return (
            {
                int(entry["neuron_index"]): {
                    "reference_spike_times_us": entry["spike_times_us"],
                    "waveform_path": next(
                        wave["pwlf"]
                        for wave in summary["waveforms"]
                        if int(wave["neuron_index"]) == int(entry["neuron_index"])
                    ),
                }
                for entry in summary["python_reference"]
            }
            ,
            {
                "time_step_dt_us": summary.get("time_step_dt_us"),
            },
        )
    raise ValueError(f"Unsupported reference summary format: {summary_path}")


def read_pwlf(path: Path) -> list[tuple[float, float]]:
    points = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        t_str, i_str = line.split()
        points.append((float(t_str), float(i_str)))
    return points


def waveform_from_points(points: list[tuple[float, float]]):
    times = np.array([p[0] for p in points], dtype=float)
    currents = np.array([p[1] for p in points], dtype=float)

    def _waveform(t_query: np.ndarray) -> np.ndarray:
        return np.interp(t_query, times, currents)

    return _waveform


def simulate_reference_from_pwlf(
    waveform_path: Path,
    duration_s: float,
    dt_s: float,
    vmem0_v: float | None = None,
    current_gain_scale: float = 1.0,
    refractory_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    base_params = load_circuit_params()
    params = base_params.with_updates(
        current_gain=base_params.current_gain * current_gain_scale,
        refractory_s=base_params.refractory_s * refractory_scale,
    )
    points = read_pwlf(waveform_path)
    return simulate_neuron(
        params=params,
        duration_s=duration_s,
        dt_s=dt_s,
        input_current=waveform_from_points(points),
        vmem0=params.v_reset if vmem0_v is None else vmem0_v,
    )


def plot_side_by_side(
    cadence_traces: list[NeuronTrace],
    python_traces: dict[int, dict[str, np.ndarray]],
    slot_to_neuron: dict[int, int],
    out_path: Path,
    window_ms: float | None = None,
) -> None:
    fig, axes = plt.subplots(len(cadence_traces), 4, figsize=(16, 2.6 * len(cadence_traces)), sharex=False)
    if len(cadence_traces) == 1:
        axes = np.array([axes])

    for row, trace in enumerate(cadence_traces):
        neuron_idx = slot_to_neuron[trace.neuron_slot]
        py = python_traces[neuron_idx]
        vmem_t_max_ms = max(float(trace.vmem_time_s[-1]), float(py["time_s"][-1])) * 1e3
        vout_t_max_ms = max(float(trace.vout_time_s[-1]), float(py["time_s"][-1])) * 1e3
        vmem_min = min(float(np.min(trace.vmem_v)), float(np.min(py["vmem_v"])))
        vmem_max = max(float(np.max(trace.vmem_v)), float(np.max(py["vmem_v"])))
        vout_min = min(float(np.min(trace.vout_v)), float(np.min(py["vout_v"])))
        vout_max = max(float(np.max(trace.vout_v)), float(np.max(py["vout_v"])))
        vmem_pad = max(0.02, 0.05 * (vmem_max - vmem_min if vmem_max > vmem_min else 1.0))
        vout_pad = max(0.02, 0.05 * (vout_max - vout_min if vout_max > vout_min else 1.0))

        axes[row, 0].plot(trace.vmem_time_s * 1e3, trace.vmem_v, color="tab:green")
        axes[row, 0].set_ylabel(f"n{neuron_idx}\nV (V)")
        axes[row, 0].grid(alpha=0.25)

        axes[row, 1].plot(py["time_s"] * 1e3, py["vmem_v"], color="black")
        axes[row, 1].grid(alpha=0.25)

        axes[row, 2].plot(trace.vout_time_s * 1e3, trace.vout_v, color="tab:red")
        axes[row, 2].grid(alpha=0.25)

        axes[row, 3].plot(py["time_s"] * 1e3, py["vout_v"], color="tab:blue")
        axes[row, 3].grid(alpha=0.25)

        if window_ms is not None:
            vmem_t_max_ms = min(vmem_t_max_ms, window_ms)
            vout_t_max_ms = min(vout_t_max_ms, window_ms)
        axes[row, 0].set_xlim(0.0, vmem_t_max_ms)
        axes[row, 1].set_xlim(0.0, vmem_t_max_ms)
        axes[row, 2].set_xlim(0.0, vout_t_max_ms)
        axes[row, 3].set_xlim(0.0, vout_t_max_ms)
        axes[row, 0].set_ylim(vmem_min - vmem_pad, vmem_max + vmem_pad)
        axes[row, 1].set_ylim(vmem_min - vmem_pad, vmem_max + vmem_pad)
        axes[row, 2].set_ylim(vout_min - vout_pad, vout_max + vout_pad)
        axes[row, 3].set_ylim(vout_min - vout_pad, vout_max + vout_pad)

    titles = ["Cadence Vmem", "Python Vmem", "Cadence Vout", "Python Vout"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title)
    for col in range(4):
        axes[-1, col].set_xlabel("Time (ms)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_aligned_overlay(
    cadence_traces: list[NeuronTrace],
    python_traces: dict[int, dict[str, np.ndarray]],
    slot_to_neuron: dict[int, int],
    out_path: Path,
    window_ms: float | None = None,
) -> None:
    fig, axes = plt.subplots(len(cadence_traces), 2, figsize=(12, 2.8 * len(cadence_traces)), sharex=False)
    if len(cadence_traces) == 1:
        axes = np.array([axes])

    for row, trace in enumerate(cadence_traces):
        neuron_idx = slot_to_neuron[trace.neuron_slot]
        py = python_traces[neuron_idx]
        cadence_spikes = detect_spike_times(trace.vout_time_s, trace.vout_v)
        python_spikes = py["spike_times_s"]
        shift_s = 0.0
        if len(cadence_spikes) > 0 and len(python_spikes) > 0:
            shift_s = cadence_spikes[0] - python_spikes[0]

        py_time_ms = (py["time_s"] + shift_s) * 1e3
        cadence_time_ms = trace.vmem_time_s * 1e3
        cadence_time_vout_ms = trace.vout_time_s * 1e3

        axes[row, 0].plot(cadence_time_ms, trace.vmem_v, color="tab:green", label="Cadence")
        axes[row, 0].plot(py_time_ms, py["vmem_v"], color="black", alpha=0.8, label="Python aligned")
        axes[row, 0].set_ylabel(f"n{neuron_idx}\nV (V)")
        axes[row, 0].grid(alpha=0.25)

        axes[row, 1].plot(cadence_time_vout_ms, trace.vout_v, color="tab:red", label="Cadence")
        axes[row, 1].plot(py_time_ms, py["vout_v"], color="tab:blue", alpha=0.8, label="Python aligned")
        axes[row, 1].grid(alpha=0.25)

        start_ms = 0.0
        stop_ms = min(
            max(float(cadence_time_ms[-1]), float(py_time_ms[-1])),
            float(cadence_time_ms[-1]),
        )
        if window_ms is not None:
            stop_ms = min(stop_ms, window_ms)
        vmem_min = min(float(np.min(trace.vmem_v)), float(np.min(py["vmem_v"])))
        vmem_max = max(float(np.max(trace.vmem_v)), float(np.max(py["vmem_v"])))
        vout_min = min(float(np.min(trace.vout_v)), float(np.min(py["vout_v"])))
        vout_max = max(float(np.max(trace.vout_v)), float(np.max(py["vout_v"])))
        vmem_pad = max(0.02, 0.05 * (vmem_max - vmem_min if vmem_max > vmem_min else 1.0))
        vout_pad = max(0.02, 0.05 * (vout_max - vout_min if vout_max > vout_min else 1.0))

        axes[row, 0].set_xlim(start_ms, stop_ms)
        axes[row, 1].set_xlim(start_ms, stop_ms)
        axes[row, 0].set_ylim(vmem_min - vmem_pad, vmem_max + vmem_pad)
        axes[row, 1].set_ylim(vout_min - vout_pad, vout_max + vout_pad)

    axes[0, 0].set_title("Aligned Vmem Overlay")
    axes[0, 1].set_title("Aligned Vout Overlay")
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].legend(loc="upper right")
    axes[-1, 0].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compare_spike_times(reference_us: np.ndarray, cadence_us: np.ndarray) -> dict[str, object]:
    paired = min(len(reference_us), len(cadence_us))
    if paired == 0:
        return {
            "paired_spikes": 0,
            "mean_abs_timing_error_us": None,
            "max_abs_timing_error_us": None,
        }
    errors = cadence_us[:paired] - reference_us[:paired]
    return {
        "paired_spikes": int(paired),
        "mean_abs_timing_error_us": float(np.mean(np.abs(errors))),
        "max_abs_timing_error_us": float(np.max(np.abs(errors))),
    }


def plot_overlay(traces: list[NeuronTrace], slot_to_neuron: dict[int, int], out_path: Path) -> None:
    fig, axes = plt.subplots(len(traces), 2, figsize=(11, 2.8 * len(traces)), sharex=False)
    if len(traces) == 1:
        axes = np.array([axes])

    for row, trace in enumerate(traces):
        neuron_idx = slot_to_neuron[trace.neuron_slot]
        axes[row, 0].plot(trace.vmem_time_s * 1e3, trace.vmem_v, color="tab:green")
        axes[row, 0].set_ylabel(f"n{neuron_idx}\nVmem (V)")
        axes[row, 0].grid(alpha=0.25)

        axes[row, 1].plot(trace.vout_time_s * 1e3, trace.vout_v, color="tab:red")
        axes[row, 1].set_ylabel(f"n{neuron_idx}\nVout (V)")
        axes[row, 1].grid(alpha=0.25)

    axes[-1, 0].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlabel("Time (ms)")
    axes[0, 0].set_title("Cadence membrane traces")
    axes[0, 1].set_title("Cadence output traces")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Section 5 Cadence neuron traces to Python reference spikes.")
    parser.add_argument("--cadence-csv", type=Path, default=DEFAULT_CADENCE_CSV)
    parser.add_argument("--reference-summary", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--match-cadence-duration",
        action="store_true",
        help="Replay the Python neuron out to the Cadence stop time using the reference PWL file and held final value.",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=None,
        help="Optional comparison/plot window in milliseconds from t=0.",
    )
    parser.add_argument(
        "--python-dt-ns",
        type=float,
        default=None,
        help="Override Python replay timestep in nanoseconds.",
    )
    parser.add_argument(
        "--current-gain-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the Section 2 fitted current_gain for Python replay.",
    )
    parser.add_argument(
        "--refractory-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the Section 2 fitted refractory_s for Python replay.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_traces = parse_cadence_csv(args.cadence_csv)
    stop_s = None if args.window_ms is None else args.window_ms * 1e-3
    traces = [clip_trace(trace, stop_s) for trace in raw_traces]
    reference, reference_meta = load_reference(args.reference_summary)
    ordered_neurons = list(reference.keys())
    slot_to_neuron = {
        trace.neuron_slot: ordered_neurons[trace.neuron_slot - 1]
        for trace in traces
        if trace.neuron_slot - 1 < len(ordered_neurons)
    }
    dt_s = (reference_meta.get("time_step_dt_us") or 5.0) * 1e-6
    if args.python_dt_ns is not None:
        dt_s = args.python_dt_ns * 1e-9

    per_neuron = []
    python_traces: dict[int, dict[str, np.ndarray]] = {}
    for trace in traces:
        neuron_idx = slot_to_neuron[trace.neuron_slot]
        waveform_path = Path(reference[neuron_idx]["waveform_path"])
        vmem0_v = float(trace.vmem_v[0]) if len(trace.vmem_v) > 0 else None
        if args.match_cadence_duration:
            py_trace = simulate_reference_from_pwlf(
                waveform_path=waveform_path,
                duration_s=float(trace.vout_time_s[-1]),
                dt_s=dt_s,
                vmem0_v=vmem0_v,
                current_gain_scale=args.current_gain_scale,
                refractory_scale=args.refractory_scale,
            )
            ref_spikes_us = py_trace["spike_times_s"] * 1e6
        else:
            py_trace = simulate_reference_from_pwlf(
                waveform_path=waveform_path,
                duration_s=float(trace.vout_time_s[-1]),
                dt_s=dt_s,
                vmem0_v=vmem0_v,
                current_gain_scale=args.current_gain_scale,
                refractory_scale=args.refractory_scale,
            )
            ref_spikes_us = np.array(reference[neuron_idx]["reference_spike_times_us"], dtype=float)
        python_traces[neuron_idx] = py_trace
        cadence_spikes_us = detect_spike_times(trace.vout_time_s, trace.vout_v) * 1e6
        per_neuron.append(
            {
                "slot": trace.neuron_slot,
                "neuron_index": neuron_idx,
                "waveform_path": str(waveform_path),
                "reference_spike_times_us": ref_spikes_us.tolist(),
                "cadence_spike_times_us": cadence_spikes_us.tolist(),
                "cadence_spike_count": int(len(cadence_spikes_us)),
                "reference_spike_count": int(len(ref_spikes_us)),
                "cadence_vmem_max_v": float(np.max(trace.vmem_v)),
                "cadence_vout_max_v": float(np.max(trace.vout_v)),
                "python_dt_ns": dt_s * 1e9,
                "python_initial_vmem_v": vmem0_v,
                "python_current_gain_scale": args.current_gain_scale,
                "python_refractory_scale": args.refractory_scale,
                **compare_spike_times(ref_spikes_us, cadence_spikes_us),
                "mean_abs_timing_error_ns": None
                if compare_spike_times(ref_spikes_us, cadence_spikes_us)["mean_abs_timing_error_us"] is None
                else compare_spike_times(ref_spikes_us, cadence_spikes_us)["mean_abs_timing_error_us"] * 1e3,
            }
        )

    summary = {
        "cadence_csv": str(args.cadence_csv),
        "reference_summary": str(args.reference_summary),
        "window_ms": args.window_ms,
        "python_current_gain_scale": args.current_gain_scale,
        "python_refractory_scale": args.refractory_scale,
        "notes": [
            "Cadence traces were parsed from a single CSV containing interleaved vmem/vout columns for four neurons.",
            "Spike times are extracted from rising threshold crossings on vout using a 1.6 V threshold.",
            "Neuron-slot to Python-neuron mapping assumes the same order as the reference summary.",
            "When --match-cadence-duration is used, the Python replay extends to the Cadence stop time and holds the last PWL current value after the final waveform point.",
            "The Python replay uses the first Cadence vmem sample as its initial membrane voltage to better match startup conditions.",
            "Optional Python replay scaling lets Section 5 timing be tuned without overwriting the original Section 2 fit.",
        ],
        "match_cadence_duration": args.match_cadence_duration,
        "per_neuron": per_neuron,
    }

    summary_path = args.out_dir / "section5_cadence_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    plot_overlay(traces, slot_to_neuron, args.out_dir / "section5_cadence_overlay.png")
    plot_side_by_side(
        cadence_traces=traces,
        python_traces=python_traces,
        slot_to_neuron=slot_to_neuron,
        out_path=args.out_dir / "section5_cadence_python_side_by_side.png",
        window_ms=args.window_ms,
    )
    plot_aligned_overlay(
        cadence_traces=traces,
        python_traces=python_traces,
        slot_to_neuron=slot_to_neuron,
        out_path=args.out_dir / "section5_cadence_python_aligned_overlay.png",
        window_ms=args.window_ms,
    )

    print(f"Comparison summary: {summary_path}")
    for entry in per_neuron:
        print(
            f"slot {entry['slot']} / neuron {entry['neuron_index']}: "
            f"ref_spikes={entry['reference_spike_count']} cadence_spikes={entry['cadence_spike_count']} "
            f"mean_abs_timing_error_us={entry['mean_abs_timing_error_us']}"
        )


if __name__ == "__main__":
    main()
