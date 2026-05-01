from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is optional for pure Python use
    pd = None

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch is optional for pure Python use
    torch = None
    nn = None


@dataclass(frozen=True)
class CadenceNeuronParameters:
    """Behavioral neuron parameters derived from the Cadence circuit."""

    vdd: float = 3.2
    c_mem: float = 24.5e-12
    c_fb: float = 3.757e-12
    v_reset: float = 1.12
    v_threshold: float = 1.83
    refractory_s: float = 2e-6
    pulse_width_s: float = 2e-6
    leak_conductance_s: float = 0.0
    feedback_cap_scale: float = 1.0
    current_gain: float = 1.0

    @property
    def total_capacitance(self) -> float:
        return self.c_mem + self.feedback_cap_scale * self.c_fb

    @property
    def delta_v(self) -> float:
        return self.v_threshold - self.v_reset

    def with_updates(self, **kwargs) -> "CadenceNeuronParameters":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class CadenceTraceMeasurement:
    current_a: float
    time_s: np.ndarray
    vmem_v: np.ndarray
    vout_v: np.ndarray
    spike_times_s: np.ndarray
    firing_rate_hz: float
    pulse_width_s: float
    threshold_v: float
    reset_v: float


@dataclass(frozen=True)
class CadenceFitResult:
    params: CadenceNeuronParameters
    currents_a: np.ndarray
    measured_rates_hz: np.ndarray
    predicted_rates_hz: np.ndarray
    rmse_hz: float
    mape_percent: float
    mean_threshold_v: float
    mean_reset_v: float
    mean_pulse_width_s: float


def calibrate_current_gain_from_point(
    params: CadenceNeuronParameters,
    input_current_a: float,
    measured_rate_hz: float,
) -> float:
    """Fit the effective charging gain from a measured constant-current point."""
    if input_current_a <= 0:
        raise ValueError("input_current_a must be positive")
    if measured_rate_hz <= 0:
        raise ValueError("measured_rate_hz must be positive")

    period_s = 1.0 / measured_rate_hz
    integration_time_s = period_s - params.refractory_s
    if integration_time_s <= 0:
        raise ValueError("Measured rate is inconsistent with the refractory time")

    numerator = params.total_capacitance * params.delta_v
    denominator = input_current_a * integration_time_s
    return numerator / denominator


def analytical_rate_hz(
    params: CadenceNeuronParameters,
    input_current_a: float,
) -> float:
    if input_current_a <= 0:
        return 0.0

    effective_current = params.current_gain * input_current_a
    integration_time_s = params.total_capacitance * params.delta_v / effective_current
    period_s = integration_time_s + params.refractory_s
    return 1.0 / period_s


def fi_curve(
    params: CadenceNeuronParameters,
    currents_a: Iterable[float],
) -> np.ndarray:
    currents = np.asarray(list(currents_a), dtype=float)
    return np.array([analytical_rate_hz(params, i) for i in currents], dtype=float)


def constant_current_waveform(current_a: float) -> Callable[[np.ndarray], np.ndarray]:
    def _waveform(t: np.ndarray) -> np.ndarray:
        return np.full_like(t, current_a, dtype=float)

    return _waveform


def pulsed_current_waveform(
    baseline_a: float,
    pulse_a: float,
    start_s: float,
    width_s: float,
    period_s: float,
) -> Callable[[np.ndarray], np.ndarray]:
    def _waveform(t: np.ndarray) -> np.ndarray:
        phase = np.mod(np.maximum(t - start_s, 0.0), period_s)
        active = (t >= start_s) & (phase < width_s)
        return np.where(active, pulse_a, baseline_a)

    return _waveform


def simulate_neuron(
    params: CadenceNeuronParameters,
    duration_s: float,
    dt_s: float,
    input_current: float | Callable[[np.ndarray], np.ndarray],
    vmem0: float | None = None,
) -> dict[str, np.ndarray]:
    """Simulate a behavioral integrate-and-fire neuron."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")

    time = np.arange(0.0, duration_s + dt_s, dt_s)
    if callable(input_current):
        current = np.asarray(input_current(time), dtype=float)
    else:
        current = np.full_like(time, input_current, dtype=float)

    vmem = np.zeros_like(time)
    vout = np.zeros_like(time)
    vmem[0] = params.v_reset if vmem0 is None else vmem0

    refractory_until = -np.inf
    pulse_active_until = -np.inf
    spike_times: list[float] = []

    for idx in range(1, len(time)):
        t_prev = time[idx - 1]
        t_now = time[idx]
        previous_v = vmem[idx - 1]

        if t_prev < refractory_until:
            next_v = params.v_reset
        else:
            dv_charge = params.current_gain * current[idx - 1] * dt_s / params.total_capacitance
            dv_leak = params.leak_conductance_s * (previous_v - params.v_reset) * dt_s
            next_v = previous_v + dv_charge - dv_leak

        if next_v >= params.v_threshold and t_prev >= refractory_until:
            spike_times.append(t_now)
            next_v = params.v_reset
            refractory_until = t_now + params.refractory_s
            pulse_active_until = t_now + params.pulse_width_s

        vmem[idx] = next_v
        vout[idx] = params.vdd if t_now < pulse_active_until else 0.0

    return {
        "time_s": time,
        "input_current_a": current,
        "vmem_v": vmem,
        "vout_v": vout,
        "spike_times_s": np.asarray(spike_times, dtype=float),
    }


def measure_firing_rate(spike_times_s: np.ndarray, duration_s: float) -> float:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    spike_times_s = np.asarray(spike_times_s, dtype=float)
    if len(spike_times_s) >= 2:
        interval_s = spike_times_s[-1] - spike_times_s[0]
        if interval_s > 0:
            return float((len(spike_times_s) - 1) / interval_s)
    return float(len(spike_times_s) / duration_s)


def nearest_neighbor_timing_error_s(
    reference_spikes_s: np.ndarray,
    candidate_spikes_s: np.ndarray,
) -> float:
    if len(reference_spikes_s) == 0 or len(candidate_spikes_s) == 0:
        return float("inf")

    errors = []
    for spike in np.asarray(reference_spikes_s, dtype=float):
        errors.append(np.min(np.abs(candidate_spikes_s - spike)))
    return float(np.mean(errors))


def _to_float_array(values: Sequence[float]) -> np.ndarray:
    if pd is not None and hasattr(values, "to_numpy"):
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.asarray(values, dtype=float)


def _clean_trace(time_s: Sequence[float], values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    time_s = _to_float_array(time_s)
    values = _to_float_array(values)
    mask = np.isfinite(time_s) & np.isfinite(values)
    return time_s[mask], values[mask]


def extract_spike_edges_from_vout(
    time_s: Sequence[float],
    vout_v: Sequence[float],
    edge_threshold_v: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    time_s, vout_v = _clean_trace(time_s, vout_v)
    if time_s.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    if edge_threshold_v is None:
        v_min = float(np.min(vout_v))
        v_max = float(np.max(vout_v))
        edge_threshold_v = v_min + 0.5 * (v_max - v_min)

    high = vout_v >= edge_threshold_v
    rising_idx = np.flatnonzero((~high[:-1]) & high[1:]) + 1
    falling_idx = np.flatnonzero(high[:-1] & (~high[1:])) + 1

    if falling_idx.size > 0 and rising_idx.size > 0 and falling_idx[0] < rising_idx[0]:
        falling_idx = falling_idx[1:]

    pulse_count = min(rising_idx.size, falling_idx.size)
    rising_idx = rising_idx[:pulse_count]
    falling_idx = falling_idx[:pulse_count]
    return time_s[rising_idx], time_s[falling_idx]


def extract_spike_times_from_vout(
    time_s: Sequence[float],
    vout_v: Sequence[float],
    edge_threshold_v: float | None = None,
) -> np.ndarray:
    rising_edges_s, _ = extract_spike_edges_from_vout(time_s, vout_v, edge_threshold_v)
    return rising_edges_s


def estimate_threshold_and_reset_from_vmem(
    time_s: Sequence[float],
    vmem_v: Sequence[float],
    spike_times_s: Sequence[float],
    reset_times_s: Sequence[float] | None = None,
    window_samples: int = 8,
) -> tuple[float, float]:
    time_s, vmem_v = _clean_trace(time_s, vmem_v)
    spike_times_s = np.asarray(spike_times_s, dtype=float)
    if time_s.size == 0 or spike_times_s.size == 0:
        return float("nan"), float("nan")
    if reset_times_s is None:
        reset_times_s = spike_times_s
    reset_times_s = np.asarray(reset_times_s, dtype=float)

    peak_values = []
    reset_values = []
    for spike_time_s in spike_times_s:
        idx = int(np.searchsorted(time_s, spike_time_s))
        peak_lo = max(0, idx - window_samples)
        peak_hi = min(time_s.size, idx + 1)
        if peak_lo < peak_hi:
            peak_values.append(float(np.max(vmem_v[peak_lo:peak_hi])))

    for reset_time_s in reset_times_s:
        idx = int(np.searchsorted(time_s, reset_time_s))
        reset_lo = max(0, idx)
        reset_hi = min(time_s.size, idx + window_samples)
        if reset_lo < reset_hi:
            reset_values.append(float(np.min(vmem_v[reset_lo:reset_hi])))

    threshold_v = float(np.nanmedian(peak_values)) if peak_values else float("nan")
    reset_v = float(np.nanmedian(reset_values)) if reset_values else float("nan")
    return threshold_v, reset_v


def measure_pulse_width_s(
    time_s: Sequence[float],
    vout_v: Sequence[float],
    edge_threshold_v: float | None = None,
) -> float:
    rising_edges_s, falling_edges_s = extract_spike_edges_from_vout(time_s, vout_v, edge_threshold_v)
    if rising_edges_s.size == 0:
        return float("nan")
    return float(np.mean(falling_edges_s - rising_edges_s))


def load_cadence_trace_csv(
    csv_path: str | Path,
    current_a: float = float("nan"),
    vmem_time_column: str = "/vmem X",
    vmem_value_column: str = "/vmem Y",
    vout_time_column: str = "/vout X",
    vout_value_column: str = "/vout Y",
) -> CadenceTraceMeasurement:
    if pd is None:
        raise ImportError("pandas is required to load Cadence CSV exports")

    df = pd.read_csv(csv_path, low_memory=False)
    time_s, vmem_v = _clean_trace(df[vmem_time_column], df[vmem_value_column])
    vout_time_s, vout_v = _clean_trace(df[vout_time_column], df[vout_value_column])
    return _build_trace_measurement(current_a, time_s, vmem_v, vout_time_s, vout_v)


def load_cadence_fi_sweep(csv_path: str | Path) -> list[CadenceTraceMeasurement]:
    if pd is None:
        raise ImportError("pandas is required to load Cadence CSV exports")

    df = pd.read_csv(csv_path, low_memory=False)
    pattern = re.compile(r"^/vmem \(iin=([^)]+)\) Y$")
    measurements = []

    for column in df.columns:
        match = pattern.match(column)
        if match is None:
            continue

        current_text = match.group(1)
        current_a = float(current_text)
        vmem_time_column = column[:-1] + "X"
        vout_time_column = f"/vout (iin={current_text}) X"
        vout_value_column = f"/vout (iin={current_text}) Y"
        time_s, vmem_v = _clean_trace(df[vmem_time_column], df[column])
        vout_time_s, vout_v = _clean_trace(df[vout_time_column], df[vout_value_column])
        measurements.append(_build_trace_measurement(current_a, time_s, vmem_v, vout_time_s, vout_v))

    measurements.sort(key=lambda item: item.current_a)
    return measurements


def _build_trace_measurement(
    current_a: float,
    time_s: np.ndarray,
    vmem_v: np.ndarray,
    vout_time_s: np.ndarray,
    vout_v: np.ndarray,
) -> CadenceTraceMeasurement:
    spike_times_s, reset_times_s = extract_spike_edges_from_vout(vout_time_s, vout_v)
    pulse_width_s = float(np.mean(reset_times_s - spike_times_s)) if spike_times_s.size > 0 else float("nan")
    threshold_v, reset_v = estimate_threshold_and_reset_from_vmem(
        time_s,
        vmem_v,
        spike_times_s,
        reset_times_s=reset_times_s,
    )
    duration_s = float(vout_time_s[-1] - vout_time_s[0]) if vout_time_s.size > 1 else 0.0

    return CadenceTraceMeasurement(
        current_a=current_a,
        time_s=time_s,
        vmem_v=vmem_v,
        vout_v=vout_v,
        spike_times_s=spike_times_s,
        firing_rate_hz=measure_firing_rate(spike_times_s, duration_s) if duration_s > 0 else 0.0,
        pulse_width_s=pulse_width_s,
        threshold_v=threshold_v,
        reset_v=reset_v,
    )


def fit_cadence_parameters_from_sweep(
    base_params: CadenceNeuronParameters,
    measurements: Sequence[CadenceTraceMeasurement],
) -> CadenceFitResult:
    if len(measurements) == 0:
        raise ValueError("measurements must not be empty")

    currents_a = np.array([m.current_a for m in measurements], dtype=float)
    measured_rates_hz = np.array([m.firing_rate_hz for m in measurements], dtype=float)
    valid = (currents_a > 0) & (measured_rates_hz > 0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two positive-current firing-rate points are required")

    periods_s = 1.0 / measured_rates_hz[valid]
    inverse_current = 1.0 / currents_a[valid]
    design = np.column_stack([inverse_current, np.ones_like(inverse_current)])
    charge_term_s_a, refractory_s = np.linalg.lstsq(design, periods_s, rcond=None)[0]
    charge_term_s_a = max(float(charge_term_s_a), np.finfo(float).tiny)
    refractory_s = max(float(refractory_s), 0.0)

    threshold_values = np.array([m.threshold_v for m in measurements], dtype=float)
    reset_values = np.array([m.reset_v for m in measurements], dtype=float)
    pulse_width_values = np.array([m.pulse_width_s for m in measurements], dtype=float)

    mean_threshold_v = float(np.nanmedian(threshold_values))
    mean_reset_v = float(np.nanmedian(reset_values))
    mean_pulse_width_s = float(np.nanmedian(pulse_width_values))
    delta_v = mean_threshold_v - mean_reset_v
    if not np.isfinite(delta_v) or delta_v <= 0:
        raise ValueError("Estimated threshold and reset voltages are invalid")

    current_gain = base_params.total_capacitance * delta_v / charge_term_s_a
    params = base_params.with_updates(
        v_reset=mean_reset_v,
        v_threshold=mean_threshold_v,
        refractory_s=refractory_s,
        pulse_width_s=mean_pulse_width_s,
        current_gain=current_gain,
    )

    predicted_rates_hz = fi_curve(params, currents_a)
    rmse_hz = float(np.sqrt(np.mean((predicted_rates_hz - measured_rates_hz) ** 2)))
    mape_percent = float(
        np.mean(np.abs((predicted_rates_hz[valid] - measured_rates_hz[valid]) / measured_rates_hz[valid])) * 100.0
    )

    return CadenceFitResult(
        params=params,
        currents_a=currents_a,
        measured_rates_hz=measured_rates_hz,
        predicted_rates_hz=predicted_rates_hz,
        rmse_hz=rmse_hz,
        mape_percent=mape_percent,
        mean_threshold_v=mean_threshold_v,
        mean_reset_v=mean_reset_v,
        mean_pulse_width_s=mean_pulse_width_s,
    )


if torch is not None:

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


    class CircuitIFNode(nn.Module):
        """Torch module for time-stepped SNN experiments."""

        def __init__(
            self,
            params: CadenceNeuronParameters,
            dt_s: float,
            step_mode: str = "s",
        ):
            super().__init__()
            if dt_s <= 0:
                raise ValueError("dt_s must be positive")
            if step_mode not in {"s", "m"}:
                raise ValueError('step_mode must be "s" or "m"')

            self.params = params
            self.dt_s = dt_s
            self.step_mode = step_mode
            self.register_buffer("_vmem", torch.tensor(float(params.v_reset)))
            self.register_buffer("_refrac_steps_left", torch.tensor(0.0))

        def reset(self) -> None:
            self._vmem = torch.tensor(float(self.params.v_reset), device=self._vmem.device, dtype=self._vmem.dtype)
            self._refrac_steps_left = torch.zeros_like(self._refrac_steps_left)

        def reset_state(self) -> None:
            self.reset()

        def _ensure_state_shape(self, input_current_a: torch.Tensor) -> None:
            if (
                self._vmem.ndim == 0
                or self._vmem.shape != input_current_a.shape
                or self._vmem.device != input_current_a.device
                or self._vmem.dtype != input_current_a.dtype
            ):
                self._vmem = torch.full_like(input_current_a, self.params.v_reset)
                self._refrac_steps_left = torch.zeros_like(input_current_a)

        def single_step_forward(self, input_current_a: torch.Tensor) -> torch.Tensor:
            self._ensure_state_shape(input_current_a)

            active = self._refrac_steps_left <= 0
            dv = (
                self.params.current_gain
                * input_current_a
                * self.dt_s
                / self.params.total_capacitance
            )
            leak = self.params.leak_conductance_s * (self._vmem - self.params.v_reset) * self.dt_s
            updated_v = torch.where(
                active,
                self._vmem + dv - leak,
                torch.full_like(self._vmem, self.params.v_reset),
            )

            spikes = _SurrogateSpike.apply(updated_v - self.params.v_threshold)
            self._vmem = torch.where(spikes > 0, torch.full_like(updated_v, self.params.v_reset), updated_v)

            refrac_steps = max(1, int(round(self.params.refractory_s / self.dt_s)))
            reset_steps = torch.full_like(self._refrac_steps_left, float(refrac_steps))
            self._refrac_steps_left = torch.where(
                spikes > 0,
                reset_steps,
                torch.clamp(self._refrac_steps_left - 1.0, min=0.0),
            )
            return spikes

        def multi_step_forward(self, input_current_seq_a: torch.Tensor) -> torch.Tensor:
            spikes = []
            for step_input in input_current_seq_a:
                spikes.append(self.single_step_forward(step_input))
            return torch.stack(spikes, dim=0)

        def forward(self, input_current_a: torch.Tensor) -> torch.Tensor:
            if self.step_mode == "m":
                return self.multi_step_forward(input_current_a)
            return self.single_step_forward(input_current_a)

        @property
        def v(self) -> torch.Tensor:
            return self._vmem
