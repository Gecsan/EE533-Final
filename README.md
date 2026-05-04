# EE533-Final

EE 533 Final Project code and generated artifacts for the circuit-informed SNN workflow.

## Project Layout

- `circuit_if_neuron.py`
  Behavioral circuit-informed neuron model and fitting utilities.
- `section2_circuit_workflow.py`
  Section 2 workflow: fit the Python neuron to Cadence data and generate validation plots.
- `section3_validation_workflow.py`
  Section 3 single-neuron validation against exported Cadence traces.
- `section3_mnist_workflow.py`
  Section 6-style default LIF vs circuit-informed SNN sweeps.
- `section4_training_workflow.py`
  Section 4 MNIST training with the circuit-informed neuron and selectable hidden-layer size.
- `section5_waveform_export.py`
  Export hidden-neuron current waveforms for Cadence.
- `section5_rescale_waveforms.py`
  Generate lower-current waveform variants for Cadence experiments.
- `section5_compare_cadence.py`
  Compare Cadence-exported neuron traces against the Python neuron replay.

## References and How They Are Used

The SpikingJelly references are used for the SNN framework side of the project:

1. Fang, Wei, et al. "SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence." *Science Advances* 9, no. 40 (2023): eadi1480.
2. Fang, W. *SpikingJelly: An open-source machine learning framework for spiking neural networks*. GitHub. https://github.com/fangwei123456/spikingjelly

In this repo, those references support:

- the default spiking-neuron baseline used for comparison
- the idea of replacing the default neuron with a custom `CircuitIFNode`
- the MNIST SNN workflow and evaluation structure

They are **not** the source of the analog circuit parameters. The circuit-informed neuron model is derived from Cadence-exported data and fitted in Python.

## Section 2

Fit the circuit-informed neuron and generate the core plots:

```bash
python section2_circuit_workflow.py
```

Main outputs:

- `outputs/section2/section2_summary.json`
- `outputs/section2/section2_fi_curve.png`
- `outputs/section2/section2_constant_current_overlay.png`
- `outputs/section2/section2_measured_pulse_trace.png`

## Section 3

Validate the single-neuron Python model against Cadence exports:

```bash
python section3_validation_workflow.py
```

Main outputs:

- `outputs/section3_validation/section3_validation_summary.json`
- `outputs/section3_validation/section3_constant_current_overlay.png`
- `outputs/section3_validation/section3_pulsed_current_overlay.png`

## Section 4

Train the circuit-informed SNN on MNIST using one hidden layer:

```bash
python section4_training_workflow.py --epochs 3 --train-subset 10000 --test-subset 2000 --hidden-sizes 512 256 128 64 32
```

Main outputs:

- `outputs/section4/section4_summary.json`
- `outputs/section4/section4_hidden_size_accuracy.png`

## Section 5

### Export Cadence input waveforms

```bash
python section5_waveform_export.py
```

Optional low-current rescale:

```bash
python section5_rescale_waveforms.py
```

### Compare Cadence traces to Python

Current tuned comparison command:

```bash
python section5_compare_cadence.py \
  --cadence-csv ~/Downloads/533FinalPart5Cadence.csv \
  --reference-summary outputs/section5_scaled_4na/section5_scaled_4na_summary.json \
  --match-cadence-duration \
  --window-ms 20 \
  --python-dt-ns 100 \
  --current-gain-scale 0.92 \
  --refractory-scale 0.8
```

Main outputs:

- `outputs/section5_compare/section5_cadence_comparison_summary.json`
- `outputs/section5_compare/section5_cadence_python_side_by_side.png`
- `outputs/section5_compare/section5_cadence_python_aligned_overlay.png`

## Notes

- The tuned Section 5 comparison scales the Python replay only for the Section 5 timing study. It does not overwrite the original Section 2 fitted parameters on disk.
- If Cadence runs much longer than the last PWL point, Spectre will hold the final current value unless the waveform is explicitly driven back to zero.
