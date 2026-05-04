from __future__ import annotations

import json
from pathlib import Path

from section7_simplification_workflow import save_scatter


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "section7"


def load_results(path: Path) -> tuple[list[dict[str, float]], float]:
    summary = json.loads(path.read_text())
    return summary["candidate_results"], float(summary["config"]["target_accuracy"])


def main() -> None:
    pass1_results, target_accuracy = load_results(ROOT / "outputs" / "section7_pass1" / "section7_summary.json")
    pass2_results, _ = load_results(ROOT / "outputs" / "section7_pass2" / "section7_summary.json")
    resolution_results = [
        {"image_size": 14, "hidden_size": 64, "time_steps": 20, "quant_bits": 4, "test_accuracy": 0.6850},
        {"image_size": 14, "hidden_size": 64, "time_steps": 20, "quant_bits": 8, "test_accuracy": 0.6940},
        {"image_size": 14, "hidden_size": 128, "time_steps": 20, "quant_bits": 4, "test_accuracy": 0.7330},
        {"image_size": 14, "hidden_size": 128, "time_steps": 20, "quant_bits": 8, "test_accuracy": 0.7390},
        {"image_size": 14, "hidden_size": 64, "time_steps": 10, "quant_bits": 4, "test_accuracy": 0.6660},
        {"image_size": 14, "hidden_size": 64, "time_steps": 10, "quant_bits": 8, "test_accuracy": 0.6760},
        {"image_size": 14, "hidden_size": 128, "time_steps": 10, "quant_bits": 4, "test_accuracy": 0.6860},
        {"image_size": 14, "hidden_size": 128, "time_steps": 10, "quant_bits": 8, "test_accuracy": 0.6695},
        {"image_size": 7, "hidden_size": 64, "time_steps": 20, "quant_bits": 4, "test_accuracy": 0.4445},
        {"image_size": 7, "hidden_size": 64, "time_steps": 20, "quant_bits": 8, "test_accuracy": 0.5060},
        {"image_size": 7, "hidden_size": 128, "time_steps": 20, "quant_bits": 4, "test_accuracy": 0.5535},
        {"image_size": 7, "hidden_size": 128, "time_steps": 20, "quant_bits": 8, "test_accuracy": 0.5240},
    ]

    save_scatter(
        pass1_results,
        target_accuracy,
        title="Section 7: Hidden size and quantization simplification",
        output_path=OUTPUT_DIR / "section7_hidden_quantization_candidates.png",
    )
    save_scatter(
        pass2_results,
        target_accuracy,
        title="Section 7: Time-step simplification",
        output_path=OUTPUT_DIR / "section7_time_step_candidates.png",
    )
    save_scatter(
        resolution_results,
        target_accuracy,
        title="Section 7: Resolution simplification",
        output_path=OUTPUT_DIR / "section7_resolution_candidates.png",
    )

    print("Section 7 plots refreshed.")


if __name__ == "__main__":
    main()
