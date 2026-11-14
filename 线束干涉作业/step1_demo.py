"""Manually move one bundle and let the supplied evaluator score the scene."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="Directory containing the supplied evaluate.py")
    parser.add_argument("--dy", type=float, default=-5, help="Y displacement in mm at the middle")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    project = args.project.resolve()
    if not (project / "evaluate.py").is_file():
        parser.error("project must be the directory containing evaluate.py")

    sys.path.insert(0, str(project))
    from evaluate import evaluate_paths, load_submission
    from main import load_data
    from optimize import validate_candidate

    bundles, occupancy, scene = load_data(project)
    bundle = next(item for item in bundles if item["id"] == 31)
    original = bundle["points"]

    # Scale the movement by point index: zero at both ends and one in the middle.
    t = np.linspace(0, 1, len(original))
    weights = 4 * t * (1 - t)
    candidate = original.copy()
    candidate[:, 1] += args.dy * weights
    validate_candidate(original, candidate, scene)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    solution_path = args.output_dir / f"step1_solution_y{args.dy:+g}.json"
    submission = {
        "format": "BundleAvoidanceSubmission",
        "units": "Millimeters",
        "bundle_id": bundle["id"],
        "points": candidate.tolist(),
    }
    solution_path.write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")

    # Read the saved file through the same parser used by evaluate.py.
    paths = load_submission(solution_path, bundles, scene)
    report = evaluate_paths(bundles, occupancy, scene, paths)
    report_path = args.output_dir / f"step1_evaluation_y{args.dy:+g}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for row in report["bundles"]:
        before, after = row["before_length_mm"], row["after_length_mm"]
        if abs(after - before) > 1e-6:
            print(f"Bundle {row['id']}: {before:.4f} -> {after:.4f} mm")
    print(f"All bundles: {report['total_before_length_mm']:.4f} -> "
          f"{report['total_after_length_mm']:.4f} mm ({report['total_improvement_pct']:+.4f}%)")
    print(f"Chassis: {report['chassis_before_length_mm']:.4f} -> "
          f"{report['chassis_after_length_mm']:.4f} mm")
    print(f"Solution: {solution_path}\nEvaluation: {report_path}")


if __name__ == "__main__":
    main()
