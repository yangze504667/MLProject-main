"""Compute a lower bound on unavoidable cable interference from fixed endpoints."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="Directory containing the supplied data folder")
    args = parser.parse_args()
    data = json.loads((args.project / "data/bundles.json").read_text(encoding="utf-8"))
    threshold = 2 * data["bundle_radius"]
    endpoints = np.array([np.array(bundle["points"])[[0, -1]] for bundle in data["bundles"]])

    lower_bound = 0.0
    for i, ends in enumerate(endpoints):
        other_ends = np.delete(endpoints, i, axis=0).reshape(-1, 3)
        distances = np.linalg.norm(ends[:, None, :] - other_ends, axis=2)
        # Each end contributes an interval [0, radius sum - closest distance).
        guaranteed = np.maximum(0, threshold - distances.min(axis=1))
        chord_length = np.linalg.norm(ends[1] - ends[0])
        # The curve is at least as long as its chord. Cap the sum to avoid
        # double counting if the two guaranteed intervals could overlap.
        lower_bound += min(chord_length, guaranteed.sum())

    baseline = json.loads((Path(__file__).parent / "baseline_check.json").read_text(encoding="utf-8"))
    upper_bound = 100 * (1 - lower_bound / baseline["total_before_length_mm"])
    print(f"Unavoidable interference: at least {lower_bound:.6f} mm")
    print(f"Scene improvement: at most {upper_bound:.6f}%")


if __name__ == "__main__":
    main()
