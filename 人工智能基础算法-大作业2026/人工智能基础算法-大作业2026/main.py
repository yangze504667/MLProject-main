"""Read the supplied clustered bundles and call the empty student optimizer."""

import argparse
import json
from pathlib import Path

import numpy as np

from interference import InterferenceIndex
from optimize import optimize_bundle, optimize_scene, validate_candidate


ROOT = Path(__file__).resolve().parent


def load_data(root=ROOT):
    directory = Path(root) / "data"
    data = json.loads((directory / "bundles.json").read_text(encoding="utf-8"))
    scene = json.loads((directory / "scene.json").read_text(encoding="utf-8"))
    occupancy = np.load(directory / "occupancy.npy", mmap_mode="r", allow_pickle=False)
    radius = float(data["bundle_radius"])
    if data["units"] != "Millimeters" or scene["axis_order"] != "xyz":
        raise ValueError("expected millimeter coordinates and xyz voxel axes")
    if occupancy.shape != tuple(scene["dimensions"]) or occupancy.dtype != np.uint8:
        raise ValueError("occupancy does not match scene metadata")
    if not np.isfinite(radius) or radius <= 0 or not np.isclose(data["bundle_width"], 2 * radius, rtol=0, atol=1e-9):
        raise ValueError("invalid shared width or radius")
    bundles = [{"id": int(item["id"]), "radius": radius, "points": np.asarray(item["points"], dtype=float)}
               for item in data["bundles"]]
    if len(bundles) != data["bundle_count"] or len({item["id"] for item in bundles}) != len(bundles):
        raise ValueError("bundle count or ids are inconsistent")
    return bundles, occupancy, scene


def main():
    parser = argparse.ArgumentParser(description="Run individual or joint cable-cable interference optimization")
    parser.add_argument("--bundle", type=int, default=31)
    parser.add_argument("--joint", action="store_true", help="call optimize_scene for all bundles")
    parser.add_argument("--output", type=Path, default=Path("solution.json"))
    parser.add_argument("--report", type=Path, default=Path("evaluation.json"))
    args = parser.parse_args()
    bundles, occupancy, scene = load_data()
    bundle = next((item for item in bundles if item["id"] == args.bundle), None)
    if bundle is None and not args.joint:
        parser.error(f"unknown bundle id {args.bundle}")
    index = InterferenceIndex(bundles, occupancy, scene)
    print(f"Loaded {len(bundles)} clustered bundles; shared radius {bundles[0]['radius']:.6f} mm")
    if not args.joint:
        print(f"Bundle {args.bundle}: original cable-cable collided length {index.cable_collision_length(args.bundle):.6f} mm")
    try:
        if args.joint:
            candidates = optimize_scene([{**item, "points": item["points"].copy()} for item in bundles], index)
        else:
            candidates = {args.bundle: optimize_bundle(args.bundle, bundle["points"].copy(), bundle["radius"], index)}
    except NotImplementedError as error:
        print(f"Empty optimization interface: {error}")
        return
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("optimizer must return at least one candidate path")
    originals = {item["id"]: item["points"] for item in bundles}
    paths = {}
    for name, points in candidates.items():
        if not isinstance(name, int) or isinstance(name, bool) or name not in originals:
            raise ValueError(f"unknown bundle id: {name}")
        paths[name] = validate_candidate(originals[name], points, scene)
    from evaluate import evaluate_paths
    report = evaluate_paths(bundles, occupancy, scene, paths)
    result = {"format": "BundleAvoidanceSubmissionSet", "units": "Millimeters",
              "bundles": [{"id": name, "points": points.tolist()} for name, points in paths.items()]}
    if not args.joint:
        result = {"format": "BundleAvoidanceSubmission", "units": "Millimeters", "bundle_id": args.bundle,
                  "points": paths[args.bundle].tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    percent = "N/A" if report["total_improvement_pct"] is None else f"{report['total_improvement_pct']:.2f}%"
    print(f"Cable-cable collided length: {report['total_before_length_mm']:.6f} -> {report['total_after_length_mm']:.6f} mm; improvement {percent}")
    print(f"Chassis auxiliary length: {report['chassis_before_length_mm']:.6f} -> {report['chassis_after_length_mm']:.6f} mm")
    print(f"Saved paths to {args.output}; evaluation to {args.report}")


if __name__ == "__main__":
    main()
