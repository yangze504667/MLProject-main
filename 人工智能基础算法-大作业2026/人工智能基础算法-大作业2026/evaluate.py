"""Continuous interference evaluation; no optimizer or reference solution is included."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from interference import InterferenceIndex
from main import load_data
from optimize import validate_candidate


def load_submission(path, bundles, scene):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("units") != "Millimeters" or any(key in data for key in ("radius", "bundle_radius", "bundle_width")):
        raise ValueError("submission must use millimeters and must not redefine bundle width")
    if data.get("format") == "BundleAvoidanceSubmission":
        records = [{"id": data["bundle_id"], "points": data["points"]}]
    elif data.get("format") == "BundleAvoidanceSubmissionSet":
        records = data["bundles"]
    else:
        raise ValueError("unknown submission format")
    originals = {item["id"]: item for item in bundles}
    paths = {}
    for item in records:
        bundle_id = item["id"]
        if not isinstance(bundle_id, int) or isinstance(bundle_id, bool) or bundle_id not in originals or bundle_id in paths:
            raise ValueError(f"unknown or duplicate bundle id: {bundle_id}")
        if any(key in item for key in ("radius", "bundle_radius", "bundle_width")):
            raise ValueError("submission must not redefine bundle width")
        original = originals[bundle_id]["points"]
        points = validate_candidate(original, item["points"], scene)
        paths[bundle_id] = points
    if not paths:
        raise ValueError("submission contains no paths")
    return paths


def path_length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def corresponding_displacement(original, candidate):
    samples = np.linspace(0, 1, 1001)
    aligned = []
    for points in (original, candidate):
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        aligned.append(np.column_stack([np.interp(samples, arc / arc[-1], points[:, axis]) for axis in range(3)]))
    return float(np.linalg.norm(aligned[0] - aligned[1], axis=1).max())


def collision_pairs(index, bundles):
    return {tuple(sorted((item["id"], int(hit["cable_b"]))))
            for item in bundles for hit in index.query_cables(item["id"])}


def evaluate_paths(bundles, occupancy, scene, paths):
    started = perf_counter()
    before_index = InterferenceIndex(bundles, occupancy, scene)
    after_bundles = [{**item, "points": paths.get(item["id"], item["points"])} for item in bundles]
    after_index = InterferenceIndex(after_bundles, occupancy, scene) if paths else before_index
    before_pairs = collision_pairs(before_index, bundles)
    after_pairs = collision_pairs(after_index, bundles) if paths else before_pairs
    before_members = {name for pair in before_pairs for name in pair}
    after_members = {name for pair in after_pairs for name in pair}
    rows = []
    for item, candidate in zip(bundles, after_bundles):
        bundle_id = item["id"]
        before = before_index.cable_collision_length(bundle_id)
        after = after_index.cable_collision_length(bundle_id) if paths else before
        chassis_before = before_index.chassis_collision_length(bundle_id)
        chassis_after = after_index.chassis_collision_length(bundle_id) if bundle_id in paths else chassis_before
        chassis_before_hit = bool(before_index.query_chassis(bundle_id))
        chassis_after_hit = bool(after_index.query_chassis(bundle_id)) if bundle_id in paths else chassis_before_hit
        original_length, candidate_length = path_length(item["points"]), path_length(candidate["points"])
        rows.append({"id": bundle_id, "submitted": bundle_id in paths, "before_length_mm": before,
                     "after_length_mm": after, "improvement_pct": None if before <= 1e-9 else 100 * (before - after) / before,
                     "before_collides": bundle_id in before_members, "after_collides": bundle_id in after_members,
                     "chassis_before_length_mm": chassis_before, "chassis_after_length_mm": chassis_after,
                     "chassis_before_collides": chassis_before_hit, "chassis_after_collides": chassis_after_hit,
                     "original_path_length_mm": original_length, "candidate_path_length_mm": candidate_length,
                     "path_length_increase_pct": 100 * (candidate_length / original_length - 1),
                     "sampled_max_displacement_mm": corresponding_displacement(item["points"], candidate["points"]) if bundle_id in paths else 0.0})
    before, after = sum(row["before_length_mm"] for row in rows), sum(row["after_length_mm"] for row in rows)
    selected = [row for row in rows if row["submitted"]]
    selected_before, selected_after = sum(row["before_length_mm"] for row in selected), sum(row["after_length_mm"] for row in selected)
    chassis_before = sum(row["chassis_before_length_mm"] for row in rows)
    chassis_after = sum(row["chassis_after_length_mm"] for row in rows)
    return {"format": "BundleInterferenceEvaluation", "version": "2.0", "primary_metric": "cable_collision_length",
            "units": "Millimeters", "bundle_count": len(bundles),
            "submitted_count": len(paths), "total_before_length_mm": before, "total_after_length_mm": after,
            "total_improvement_pct": None if before <= 1e-9 else 100 * (before - after) / before,
            "submitted_improvement_pct": None if selected_before <= 1e-9 else 100 * (selected_before - selected_after) / selected_before,
            "before_cable_colliding_bundles": len(before_members), "after_cable_colliding_bundles": len(after_members),
            "chassis_before_length_mm": chassis_before, "chassis_after_length_mm": chassis_after,
            "chassis_improvement_pct": None if chassis_before <= 1e-9 else 100 * (chassis_before - chassis_after) / chassis_before,
            "before_chassis_colliding_bundles": sum(row["chassis_before_collides"] for row in rows),
            "after_chassis_colliding_bundles": sum(row["chassis_after_collides"] for row in rows),
            "before_pair_count": len(before_pairs), "after_pair_count": len(after_pairs),
            "new_pairs": sorted(map(list, after_pairs - before_pairs)), "removed_pairs": sorted(map(list, before_pairs - after_pairs)),
            "before_pairs": sorted(map(list, before_pairs)), "after_pairs": sorted(map(list, after_pairs)),
            "evaluation_seconds": perf_counter() - started, "bundles": rows}


def main():
    parser = argparse.ArgumentParser(description="Evaluate original data or a single/multiple-bundle submission")
    parser.add_argument("--solution", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    args = parser.parse_args()
    bundles, occupancy, scene = load_data()
    report = evaluate_paths(bundles, occupancy, scene, load_submission(args.solution, bundles, scene))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    percent = report["total_improvement_pct"]
    print(f"Evaluated {len(bundles)} bundles; submitted {report['submitted_count']}")
    print(f"Cable-cable collided length (primary): {report['total_before_length_mm']:.6f} -> {report['total_after_length_mm']:.6f} mm")
    print(f"Cable-cable scene improvement: {percent:.2f}%" if percent is not None else "Cable-cable scene improvement: N/A")
    print(f"Cable pairs: {report['before_pair_count']} -> {report['after_pair_count']}; new {len(report['new_pairs'])}")
    print(f"Chassis collided length (auxiliary): {report['chassis_before_length_mm']:.6f} -> {report['chassis_after_length_mm']:.6f} mm")
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
