"""Run simulated annealing with optional interference-guided proposals."""

import argparse
import json
import math
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np

from step2_accept import acceptance_probability


def propose(points, rng, max_shift, conflicts=None, direction="random"):
    """Move a contiguous span, with zero shift at both boundaries."""
    span = max(3, len(points) // 3)
    last_start = len(points) - span
    if conflicts:
        target = rng.choice(sorted(conflicts))
        # Center the moved span around the hit, clipping near either fixed endpoint.
        start = max(0, min(last_start, target - (span - 2) // 2))
    else:
        target = None
        start = rng.randrange(last_start + 1)
    end = start + span
    axis = rng.randrange(3)
    shift = rng.uniform(-max_shift, max_shift)

    # Sample the same movement magnitude in both modes, then choose its direction.
    displacement = np.zeros(3)
    displacement[axis] = shift
    move = {"first_point_index": start, "last_point_index": end - 1,
            "axis": "XYZ"[axis], "shift_mm": shift, "target_segment": target}
    if direction == "away" and target is not None:
        hit = conflicts[target]
        away = np.asarray(hit["point_a"]) - np.asarray(hit["point_b"])
        distance = np.linalg.norm(away)
        if distance > 1e-9:
            displacement = abs(shift) * away / distance
            move.update(axis=None, shift_mm=abs(shift), neighbor_id=int(hit["cable_b"]))

    t = np.linspace(0, 1, span)
    weights = 4 * t * (1 - t)
    candidate = points.copy()
    candidate[start:end] += weights[:, None] * displacement
    move["displacement_mm"] = displacement.tolist()
    return candidate, move


def select_bundle(movable, lengths, rng, selection):
    uniform_probability = 1 / len(movable)
    if selection == "uniform":
        return rng.choice(movable), uniform_probability

    total = sum(lengths[i] for i in movable)
    if total == 0:
        return rng.choice(movable), uniform_probability

    # Mix 80% interference-weighted sampling with 20% uniform sampling.
    probabilities = [0.8 * lengths[i] / total + 0.2 * uniform_probability for i in movable]
    position = rng.choices(range(len(movable)), weights=probabilities)[0]
    return movable[position], probabilities[position]


def affected_bundles(bundles, moved, candidate, first, last):
    """Find cables close enough to either position of the edited span to change score."""
    old_points = bundles[moved]["points"][first:last + 1]
    new_points = candidate[first:last + 1]
    starts = np.concatenate((old_points[:-1], new_points[:-1]))
    ends = np.concatenate((old_points[1:], new_points[1:]))
    lo = np.minimum(starts, ends) - bundles[moved]["radius"]
    hi = np.maximum(starts, ends) + bundles[moved]["radius"]
    affected = [moved]
    for i, bundle in enumerate(bundles):
        if i == moved:
            continue
        points, radius = bundle["points"], bundle["radius"]
        peer_lo = np.minimum(points[:-1], points[1:]) - radius
        peer_hi = np.maximum(points[:-1], points[1:]) + radius
        if np.any(np.all(peer_lo[:, None] <= hi, axis=2) & np.all(peer_hi[:, None] >= lo, axis=2)):
            affected.append(i)
    return affected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="Directory containing the supplied evaluate.py")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=5.0, help="Starting temperature")
    parser.add_argument("--cooling", type=float, default=0.90, help="Temperature multiplier per step")
    parser.add_argument("--max-shift", type=float, default=6.0, help="Displacement limit per move, in mm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection", choices=("uniform", "weighted"), default="uniform")
    parser.add_argument("--segment", choices=("random", "conflict"), default="random",
                        help="Choose a random span or a span around an interfering segment")
    parser.add_argument("--direction", choices=("random", "away"), default="random",
                        help="Use a random axis or move away from a nearby interfering cable")
    parser.add_argument("--chassis-weight", type=float, default=0,
                        help="Weight of chassis collision length in the search objective")
    parser.add_argument("--incremental", action="store_true",
                        help="Reuse exact cable scores outside the neighborhood of the edited span")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")
    if not 0 < args.cooling < 1:
        parser.error("--cooling must be between 0 and 1")
    if not math.isfinite(args.max_shift) or args.max_shift <= 0:
        parser.error("--max-shift must be finite and positive")
    if args.direction == "away" and args.segment != "conflict":
        parser.error("--direction away requires --segment conflict")
    if not math.isfinite(args.chassis_weight) or args.chassis_weight < 0:
        parser.error("--chassis-weight must be finite and nonnegative")

    project = args.project.resolve()
    if not (project / "evaluate.py").is_file():
        parser.error("project must contain the supplied evaluate.py")
    sys.path.insert(0, str(project))
    from evaluate import evaluate_paths, load_submission
    from interference import InterferenceIndex
    from main import load_data
    from optimize import validate_candidate

    def collision_lengths(index, scene_bundles):
        return [index.cable_collision_length(item["id"]) for item in scene_bundles]

    original, occupancy, scene = load_data(project)
    movable = [i for i, item in enumerate(original) if len(item["points"]) >= 3]
    if not movable:
        parser.error("the scene has no internal points to move")
    rng = random.Random(args.seed)
    temperature = args.temperature
    history = []

    started = perf_counter()
    current_scene = best_scene = original
    chassis_args = (occupancy, scene) if args.chassis_weight else ()
    current_index = InterferenceIndex(original, *chassis_args)
    current_lengths = collision_lengths(current_index, original)
    current_chassis_lengths = ([current_index.chassis_collision_length(item["id"]) for item in original]
                               if args.chassis_weight else [0.0] * len(original))
    baseline_cable = sum(current_lengths)
    best_cable = baseline_cable
    best_chassis = sum(current_chassis_lengths)
    baseline_score = current_score = best_score = baseline_cable + args.chassis_weight * best_chassis
    evaluations = 1  # The initial score plus one for each valid candidate.
    cable_queries = len(original)
    for step in range(1, args.steps + 1):
        index, selection_probability = select_bundle(movable, current_lengths, rng, args.selection)
        item = current_scene[index]
        conflicts = {}
        if args.segment == "conflict" and current_lengths[index] > 0:
            hits = current_index.query_cables(item["id"], max_hits=None)
            # Retain one hit per segment. The closest neighbor guides the direction.
            for hit in hits:
                segment = hit["segment_a"]
                if segment not in conflicts or hit["distance"] < conflicts[segment]["distance"]:
                    conflicts[segment] = hit
        new_points, move = propose(item["points"], rng, args.max_shift, conflicts, args.direction)
        before = current_score
        row = {"step": step, "bundle_id": item["id"], **move,
               "bundle_before_mm": current_lengths[index], "selection_probability": selection_probability,
               "temperature": temperature, "before_mm": before, "decision": "invalid"}

        try:
            validate_candidate(original[index]["points"], new_points, scene)
        except ValueError as error:
            row["validation_error"] = str(error)
        else:
            # Copy the list and replace one entry. Earlier states retain their paths.
            candidate_scene = current_scene.copy()
            candidate_scene[index] = {**item, "points": new_points}
            # Rebuild with every candidate path to score changes affecting neighbors.
            candidate_index = InterferenceIndex(candidate_scene, *chassis_args)
            if args.incremental:
                affected = affected_bundles(current_scene, index, new_points,
                                            move["first_point_index"], move["last_point_index"])
                candidate_lengths = current_lengths.copy()
                for peer in affected:
                    candidate_lengths[peer] = candidate_index.cable_collision_length(candidate_scene[peer]["id"])
                cable_queries += len(affected)
            else:
                candidate_lengths = collision_lengths(candidate_index, candidate_scene)
                cable_queries += len(candidate_scene)
            candidate_cable = sum(candidate_lengths)
            candidate_chassis_lengths = current_chassis_lengths.copy()
            if args.chassis_weight:
                # The chassis is fixed, so only the moved cable's chassis length changes.
                candidate_chassis_lengths[index] = candidate_index.chassis_collision_length(item["id"])
            candidate_chassis = sum(candidate_chassis_lengths)
            candidate_score = candidate_cable + args.chassis_weight * candidate_chassis
            evaluations += 1
            delta = candidate_score - current_score
            probability = acceptance_probability(delta, temperature)
            draw = rng.random()
            accepted = draw < probability
            if accepted:
                current_scene, current_score = candidate_scene, candidate_score
                current_index = candidate_index
                current_lengths = candidate_lengths
                current_chassis_lengths = candidate_chassis_lengths
                if current_score < best_score:
                    best_scene, best_score = current_scene, current_score
                    best_cable, best_chassis = candidate_cable, candidate_chassis
            row.update(candidate_mm=candidate_score, delta_mm=delta,
                       acceptance_probability=probability, random_draw=draw,
                       decision="accept" if accepted else "reject")
            if args.chassis_weight:
                row.update(candidate_cable_mm=candidate_cable, candidate_chassis_mm=candidate_chassis)

        row.update(current_mm=current_score, best_mm=best_score)
        history.append(row)
        if step <= 3 or step % 10 == 0 or step == args.steps:
            print(f"{step:3d}  ID {item['id']:>2}  T={temperature:.3f}  {row['decision']:>7}  "
                  f"current={current_score:.3f}  best={best_score:.3f}", flush=True)
        temperature = max(sys.float_info.min, temperature * args.cooling)
    search_seconds = perf_counter() - started

    default_folder = "step2_run" if args.selection == "uniform" else "step3_weighted_run"
    if args.segment == "conflict":
        default_folder = f"step3_{args.selection}_conflict_run"
    if args.direction == "away":
        default_folder = f"step3_{args.selection}_away_run"
    if args.chassis_weight:
        default_folder = f"{default_folder}_chassis{args.chassis_weight:g}"
    output_dir = args.output_dir or Path(__file__).resolve().parent / default_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    solution_path = output_dir / "solution.json"
    # Whole scene submission also handles a best solution equal to the baseline.
    submission = {"format": "BundleAvoidanceSubmissionSet", "units": "Millimeters",
                  "bundles": [{"id": item["id"], "points": item["points"].tolist()} for item in best_scene]}
    solution_path.write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")

    paths = load_submission(solution_path, original, scene)
    report = evaluate_paths(original, occupancy, scene, paths)
    if not math.isclose(report["total_after_length_mm"], best_cable, rel_tol=0, abs_tol=1e-6):
        raise RuntimeError("the saved solution did not reproduce the search score")
    if args.chassis_weight and not math.isclose(report["chassis_after_length_mm"], best_chassis,
                                                rel_tol=0, abs_tol=1e-6):
        raise RuntimeError("the saved solution did not reproduce the chassis score")
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    run = {"selection": args.selection, "segment": args.segment, "direction": args.direction,
           "steps": args.steps, "chassis_weight": args.chassis_weight, "incremental": args.incremental,
           "initial_temperature": args.temperature, "cooling": args.cooling,
           "max_shift_mm": args.max_shift, "seed": args.seed, "scene_evaluations": evaluations,
           "cable_length_queries": cable_queries,
           "search_seconds": search_seconds, "baseline_mm": baseline_score, "best_mm": best_score,
           "history": history}
    (output_dir / "trace.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")

    if args.chassis_weight:
        print(f"Best objective: {baseline_score:.4f} -> {best_score:.4f} mm (cable + weighted chassis)")
        print(f"Cable interference: {baseline_cable:.4f} -> {best_cable:.4f} mm "
              f"({report['total_improvement_pct']:+.2f}% improvement)")
        print(f"Chassis interference: {report['chassis_before_length_mm']:.4f} -> {best_chassis:.4f} mm")
    else:
        print(f"Best score: {baseline_score:.4f} -> {best_score:.4f} mm "
              f"({report['total_improvement_pct']:+.2f}% improvement)")
    print(f"Search: {search_seconds:.2f} s, {evaluations} scene evaluations including the baseline")
    print(f"Artifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
