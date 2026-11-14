"""Try one simulated annealing decision with the paths from step 1."""

import argparse
import json
import math
from pathlib import Path
import random


def score(displacement):
    report_path = Path(__file__).with_name(f"step1_evaluation_y{displacement:+g}.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report["total_after_length_mm"]


def acceptance_probability(delta, temperature):
    """Always accept improvements and sometimes accept a worse score."""
    if delta <= 0:
        return 1.0
    return math.exp(-delta / temperature)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("temperature must be finite and positive")

    current_dy, candidate_dy = -5, -2
    try:
        current_score, candidate_score = score(current_dy), score(candidate_dy)
    except FileNotFoundError as error:
        parser.error(f"run step1_demo.py with --dy -5 and --dy -2 first: {error.filename}")

    # The lesson starts with the -5 mm path as both the current and best state.
    best_dy, best_score = current_dy, current_score
    delta = candidate_score - current_score
    probability = acceptance_probability(delta, args.temperature)
    draw = random.Random(args.seed).random()
    accepted = draw < probability
    if accepted:
        current_dy, current_score = candidate_dy, candidate_score
        if current_score < best_score:
            best_dy, best_score = current_dy, current_score

    print(f"Temperature: {args.temperature:g}; seed: {args.seed}")
    print(f"Candidate score: {candidate_score:.6f} mm; change: {delta:+.6f} mm")
    print(f"Acceptance probability: {probability:.4%}; random draw: {draw:.6f}")
    print("Decision: accept" if accepted else "Decision: reject")
    print(f"Current path after decision: dy={current_dy:g} mm, score={current_score:.6f} mm")
    print(f"Best path so far: dy={best_dy:g} mm, score={best_score:.6f} mm")


if __name__ == "__main__":
    main()
