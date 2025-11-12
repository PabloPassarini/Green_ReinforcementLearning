#!/usr/bin/env python3
"""Run experiments: run one algorithm fully, then the other (if requested)."""

import argparse
from pathlib import Path

import numpy as np
import tsplib95

from algorithms.q_learning import QLearningTrainer
from algorithms.sarsa import SarsaTrainer


def get_instance(filename: str) -> tsplib95.models.StandardProblem:
    """Load TSPLIB instance from the project's instances directory."""
    base_dir = Path(__file__).resolve().parent
    instance_path = base_dir / "instances" / filename
    return tsplib95.load(instance_path)


def run_algorithm(
    algorithm: str,
    instances: list,
    episodes: int,
    epsilon: float,
    alpha: float,
) -> None:
    """Run the selected algorithm across all instances and hyperparameters."""
    epsilon_decay_types = ["linear", "concave", "convex", "step"]
    reward_types = ["R1", "R2", "R3"]
    gamma_set = [0.01, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.99]

    for instance_name in instances:
        problem = get_instance(instance_name)
        nodes = list(problem.get_nodes())
        dist_matrix = np.array([[problem.get_weight(i, j) for j in nodes] for i in nodes])
        n_points = problem.dimension

        for gamma in gamma_set:
            for e_type in epsilon_decay_types:
                for r_type in reward_types:
                    print(
                        f"{algorithm.upper()}: instance={instance_name} "
                        f"e_decay={e_type} reward={r_type} gamma={gamma}"
                    )

                    if algorithm == "qlearning":
                        trainer = QLearningTrainer(
                            instance=instance_name,
                            r_type=r_type,
                            e_type=e_type,
                            matrix_d=dist_matrix,
                            n_points=n_points,
                            episodes=episodes,
                            alpha=alpha,
                            gamma=gamma,
                            epsilon=epsilon,
                        )
                        trainer.train()

                    elif algorithm == "sarsa":
                        trainer = SarsaTrainer(
                            instance=instance_name,
                            r_type=r_type,
                            e_type=e_type,
                            matrix_d=dist_matrix,
                            n_points=n_points,
                            episodes=episodes,
                            alpha=alpha,
                            gamma=gamma,
                            epsilon=epsilon,
                        )
                        trainer.train()

                    else:
                        raise ValueError(f"Unknown algorithm: {algorithm}")


def main() -> None:
    """Parse CLI and run experiments in the requested order."""
    parser = argparse.ArgumentParser(description="Run TSP RL experiments")
    parser.add_argument(
        "--algorithm",
        choices=["qlearning", "sarsa", "both"],
        default="both",
        help="Which algorithm to run (qlearning, sarsa, or both).",
    )
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes per run.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon value.")
    parser.add_argument("--alpha", type=float, default=0.01, help="Learning rate.")
    parser.add_argument(
        "--instances",
        nargs="+",
        default=[
            "berlin52.tsp",
            "br17.atsp",
            "eil51.tsp",
            "ftv33.atsp",
            "ftv64.atsp",
            "kroA100.tsp",
            "st70.tsp",
            "tsp225.tsp",
        ],
        help="List of instance filenames located in ./instances",
    )
    args = parser.parse_args()

    # Run in the requested order: if both, run qlearning fully then sarsa fully.
    if args.algorithm == "qlearning":
        run_algorithm("qlearning", args.instances, args.episodes, args.epsilon, args.alpha)
    elif args.algorithm == "sarsa":
        run_algorithm("sarsa", args.instances, args.episodes, args.epsilon, args.alpha)
    else:  # both
        run_algorithm("qlearning", args.instances, args.episodes, args.epsilon, args.alpha)
        run_algorithm("sarsa", args.instances, args.episodes, args.epsilon, args.alpha)


if __name__ == "__main__":
    main()
