from pathlib import Path
from typing import List, Tuple

import numpy as np
from codecarbon import EmissionsTracker

from utils.file_utils import save_per_episode, save_summary
from utils.reward_utils import epsilon_decay


class BaseTrainer:
    """Base trainer with common utilities for tabular RL on TSP."""

    def __init__(
        self,
        instance: str,
        r_type: str,
        e_type: str,
        matrix_d: np.ndarray,
        n_points: int,
        episodes: int,
        alpha: float,
        gamma: float,
        epsilon: float,
        results_subdir: str,
        algorithm_name: str,
    ) -> None:
        self.instance = instance
        self.r_type = r_type
        self.e_type = e_type
        self.matrix_d = matrix_d
        self.n_points = n_points
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = np.zeros((n_points, n_points))
        self.best_path: List[int] = []
        self.best_distance: float = float("inf")
        self.distance_history: List[float] = []

        self.results_dir = Path("results") / results_subdir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.base_name = f"{algorithm_name}_{self.instance}_{self.r_type}_{self.e_type}_gamma{self.gamma}"

    def _start_tracker(self) -> EmissionsTracker:
        """Start and return a CodeCarbon EmissionsTracker saving to results_dir."""
        tracker = EmissionsTracker(
            project_name=f"{self.base_name}_project",
            output_dir=self.results_dir,
            output_file=f"{self.base_name}_emissions.csv",
        )
        tracker.start()
        return tracker

    def _finalize_tracking(self, tracker: EmissionsTracker):
        """Stop tracker and return CodeCarbon's final_emissions_data object."""
        _ = tracker.stop()  # tracker.stop() returns float; final data available on tracker
        return tracker.final_emissions_data

    def _save_results(self, emissions_data) -> Tuple[str, str]:
        """Save per-episode CSV and summary CSV and return their paths as strings."""
        best_episode = int(min(range(len(self.distance_history)), key=self.distance_history.__getitem__))
        per_path = save_per_episode(self.results_dir, self.base_name, self.distance_history)
        summary_path = save_summary(
            self.results_dir,
            self.base_name,
            instance=self.instance,
            r_type=self.r_type,
            e_type=self.e_type,
            gamma=self.gamma,
            best_episode=best_episode,
            best_distance=self.best_distance,
            best_path=self.best_path,
            emissions_data=emissions_data,
        )
        return str(per_path), str(summary_path)

    def train(self) -> Tuple[str, str]:
        """Train method to implement in subclasses."""
        raise NotImplementedError("Subclasses must implement train()")
