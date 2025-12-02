# algorithms/double_q.py
import random
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from codecarbon import EmissionsTracker

from algorithms.base_trainer import BaseTrainer
from utils.reward_utils import reward_function, epsilon_decay


class DoubleQTrainer(BaseTrainer):
    """
    Double Q-Learning trainer for TSP.
    Implements two Q-tables (Q1, Q2) and alternates updates between them.
    Saves only master_episodes.csv and master_summary.csv via BaseTrainer._save_results.
    """

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
        results_subdir: str = "double-q",
        run_index: int = 0,
    ) -> None:
        super().__init__(
            instance=instance,
            r_type=r_type,
            e_type=e_type,
            matrix_d=matrix_d,
            n_points=n_points,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            results_subdir=results_subdir,
            algorithm_name="double_q",
            run_index=run_index,
        )
        # override base q_table usage: we keep two tables
        self.q1_table = np.zeros((n_points, n_points))
        self.q2_table = np.zeros((n_points, n_points))

    def train(self) -> Tuple[str, str]:
        """
        Run Double Q-Learning training loop.
        Returns paths to master episodes and master summary (via BaseTrainer._save_results).
        """
        tracker = self._start_tracker()

        try:
            for ep in range(self.episodes):
                # Select a random starting point
                current_point = random.randint(0, self.n_points - 1)
                unvisited = list(range(self.n_points))
                unvisited.remove(current_point)
                path: List[int] = [current_point]
                current_distance = 0.0

                while unvisited:
                    # epsilon-greedy using average of Q1 and Q2 for action selection
                    if random.uniform(0, 1) < self.epsilon:
                        next_point = random.choice(unvisited)
                    else:
                        q_values = {
                            p: (self.q1_table[current_point, p] + self.q2_table[current_point, p]) / 2.0
                            for p in unvisited
                        }
                        next_point = max(q_values, key=q_values.get)

                    distance = float(self.matrix_d[current_point][next_point])
                    reward = reward_function(self.r_type, distance)

                    # Randomly choose which table to update (50/50)
                    if random.random() < 0.5:
                        # Update Q1 using Q2 for the bootstrap
                        # choose best action according to Q1 at next state
                        best_action = int(np.argmax(self.q1_table[next_point, :]))
                        target = reward + self.gamma * self.q2_table[next_point, best_action]
                        self.q1_table[current_point, next_point] += self.alpha * (
                            target - self.q1_table[current_point, next_point]
                        )
                    else:
                        # Update Q2 using Q1 for the bootstrap
                        best_action = int(np.argmax(self.q2_table[next_point, :]))
                        target = reward + self.gamma * self.q1_table[next_point, best_action]
                        self.q2_table[current_point, next_point] += self.alpha * (
                            target - self.q2_table[current_point, next_point]
                        )

                    current_distance += distance
                    path.append(next_point)
                    current_point = next_point
                    unvisited.remove(next_point)

                # close the cycle back to start
                last_point = path[-1]
                current_distance += float(self.matrix_d[last_point][path[0]])
                path.append(path[0])

                # record episode result
                self.distance_history.append(current_distance)
                if current_distance < self.best_distance:
                    self.best_distance = current_distance
                    self.best_path = path.copy()

                # decay epsilon for next episode
                self.epsilon = epsilon_decay(self.e_type, ep, self.episodes)

            # finalize emissions tracking and save masters
            emissions_data = self._finalize_tracking(tracker)
            return self._save_results(emissions_data)

        except Exception:
            # ensure tracker is stopped on exception
            try:
                _ = tracker.stop()
            except Exception:
                pass
            raise
