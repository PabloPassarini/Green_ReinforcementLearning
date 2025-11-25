# algorithms/dqn.py
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from codecarbon import EmissionsTracker

from algorithms.base_trainer import BaseTrainer
from utils.file_utils import save_per_episode, save_summary
from utils.reward_utils import reward_function, epsilon_decay


class QNet(nn.Module):
    """Simple MLP over state vector: [one-hot current_node | mask of unvisited]."""
    def __init__(self, n_points: int, hidden: int = 256):
        super().__init__()
        input_dim = n_points + n_points
        output_dim = n_points
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # shape: [batch, n_points]


def build_state(current: int, unvisited_mask: np.ndarray, n_points: int) -> np.ndarray:
    """State vector: concat one-hot(current) and unvisited mask."""
    one_hot = np.zeros(n_points, dtype=np.float32)
    one_hot[current] = 1.0
    return np.concatenate([one_hot, unvisited_mask.astype(np.float32)], axis=0)


class DQNTrainer(BaseTrainer):
    """
    Deep Q-Network trainer for TSP.
    State: one-hot current node + unvisited mask.
    Action: choose next node among unvisited (masked invalid actions).
    Reward: from reward_function(r_type, distance).
    """

    def __init__(
        self,
        instance: str,
        r_type: str,
        e_type: str,
        matrix_d: np.ndarray,
        n_points: int,
        episodes: int,
        alpha: float,  # used as learning rate
        gamma: float,
        epsilon: float,
        results_subdir: str = "dqn",
        run_index: int = 0,
        hidden: int = 256,
        target_update_freq: int = 10,
        batch_size: int = 64,
        replay_capacity: int = 50000,
        device: Optional[str] = None,
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
            algorithm_name="dqn",
            run_index=run_index,
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy_net = QNet(n_points, hidden).to(self.device)
        self.target_net = QNet(n_points, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # simple replay buffer
        self.capacity = replay_capacity
        self.buffer_s: List[np.ndarray] = []
        self.buffer_a: List[int] = []
        self.buffer_r: List[float] = []
        self.buffer_s2: List[np.ndarray] = []
        self.buffer_done: List[bool] = []

    def _push(self, s, a, r, s2, done):
        if len(self.buffer_s) >= self.capacity:
            # FIFO
            self.buffer_s.pop(0)
            self.buffer_a.pop(0)
            self.buffer_r.pop(0)
            self.buffer_s2.pop(0)
            self.buffer_done.pop(0)
        self.buffer_s.append(s)
        self.buffer_a.append(int(a))
        self.buffer_r.append(float(r))
        self.buffer_s2.append(s2)
        self.buffer_done.append(bool(done))

    def _sample_and_learn(self):
        if len(self.buffer_s) < self.batch_size:
            return
        idx = np.random.choice(len(self.buffer_s), size=self.batch_size, replace=False)
        s = torch.tensor(np.stack([self.buffer_s[i] for i in idx]), dtype=torch.float32, device=self.device)
        a = torch.tensor([self.buffer_a[i] for i in idx], dtype=torch.int64, device=self.device)
        r = torch.tensor([self.buffer_r[i] for i in idx], dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.stack([self.buffer_s2[i] for i in idx]), dtype=torch.float32, device=self.device)
        done = torch.tensor([self.buffer_done[i] for i in idx], dtype=torch.float32, device=self.device)

        # Q(s,a)
        q_values = self.policy_net(s)  # [B, n_points]
        q_sa = q_values.gather(1, a.view(-1, 1)).squeeze(1)

        # Target: r + gamma * max_a' Q_target(s', a') over valid actions
        with torch.no_grad():
            q_next_all = self.target_net(s2)  # [B, n_points]
            # mask invalid actions using the unvisited mask portion in s2
            n_points = self.n_points
            unvisited_mask = s2[:, n_points:] > 0.5  # boolean mask [B, n_points]
            # invalid actions get very negative value so they are not chosen
            masked_q_next = q_next_all.masked_fill(~unvisited_mask, -1e9)
            q_next_max = masked_q_next.max(dim=1).values
            target = r + self.gamma * (1.0 - done) * q_next_max

        loss = nn.SmoothL1Loss()(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def _select_action(self, current_point: int, unvisited_mask: np.ndarray, epsilon: float) -> int:
        valid_actions = np.where(unvisited_mask > 0.5)[0]
        if len(valid_actions) == 0:
            return current_point  # fallback; shouldn't happen
        if np.random.rand() < epsilon:
            return int(np.random.choice(valid_actions))

        state = torch.tensor(build_state(current_point, unvisited_mask, self.n_points), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_all = self.policy_net(state).squeeze(0).cpu().numpy()
        q_masked = np.full_like(q_all, -1e9, dtype=np.float32)
        q_masked[valid_actions] = q_all[valid_actions]
        return int(np.argmax(q_masked))

    def train(self) -> Tuple[str, str]:
        tracker = self._start_tracker()
        try:
            for ep in range(self.episodes):
                # Episode init
                current_point = int(np.random.randint(0, self.n_points))
                unvisited = list(range(self.n_points))
                unvisited.remove(current_point)
                unvisited_mask = np.zeros(self.n_points, dtype=np.float32)
                unvisited_mask[unvisited] = 1.0

                path = [current_point]
                current_distance = 0.0
                eps = epsilon_decay(self.e_type, ep, self.episodes)

                while len(unvisited) > 0:
                    next_point = self._select_action(current_point, unvisited_mask, eps)
                    distance = float(self.matrix_d[current_point][next_point])
                    reward = float(reward_function(self.r_type, distance))

                    # build next state and push transition
                    s = build_state(current_point, unvisited_mask, self.n_points)
                    # update masks
                    current_distance += distance
                    path.append(next_point)

                    # transition
                    unvisited_mask[next_point] = 0.0
                    unvisited.remove(next_point)
                    done = len(unvisited) == 0
                    s2 = build_state(next_point, unvisited_mask, self.n_points)

                    self._push(s, next_point, reward, s2, done)
                    self._sample_and_learn()

                    current_point = next_point

                # close cycle
                last_point = path[-1]
                current_distance += float(self.matrix_d[last_point][path[0]])
                path.append(path[0])

                self.distance_history.append(current_distance)
                if current_distance < self.best_distance:
                    self.best_distance = current_distance
                    self.best_path = path.copy()

                # periodic target update
                if (ep + 1) % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            emissions_data = self._finalize_tracking(tracker)
            return self._save_results(emissions_data)
        except Exception:
            # ensure tracker is stopped on error
            try:
                _ = tracker.stop()
            except Exception:
                pass
            raise

