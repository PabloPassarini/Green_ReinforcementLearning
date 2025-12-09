#!/usr/bin/env python3
"""
plot_bestpaths.py

Reads master_summary.csv from a user-specified folder and the corresponding .tsp instance,
plots the route indicated in BestPath for each line (run_index, gamma, BestEpisode).
Saves a PNG per combination in plots/<instance>/run_<idx>_gamma_<val>.png
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_tsp(path: Path) -> dict[int, tuple[float, float]]:
    """Parse a TSPLIB .tsp file and return coordinates as {id: (x, y)}."""
    coords: dict[int, tuple[float, float]] = {}
    with path.open() as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("NODE_COORD_SECTION"):
            start = i + 1
            break

    if start is None:
        raise ValueError(f"NODE_COORD_SECTION not found in {path}")

    for line in lines[start:]:
        s = line.strip()
        if not s or s.upper().startswith("EOF"):
            break
        parts = s.split()
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
        except ValueError:
            continue
        coords[idx] = (x, y)
    return coords


def parse_path_string(path_str: str) -> list[int]:
    """Convert a BestPath string into a list of node IDs."""
    raw = re.sub(r'[\[\]\(\)]', ' ', path_str)
    raw = re.sub(r'->', ' ', raw)
    raw = raw.replace(',', ' ')
    parts = re.split(r'\s+', raw.strip())
    return [int(p) for p in parts if p.isdigit()]


def plot_route(coords: dict[int, tuple[float, float]],
               route: list[int],
               title: str,
               best_episode: int,
               out_file: Path) -> None:
    """Plot a route over the node coordinates and save to file."""
    xs, ys = [], []
    for pid in route:
        if pid not in coords:
            continue
        x, y = coords[pid]
        xs.append(x)
        ys.append(y)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        [x for _, (x, y) in coords.items()],
        [y for _, (x, y) in coords.items()],
        color="blue",
        marker="o",
        label="Nodes",
    )
    plt.plot(xs, ys, "-", label=f"Best Path (Best Episode={best_episode})")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    # Ask user for the folder containing master_summary.csv
    folder = input("Enter the folder containing master_summary.csv: ").strip()
    summary_path = Path(folder) / "master_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"master_summary.csv not found at {summary_path}")

    df = pd.read_csv(summary_path)

    for _, row in df.iterrows():
        inst_raw = str(row["instance"]).strip()

        if inst_raw.lower().endswith(".atsp"):
            print(f"Skipping ATSP instance: {inst_raw}")
            continue

        inst_name = (
            inst_raw if inst_raw.lower().endswith(".tsp") else inst_raw + ".tsp"
        )
        tsp_file = Path("instances") / inst_name

        try:
            coords = read_tsp(tsp_file)
        except Exception as exc:
            print(f"Error reading {tsp_file}: {exc}")
            continue

        bestpath_str = str(row["BestPath"]).strip()
        route = parse_path_string(bestpath_str)

        # Adjust indexing if necessary
        if not all(pid in coords for pid in route):
            route = [pid + 1 for pid in route]

        run_idx = row["run_index"]
        gamma = row["gamma"]
        best_episode = int(row["BestEpisode"]) if "BestEpisode" in row else -1

        out_file = (
            Path("plots")
            / inst_name.replace(".tsp", "")
            / f"run_{run_idx}_gamma_{gamma}.png"
        )
        title = f"{inst_name} - run {run_idx}, gamma={gamma}"
        try:
            plot_route(coords, route, title, best_episode, out_file)
            print(f"Saved {out_file}")
        except Exception as exc:
            print(
                f"Failed to plot {inst_name} run {run_idx} gamma {gamma}: {exc}"
            )


if __name__ == "__main__":
    main()
