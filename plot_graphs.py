#!/usr/bin/env python3

import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# TSPLIB PARSER
# ============================================================

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
    raw = re.sub(r"[\[\]\(\)]", " ", path_str)
    raw = re.sub(r"->", " ", raw)
    raw = raw.replace(",", " ")
    parts = re.split(r"\s+", raw.strip())
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    # Ask user for folder containing both CSVs
    folder = input("Enter the folder containing master_episodes.csv and master_summary.csv: ").strip()
    folder = Path(folder)
    summary_path = folder / "master_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"master_summary.csv not found at {summary_path}")

    
    df_sum = pd.read_csv(summary_path)

    # Output directories
    plots_dir = folder / "plots"
    curves_dir = plots_dir / "learning_curves"
    bestpaths_dir = plots_dir / "bestpaths"

    curves_dir.mkdir(parents=True, exist_ok=True)
    bestpaths_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # PART 1 — LEARNING CURVES
    # ============================================================

    epsilon_decay_types = ["linear", "convex", "concave", "step", "fixed"]
    epsilon_init = [0.01, 0.05, 0.10]
    reward_types = ["R1", "R2", "R3"]
    instances = [
            "br17.atsp",
            "berlin52.tsp",
            "eil51.tsp",
            "ftv33.atsp",
            "ftv64.atsp",
            "kroA100.tsp",
            "st70.tsp",
            "tsp225.tsp",
        ]

    for e_type in epsilon_decay_types:
        for epsilon_i in epsilon_init:
            for r_type in reward_types:

                for instance in instances:
                    for run_index in range(4, 6):

                        episodes_path = folder / f"{run_index}_{instance}_master_episodes.csv"
                        if not episodes_path.exists():
                            raise FileNotFoundError(f"File not found: {episodes_path}")

                        df = pd.read_csv(episodes_path)

                        df_plot = df[
                            (df["instance"] == instance) &
                            (df["run_index"] == run_index) &
                            (df["e_type"] == e_type) &
                            (df["r_type"] == r_type) &
                            (df["epsilon_init"] == epsilon_i)
                        ].sort_values("episode")

                        if df_plot.empty:
                            print(f"⚠ No data: instance={instance}, run={run_index}")
                            continue

                        # ==================================================
                        # Plot 1 — Distance × Episode
                        # ==================================================
                        plt.figure(figsize=(10, 6))
                        plt.plot(df_plot["episode"], df_plot["distance"])
                        plt.xlabel("Episode")
                        plt.ylabel("Distance")
                        plt.title(f"Distance × Episode\n({instance}, run={run_index}, e={e_type}, r={r_type})")
                        plt.grid(True)
                        plt.tight_layout()

                        out_dist = curves_dir / (
                            f"distance_episode_{instance}_run_{run_index}"
                            f"_{e_type}_{r_type}.png"
                        )
                        plt.savefig(out_dist, dpi=150)
                        plt.close()

                        print(f"Saved: {out_dist}")

                        # ==================================================
                        # Plot 2 — Distance + Epsilon × Episode
                        # ==================================================
                        fig, ax1 = plt.subplots(figsize=(10, 6))

                        ax1.plot(df_plot["episode"],
                                df_plot["distance"],
                                label="Distance",
                                linewidth=1.2)
                        ax1.set_xlabel("Episode")
                        ax1.set_ylabel("Distance")
                        ax1.grid(True)

                        ax2 = ax1.twinx()
                        ax2.plot(
                            df_plot["episode"],
                            df_plot["epsilon"],
                            linestyle=":",
                            linewidth=2.0,
                            color="orange",
                            label="Epsilon"
                        )
                        ax2.set_ylabel("Epsilon")

                        lines1, labels1 = ax1.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

                        plt.title(
                            f"Distance and Epsilon × Episode\n"
                            f"({instance}, run={run_index}, e={e_type}, r={r_type})"
                        )
                        plt.tight_layout()

                        out_dual = curves_dir / (
                            f"distance_epsilon_episode_{instance}_run_{run_index}"
                            f"__{e_type}__{r_type}.png"
                        )
                        plt.savefig(out_dual, dpi=150)
                        plt.close()

                        print(f"Saved: {out_dual}")

    # ============================================================
    # PART 2 — BESTPATH ROUTE PLOTS
    # ============================================================

    print("\n=== Generating BestPath route plots ===")

    for _, row in df_sum.iterrows():
        inst_raw = str(row["instance"]).strip()

        # Skip ATSP instances
        if inst_raw.lower().endswith(".atsp"):
            print(f"Skipping ATSP instance: {inst_raw}")
            continue

        inst_name = inst_raw if inst_raw.lower().endswith(".tsp") else inst_raw + ".tsp"
        tsp_file = Path("instances") / inst_name

        try:
            coords = read_tsp(tsp_file)
        except Exception as exc:
            print(f"Error reading {tsp_file}: {exc}")
            continue

        route = parse_path_string(str(row["BestPath"]).strip())

        # Adjust indexing if necessary
        if not all(pid in coords for pid in route):
            route = [pid + 1 for pid in route]

        run_idx = row["run_index"]
        r_type = row["r_type"]
        e_type = row["e_type"]
        best_episode = int(row["BestEpisode"]) if "BestEpisode" in row else -1

        out_file = bestpaths_dir / f"run_{run_idx}_{r_type}_{e_type}__{inst_raw.replace('.tsp','')}.png"
        title = f"{inst_raw} - run {run_idx}, {r_type}, {e_type}"

        try:
            plot_route(coords, route, title, best_episode, out_file)
            print(f"Saved: {out_file}")
        except Exception as exc:
            print(f"Failed to plot {inst_raw} run {run_idx} {r_type} {e_type}: {exc}")


if __name__ == "__main__":
    main()
