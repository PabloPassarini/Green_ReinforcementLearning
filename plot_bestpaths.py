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

    episodes_path = folder / "master_episodes.csv"
    summary_path = folder / "master_summary.csv"

    if not episodes_path.exists():
        raise FileNotFoundError(f"master_episodes.csv not found at {episodes_path}")

    if not summary_path.exists():
        raise FileNotFoundError(f"master_summary.csv not found at {summary_path}")

    df_ep = pd.read_csv(episodes_path)
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

    epsilon_decay_types = ["linear", "convex", "concave", "step"]
    reward_types = ["R1", "R2", "R3"]

    for e_type in epsilon_decay_types:
        for r_type in reward_types:

            print(f"\n=== Processing learning curves: e_type={e_type}, r_type={r_type} ===")

            df_plot = df_ep[
                (df_ep["instance"] == "br17.atsp") &
                (df_ep["run_index"] == 1) &
                (df_ep["e_type"] == e_type) &
                (df_ep["r_type"] == r_type)
            ]

            if df_plot.empty:
                print(f"⚠ No data found for e_type={e_type}, r_type={r_type}")
                continue

            df_plot = df_plot.sort_values("episode")

            # -------------------------
            # Plot 1 — Distance × Episode
            # -------------------------
            plt.figure(figsize=(10, 6))
            plt.plot(df_plot["episode"], df_plot["distance"])
            plt.xlabel("Episode")
            plt.ylabel("Distance")
            plt.title(f"Distance × Episode\n(e_type={e_type}, r_type={r_type})")
            plt.grid(True)
            plt.tight_layout()

            out1 = curves_dir / f"distance_episode__e_{e_type}__r_{r_type}.png"
            plt.savefig(out1, dpi=150)
            plt.close()
            print(f"Saved: {out1}")

            # -------------------------
            # Plot 2 — Distance + Epsilon × Episode
            # -------------------------
            fig, ax1 = plt.subplots(figsize=(10, 6))

            ax1.plot(df_plot["episode"], df_plot["distance"], label="Distance")
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Distance")
            ax1.grid(True)

            ax2 = ax1.twinx()
            ax2.plot(df_plot["episode"], df_plot["epsilon"], linestyle="--",
                     color="orange", label="Epsilon")
            ax2.set_ylabel("Epsilon")

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

            plt.title(f"Distance and Epsilon × Episode\n(e_type={e_type}, r_type={r_type})")
            plt.tight_layout()

            out2 = curves_dir / f"distance_epsilon_episode__e_{e_type}__r_{r_type}.png"
            plt.savefig(out2, dpi=150)
            plt.close()
            print(f"Saved: {out2}")

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
        gamma = row["gamma"]
        best_episode = int(row["BestEpisode"]) if "BestEpisode" in row else -1

        out_file = bestpaths_dir / f"run_{run_idx}_gamma_{gamma}__{inst_raw.replace('.tsp','')}.png"
        title = f"{inst_raw} - run {run_idx}, gamma={gamma}"

        try:
            plot_route(coords, route, title, best_episode, out_file)
            print(f"Saved: {out_file}")
        except Exception as exc:
            print(f"Failed to plot {inst_raw} run {run_idx} gamma {gamma}: {exc}")


if __name__ == "__main__":
    main()
