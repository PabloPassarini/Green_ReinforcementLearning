from pathlib import Path
from typing import Dict, List

from datetime import datetime, timezone

import pandas as pd


def timestamp_tag() -> str:
    """Return compact UTC timestamp like 20251112T161530Z."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%z")


def ensure_dir(path: Path) -> Path:
    """Create directory if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_df_to_csv(df: pd.DataFrame, out_path: Path) -> None:
    """
    Append DataFrame to CSV; write header only if file does not exist.
    Ensures parent directory exists before writing.
    """
    ensure_dir(out_path.parent)
    header = not out_path.exists()
    # Pandas handles writing header based on the header flag.
    df.to_csv(out_path, mode="a", header=header, index=False)


def save_per_episode(
    results_dir: Path,
    base_name: str,
    distances: List[float],
    metadata: Dict[str, object],
    master_episodes_name: str = "master_episodes.csv",
) -> Path:
    """
    Save per-run episode CSV and append per-episode rows to a master CSV.

    metadata must include run_index, algorithm, instance, r_type, e_type, gamma.
    """
    ensure_dir(results_dir)
    #per_df = pd.DataFrame({"Episode": list(range(len(distances))), "Distance": distances})
    per_path = results_dir / f"{base_name}_results.csv"
    #per_df.to_csv(per_path, index=False)

    rows = []
    run_idx = metadata.get("run_index", 0)
    for i, d in enumerate(distances):
        rows.append(
            {
                "run_index": run_idx,
                "algorithm": metadata.get("algorithm"),
                "instance": metadata.get("instance"),
                #"r_type": metadata.get("r_type"),
                #"e_type": metadata.get("e_type"),
                "gamma": metadata.get("gamma"),
                "episode": i,
                "distance": d,
            }
        )
    master_df = pd.DataFrame(rows)
    master_path = results_dir / master_episodes_name
    append_df_to_csv(master_df, master_path)
    return per_path


def save_summary(
    results_dir: Path,
    base_name: str,
    summary_row: Dict[str, object],
    master_summary_name: str = "master_summary.csv",
) -> Path:
    """
    Save per-run summary file and append the single-row summary to a master CSV.

    summary_row should include a 'run_index' field and other summary columns.
    """
    ensure_dir(results_dir)
    summary_df = pd.DataFrame([summary_row])
    per_path = results_dir / f"{base_name}_summary.csv"
    #summary_df.to_csv(per_path, index=False)

    master_path = results_dir / master_summary_name
    append_df_to_csv(summary_df, master_path)
    return per_path
