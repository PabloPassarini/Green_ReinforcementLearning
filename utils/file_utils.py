from pathlib import Path
from typing import List
import pandas as pd


def save_per_episode(results_dir: Path, base_name: str, distances: List[float]) -> Path:
    """Save per-episode CSV with only Episode and Distance columns."""
    df = pd.DataFrame({"Episode": list(range(len(distances))), "Distance": distances})
    out_path = results_dir / f"{base_name}_results.csv"
    df.to_csv(out_path, index=False)
    return out_path


def save_summary(
    results_dir: Path,
    base_name: str,
    instance: str,
    r_type: str,
    e_type: str,
    gamma: float,
    best_episode: int,
    best_distance: float,
    best_path: List[int],
    emissions_data,
) -> Path:
    """Save a one-row summary CSV including experiment metadata and emissions metrics."""
    row = {
        "Instance": instance,
        "RewardType": r_type,
        "EpsilonDecay": e_type,
        "Gamma": gamma,
        "BestEpisode": best_episode,
        "BestDistance": best_distance,
        "BestPath": " -> ".join(map(str, best_path)),
        "Duration_sec": getattr(emissions_data, "duration", None),
        "Emissions_kgCO2": getattr(emissions_data, "emissions", None),
        "EmissionsRate_kgCO2_per_sec": getattr(emissions_data, "emissions_rate", None),
        "CPU_Power_W": getattr(emissions_data, "cpu_power", None),
        "GPU_Power_W": getattr(emissions_data, "gpu_power", None),
        "RAM_Power_W": getattr(emissions_data, "ram_power", None),
        "CPU_Energy_Wh": getattr(emissions_data, "cpu_energy", None),
        "GPU_Energy_Wh": getattr(emissions_data, "gpu_energy", None),
        "RAM_Energy_Wh": getattr(emissions_data, "ram_energy", None),
        "Total_Energy_Wh": getattr(emissions_data, "energy_consumed", None),
        "Water_Consumed_L": getattr(emissions_data, "water_consumed", None),
    }
    df = pd.DataFrame([row])
    out_path = results_dir / f"{base_name}_summary.csv"
    df.to_csv(out_path, index=False)
    return out_path
