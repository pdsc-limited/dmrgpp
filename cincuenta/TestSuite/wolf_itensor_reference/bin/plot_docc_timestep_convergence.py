#!/usr/bin/env python3
"""Plot causal-Cholesky double occupancy and its timestep difference.

Each input is a wolf_technical_trajectory output directory containing
trajectory.csv and metadata.toml. The script validates that the physical and
bath-construction settings agree before plotting. The finer trajectory is
linearly interpolated only for forming the lower-panel difference at the
coarser trajectory's recorded endpoint times; no interpolation is used for the
curves in the upper panel.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import tomllib

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Trajectory:
    directory: Path
    time: np.ndarray
    double_occupancy: np.ndarray
    metadata: dict

    @property
    def dt(self) -> float:
        return float(self.metadata["run"]["dt"])


def load_trajectory(directory: Path) -> Trajectory:
    trajectory_path = directory / "trajectory.csv"
    metadata_path = directory / "metadata.toml"
    with trajectory_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"time", "impurity_double_occupancy"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{trajectory_path} lacks columns {sorted(required)}")
    time = np.array([float(row["time"]) for row in rows])
    double_occupancy = np.array(
        [float(row["impurity_double_occupancy"]) for row in rows]
    )
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(double_occupancy)):
        raise ValueError(f"{trajectory_path} contains non-finite values")
    if not np.all(np.diff(time) > 0):
        raise ValueError(f"{trajectory_path} times are not strictly increasing")
    with metadata_path.open("rb") as stream:
        metadata = tomllib.load(stream)
    return Trajectory(directory, time, double_occupancy, metadata)


def common_setting(trajectories: list[Trajectory], section: str, key: str):
    values = [trajectory.metadata[section][key] for trajectory in trajectories]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"input mismatch for metadata [{section}].{key}: {values}")
    return values[0]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path, help="coarser output directory")
    parser.add_argument("fine", type=Path, help="finer output directory")
    parser.add_argument("--output", required=True, type=Path, help="output PDF or PNG")
    parser.add_argument("--title", default="Causal-Cholesky timestep comparison")
    parser.add_argument(
        "--ramp-duration", type=float, default=0.25,
        help="cosine-ramp duration t1 to record on the figure (default: 0.25)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    trajectories = [load_trajectory(arguments.coarse), load_trajectory(arguments.fine)]
    trajectories.sort(key=lambda trajectory: trajectory.dt, reverse=True)
    coarse, fine = trajectories
    if not fine.dt < coarse.dt:
        raise ValueError("fine timestep must be smaller than coarse timestep")

    interaction = common_setting(trajectories, "run", "U")
    bath_sites = common_setting(trajectories, "run", "Lb")
    rank = common_setting(trajectories, "factorization", "lesser_rank")
    factorization = common_setting(trajectories, "factorization", "method")
    midpoint = common_setting(trajectories, "factorization", "midpoint_interpolation")
    krylovdim = common_setting(trajectories, "global_krylov", "krylovdim")
    cutoff = common_setting(trajectories, "global_krylov", "cutoff")
    ramp_statuses = [
        trajectory.metadata["ramp_duration_status"] for trajectory in trajectories
    ]
    if ramp_statuses[0] != ramp_statuses[1]:
        raise ValueError(f"input mismatch for ramp status: {ramp_statuses}")
    ramp_status = ramp_statuses[0]
    t1 = arguments.ramp_duration
    if not np.isfinite(t1) or t1 <= 0:
        raise ValueError("--ramp-duration must be positive and finite")
    common_end = min(coarse.time[-1], fine.time[-1])
    coarse_mask = coarse.time <= common_end + 1e-12
    comparison_time = coarse.time[coarse_mask]
    difference = np.interp(
        comparison_time, fine.time, fine.double_occupancy
    ) - coarse.double_occupancy[coarse_mask]

    figure, (axis, difference_axis) = plt.subplots(
        2, 1, figsize=(8.0, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
        constrained_layout=False,
    )
    colors = ("#1f77b4", "#d62728")
    for trajectory, color in zip((coarse, fine), colors):
        axis.plot(
            trajectory.time,
            trajectory.double_occupancy,
            color=color,
            linewidth=1.7,
            label=rf"$\Delta t={trajectory.dt:g}$",
        )
    axis.set_ylabel(r"double occupancy $d(t)$")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title(arguments.title)

    difference_axis.axhline(0.0, color="0.35", linewidth=0.8)
    difference_axis.plot(comparison_time, difference, color="#2ca02c", linewidth=1.4)
    difference_axis.set_xlabel(r"time $t$ ($v_0^{-1}$)")
    difference_axis.set_ylabel(r"$d_{\rm fine}-d_{\rm coarse}$")
    difference_axis.grid(alpha=0.25)
    difference_axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    provenance = (
        rf"$U={interaction:g}$, $L_b={bath_sites}$ (rank {rank}+{rank}), "
        rf"$t_1={t1:g}$ provisional; Krylov {krylovdim}, cutoff ${cutoff:.0e}$" "\n"
        f"{factorization}; {midpoint}"
    )
    figure.text(0.5, 0.005, provenance, ha="center", va="bottom", fontsize=8)
    figure.subplots_adjust(left=0.12, right=0.98, top=0.91, bottom=0.17)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180)
    print(f"wrote {arguments.output}")
    print(f"maximum |fine-coarse| at coarse endpoints = {np.max(np.abs(difference)):.12g}")
    print(f"ramp status: {ramp_status}")


if __name__ == "__main__":
    main()
