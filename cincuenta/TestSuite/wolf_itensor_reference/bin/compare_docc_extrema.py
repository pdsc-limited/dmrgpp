#!/usr/bin/env python3
"""Compare trajectory extrema with Wolf Fig. 13(a)'s vector Lb=16 path.

The first dark-blue data path is extracted directly from the arXiv vector PDF
and identified as Lb=16 by the legend later in the same page content stream.
PDF coordinates are mapped through the exact plot bounds x=[0,13],
y=[0,0.05].

Extremum locations are estimated by a quadratic through a discrete local
extremum and its immediate neighbors. Wolf's retained neighboring vector
vertices are separated by 0.05 in time around every extremum considered here;
the trajectory used in the comparison is normally sampled at a finer dt.
These quadratic locations are useful resolution estimates, not unpublished
Wolf data.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from pypdf import PdfReader


@dataclass(frozen=True)
class Extremum:
    kind: str
    sample_time: float
    sample_value: float
    fitted_time: float
    fitted_value: float
    neighbor_span: float


def wolf_lb16_path(path: Path, maximum_time: float) -> np.ndarray:
    reader = PdfReader(path)
    if len(reader.pages) != 1:
        raise ValueError(f"expected a one-page standalone figure, got {len(reader.pages)}")
    text = reader.pages[0].get_contents().get_data().decode("latin-1")
    start = text.index("0 0 0.5450980392 RG")
    end = text.index("\nS\n", start)
    coordinates = [
        (float(x), float(y))
        for x, y in re.findall(
            r"([-+0-9.]+)\s+([-+0-9.]+)\s+[ml]", text[start:end]
        )
    ]
    # Exact axes in the decoded vector stream:
    # x PDF [93.6, 347.4] -> time [0, 13]
    # y PDF [72, 342] -> double occupancy [0, 0.05]
    values = np.array(sorted(
        (
            (x - 93.6) * 13.0 / 253.8,
            (y - 72.0) * 0.05 / 270.0,
        )
        for x, y in coordinates
    ))
    return values[(values[:, 0] >= 0) & (values[:, 0] <= maximum_time + 1e-10)]


def trajectory(path: Path, maximum_time: float) -> np.ndarray:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = np.array([
        (float(row["time"]), float(row["impurity_double_occupancy"]))
        for row in rows
    ])
    return values[values[:, 0] <= maximum_time + 1e-10]


def extrema(values: np.ndarray) -> list[Extremum]:
    result = []
    for index in range(1, len(values) - 1):
        before = values[index, 1] - values[index - 1, 1]
        after = values[index + 1, 1] - values[index, 1]
        if before * after >= 0:
            continue
        kind = "maximum" if before > 0 else "minimum"
        neighborhood = values[index - 1:index + 2]
        polynomial = np.polyfit(neighborhood[:, 0], neighborhood[:, 1], 2)
        fitted_time = -polynomial[1] / (2 * polynomial[0])
        fitted_value = np.polyval(polynomial, fitted_time)
        result.append(Extremum(
            kind,
            values[index, 0],
            values[index, 1],
            fitted_time,
            fitted_value,
            neighborhood[-1, 0] - neighborhood[0, 0],
        ))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wolf_pdf", type=Path)
    parser.add_argument("trajectory_csv", type=Path)
    parser.add_argument("--maximum-time", type=float, default=3.0)
    arguments = parser.parse_args()

    wolf = extrema(wolf_lb16_path(arguments.wolf_pdf, arguments.maximum_time))
    numerical = extrema(trajectory(arguments.trajectory_csv, arguments.maximum_time))
    if len(wolf) != len(numerical):
        raise ValueError(
            f"extremum count differs: Wolf={len(wolf)}, trajectory={len(numerical)}"
        )

    print("event      Wolf fitted t   numerical fitted t   delta t      delta d")
    time_differences = []
    value_differences = []
    for reference, result in zip(wolf, numerical):
        if reference.kind != result.kind:
            raise ValueError(f"extremum ordering differs: {reference.kind}, {result.kind}")
        delta_time = result.fitted_time - reference.fitted_time
        delta_value = result.fitted_value - reference.fitted_value
        time_differences.append(delta_time)
        value_differences.append(delta_value)
        print(
            f"{reference.kind:8s}  {reference.fitted_time:14.6f}  "
            f"{result.fitted_time:18.6f}  {delta_time:+10.6f}  {delta_value:+.3e}"
        )

    # One PDF coordinate point in each direction, based on the vector axes.
    time_per_pdf_point = 13.0 / 253.8
    docc_per_pdf_point = 0.05 / 270.0
    max_time = max(map(abs, time_differences))
    max_value = max(map(abs, value_differences))
    print(f"maximum |delta t| = {max_time:.12g} ({max_time / time_per_pdf_point:.3f} PDF points)")
    print(f"maximum |delta d| = {max_value:.12g} ({max_value / docc_per_pdf_point:.3f} PDF points)")
    print("published curve stroke width = 4.6 PDF points")


if __name__ == "__main__":
    main()
