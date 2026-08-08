"""Compute track midpoints via nearest perpendicular projection from right wall to left wall."""
import argparse
import csv
from pathlib import Path

import numpy as np


def load_wall_points(csv_path, side):
    points = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["type"] == side:
                points.append((float(row["x"]), float(row["y"]), float(row["z"])))
    return points


def compute_midpoints(right_points, left_points):
    """For each right wall point, project perpendicularly onto the left wall polyline
    in the XZ plane, then return the midpoint between them."""
    right_xz = np.array([[p[0], p[2]] for p in right_points])
    right_y = np.array([p[1] for p in right_points])
    left_xz = np.array([[p[0], p[2]] for p in left_points])
    left_y = np.array([p[1] for p in left_points])

    # Precompute left wall segment vectors
    A = left_xz[:-1]
    B = left_xz[1:]
    AB = B - A
    ab_len_sq = np.sum(AB ** 2, axis=1)

    midpoints = []
    for rx_rz, ry in zip(right_xz, right_y):
        AR = rx_rz - A

        # Projection parameter clamped to [0, 1] along each segment
        valid = ab_len_sq > 1e-10
        t = np.where(valid, np.clip(np.einsum("ij,ij->i", AR, AB) / np.where(valid, ab_len_sq, 1.0), 0, 1), 0.0)

        proj = A + t[:, np.newaxis] * AB
        dist_sq = np.sum((rx_rz - proj) ** 2, axis=1)

        j = int(np.argmin(dist_sq))
        tj = t[j]
        proj_xz = proj[j]
        proj_y = left_y[j] + tj * (left_y[j + 1] - left_y[j])

        midpoints.append((
            (rx_rz[0] + proj_xz[0]) / 2,
            (ry + proj_y) / 2,
            (rx_rz[1] + proj_xz[1]) / 2,
        ))

    return midpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", nargs="?", default="A03-Race", help="map folder name under Tracks/")
    args = parser.parse_args()

    csv_path = Path(__file__).parent / args.map / "points-offset.csv"

    right_points = load_wall_points(csv_path, "right")
    left_points = load_wall_points(csv_path, "left")

    print(f"Loaded {len(right_points)} right wall points, {len(left_points)} left wall points")

    midpoints = compute_midpoints(right_points, left_points)
    print(f"Computed {len(midpoints)} midpoints")

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        for mx, my, mz in midpoints:
            writer.writerow([mx, my, mz, "mid"])

    print(f"Appended {len(midpoints)} mid rows to {csv_path.name}")


if __name__ == "__main__":
    main()
