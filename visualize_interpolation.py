"""Visualize a map's interpolated track with wall distances."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from interpolate_track import TrackInterpolator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", nargs="?", default="A03-Race", help="map folder name under Tracks/")
    args = parser.parse_args()

    map_dir = Path(__file__).parent / args.map
    out_png = map_dir / "interpolation.png"

    interp = TrackInterpolator(map_name=args.map)

    # Sample the interpolated track at high resolution
    t_values = np.linspace(0, len(interp.left_points) - 1, 500)
    left_path = [interp.get_position(t, "left") for t in t_values]
    right_path = [interp.get_position(t, "right") for t in t_values]

    # Extract coordinates for plotting
    left_x = [p.x for p in left_path if p]
    left_z = [p.z for p in left_path if p]
    right_x = [p.x for p in right_path if p]
    right_z = [p.z for p in right_path if p]

    # Original discrete points
    left_px = [p.x for p in interp.left_points]
    left_pz = [p.z for p in interp.left_points]
    right_px = [p.x for p in interp.right_points]
    right_pz = [p.z for p in interp.right_points]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot track boundaries with interpolation
    ax.plot(left_x, left_z, "r-", label="Left wall (interpolated)", alpha=0.7, linewidth=2)
    ax.plot(right_x, right_z, "b-", label="Right wall (interpolated)", alpha=0.7, linewidth=2)
    ax.scatter(left_px, left_pz, c="red", s=20, alpha=0.5, label="Left (discrete)")
    ax.scatter(right_px, right_pz, c="blue", s=20, alpha=0.5, label="Right (discrete)")

    ax.set_xlabel("X (world)")
    ax.set_ylabel("Z (world)")
    ax.set_title(f"{args.map} Track Boundaries (Interpolated)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
