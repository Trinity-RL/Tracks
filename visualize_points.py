"""Top-down visualization of a map's boundary and block points."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"left": "red", "right": "red", "block": "green", "mid": "blue"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", nargs="?", default="A03-Race", help="map folder name under Tracks/")
    args = parser.parse_args()

    map_dir = Path(__file__).parent / args.map
    in_csv = map_dir / "points-offset.csv"
    out_png = map_dir / "topdown.png"

    points = {"left": [], "right": [], "block": [], "mid": []}
    with open(in_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row["type"]
            if t in points:
                points[t].append((float(row["x"]), float(row["z"])))

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw blocks first so boundary dots stay visible on top.
    for label in ("block", "left", "right", "mid"):
        xs, zs = zip(*points[label]) if points[label] else ([], [])
        ax.scatter(xs, zs, s=12 if label == "block" else 6,
                   c=COLORS[label], label=label)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title(f"{args.map} top-down (X/Z plane)")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
