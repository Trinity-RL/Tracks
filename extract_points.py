"""Extract labeled track points (left/right boundary + driven blocks) from a map's replays."""
import argparse
import csv
from pathlib import Path

from pygbx import Gbx, GbxType

BLOCK_SIZE_XZ = 32
BLOCK_SIZE_Y = 8


def load_ghost_records(path):
    gbx = Gbx(str(path))
    ghost = gbx.get_classes_by_ids([GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])[0]
    return ghost.records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", nargs="?", default="A03-Race", help="map folder name under Tracks/")
    args = parser.parse_args()

    map_dir = Path(__file__).parent / args.map
    left_replay = map_dir / "Left.Replay.Gbx"
    right_replay = map_dir / "Right.Replay.Gbx"
    out_csv = map_dir / "points.csv"

    rows = []
    block_cells = set()

    for path, label in [(left_replay, "left"), (right_replay, "right")]:
        for record in load_ghost_records(path):
            p = record.position
            rows.append((p.x, p.y, p.z, label))
            bp = record.get_block_position()
            block_cells.add((bp.x, bp.y, bp.z))

    # Convert driven block grid coordinates back into world (car) coordinates,
    # using the center of each block cell.
    for bx, by, bz in sorted(block_cells):
        wx = (bx + 0.5) * BLOCK_SIZE_XZ
        wy = (by + 0.5) * BLOCK_SIZE_Y
        wz = (bz + 0.5) * BLOCK_SIZE_XZ
        rows.append((wx, wy, wz, "block"))

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "type"])
        writer.writerows(rows)

    n_left = sum(1 for r in rows if r[3] == "left")
    n_right = sum(1 for r in rows if r[3] == "right")
    print(f"Wrote {len(rows)} points to {out_csv}")
    print(f"  left boundary points:  {n_left}")
    print(f"  right boundary points: {n_right}")
    print(f"  driven block cells:    {len(block_cells)}")


if __name__ == "__main__":
    main()
