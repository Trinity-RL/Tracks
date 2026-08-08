"""Offset boundary points outward by half car width using replay direction data."""
import argparse
import csv
from pathlib import Path

from pygbx import Gbx, GbxType

# TrackMania standard car width (adjust if needed)
CAR_WIDTH = 1.0


def load_ghost_records(path):
    gbx = Gbx(str(path))
    ghost = gbx.get_classes_by_ids([GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])[0]
    return ghost.records


def vec_subtract(p1, p2):
    """Subtract two position vectors."""
    return (p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)


def vec_length(v):
    """Length of a vector."""
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


def vec_normalize(v):
    """Normalize a vector."""
    length = vec_length(v)
    if length == 0:
        return v
    return (v[0] / length, v[1] / length, v[2] / length)


def vec_scale(v, s):
    """Scale a vector."""
    return (v[0] * s, v[1] * s, v[2] * s)


def vec_cross(v1, v2):
    """Cross product of two 3D vectors."""
    return (
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    )


def find_direction_at_index(records, idx):
    """Get forward direction at record index using neighboring points."""
    if idx < 1 or idx > len(records) - 2:
        return None

    prev_pos = records[idx - 1].position
    next_pos = records[idx + 1].position

    direction = vec_subtract(next_pos, prev_pos)
    normalized = vec_normalize(direction)
    return normalized if vec_length(normalized) > 0.01 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", nargs="?", default="A03-Race", help="map folder name under Tracks/")
    args = parser.parse_args()

    map_dir = Path(__file__).parent / args.map
    left_replay = map_dir / "Left.Replay.Gbx"
    right_replay = map_dir / "Right.Replay.Gbx"
    in_csv = map_dir / "points.csv"
    out_csv = map_dir / "points-offset.csv"

    # Load left replay to get direction data
    left_records = load_ghost_records(left_replay)
    right_records = load_ghost_records(right_replay)

    print(f"Left replay: {len(left_records)} records")
    print(f"Right replay: {len(right_records)} records")

    # Build maps of position -> record index for quick lookup
    left_pos_to_idx = {}
    right_pos_to_idx = {}

    for i, rec in enumerate(left_records):
        key = (round(rec.position.x, 2), round(rec.position.z, 2))
        left_pos_to_idx[key] = i

    for i, rec in enumerate(right_records):
        key = (round(rec.position.x, 2), round(rec.position.z, 2))
        right_pos_to_idx[key] = i

    # Read and offset points
    rows = []
    with open(in_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
            point_type = row["type"]

            if point_type == "left":
                # Find the record index for this point
                key = (round(x, 2), round(z, 2))
                idx = left_pos_to_idx.get(key)

                if idx is not None:
                    direction = find_direction_at_index(left_records, idx)
                    if direction:
                        # Get perpendicular direction (left side of car)
                        # In XZ plane, perpendicular to (dx, dz) is (-dz, dx) [left]
                        perp = vec_normalize((-direction[2], 0, direction[0]))
                        offset = vec_scale(perp, CAR_WIDTH / 2)
                        x += offset[0]
                        z += offset[2]

            elif point_type == "right":
                # Find the record index for this point
                key = (round(x, 2), round(z, 2))
                idx = right_pos_to_idx.get(key)

                if idx is not None:
                    direction = find_direction_at_index(right_records, idx)
                    if direction:
                        # Get perpendicular direction (right side of car)
                        # In XZ plane, perpendicular to (dx, dz) is (dz, -dx) [right]
                        perp = vec_normalize((direction[2], 0, -direction[0]))
                        offset = vec_scale(perp, CAR_WIDTH / 2)
                        x += offset[0]
                        z += offset[2]

            rows.append((x, y, z, point_type))

    # Write offset points
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "type"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} offset points to {out_csv}")


if __name__ == "__main__":
    main()
