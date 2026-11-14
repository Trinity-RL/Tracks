"""Write a map's start-to-finish route (route.csv) from a replay of it.

TMRC/agent/trace_walls.py drives along this route to trace the walls. Any
replay that finishes works; the game ships an author replay for each
campaign map, in GameData/Tracks/Campaigns/Nations/<colour>/<map>.Replay.gbx.

    python route_from_replay.py A01-Race [path/to/replay.Replay.Gbx]

Without a path it reads <map>/Author.Replay.Gbx.
"""
import argparse
from pathlib import Path

from pygbx import Gbx, GbxType


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("map", help="map folder name under Tracks/")
    parser.add_argument("replay", nargs="?", help="replay file (default <map>/Author.Replay.Gbx)")
    args = parser.parse_args()

    map_dir = Path(__file__).parent / args.map
    replay = Path(args.replay) if args.replay else map_dir / "Author.Replay.Gbx"
    ghost = Gbx(str(replay)).get_classes_by_ids([GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])[0]
    map_dir.mkdir(exist_ok=True)
    out = map_dir / "route.csv"
    with open(out, "w") as f:
        f.write("x,y,z\n")
        for record in ghost.records:
            p = record.position
            f.write(f"{p.x:.3f},{p.y:.3f},{p.z:.3f}\n")
    print(f"Wrote {len(ghost.records)} route points ({ghost.race_time} ms replay) to {out}")


if __name__ == "__main__":
    main()
