"""Build a map's points-offset.csv from wall traces driven by trace_walls.py.

TMRC/agent/trace_walls.py drives along each wall and records every 10 ms step,
flagging the steps where the car touched a wall. Those positions are the wall
points. This keeps the ones on the expected side of the route (a touch on the
opposite wall in a narrow section is dropped), orders them start to finish,
thins them to one per POINT_SPACING_M of route, shifts them by WALL_SHIFT_M
(see below). Gaps (no wall to touch: a start pocket, a jump, a bridge without
barriers) are filled at the wall's distance from the route, interpolated
between the gap's ends. The centre line is the route itself, up to the finish.

A map that isn't a walled corridor (A02-Race: open dirt fields) gets virtual
walls a fixed distance either side of the route instead (--corridor).

    python traces_to_points.py A01-Race path/to/traces/ --end-m 2360
    python traces_to_points.py A02-Race --corridor 12 --end-m 841

reads <traces>/<map>-left.csv and -right.csv and <map>/route.csv, writes
<map>/points.csv (before the offset) and <map>/points-offset.csv.
Check <map>/topdown.png (visualize_points.py) afterwards.
"""
import argparse
import csv
from pathlib import Path

import numpy as np


# Outward shift from the car's position when touching the wall to the wall
# line written out. Calibrated so traced walls match A03-Race's hand-made ones
# (offset_boundaries.py's +0.5 m on a human lap that didn't scrape the wall the
# whole way): tracing A03 itself and shifting by this lands a median 0.13 m
# from its walls. Matching A03 means a driver sees the same distances on every map.
WALL_SHIFT_M = -0.45
POINT_SPACING_M = 2.0    # A03's replay points are ~2.3 m apart
MAX_LATERAL_M = 80.0     # further than this from the route isn't this track's wall
MIN_LATERAL_M = 4.0      # closer than this is a post or pillar the route passes, not a wall


def load_route(path):
    xz = np.loadtxt(path, delimiter=",", skiprows=1)[:, [0, 2]]
    keep = np.concatenate([[True], np.linalg.norm(np.diff(xz, axis=0), axis=1) > 0.05])
    return xz[keep]


def project(route, p):
    """(arc length, signed lateral offset) of p against the route. Positive =
    to the left of the direction of travel in x-z (cross product > 0)."""
    a, b = route[:-1], route[1:]
    ab = b - a
    seg_len = np.linalg.norm(ab, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    t = np.clip(np.sum((p - a) * ab, axis=1) / np.maximum(seg_len ** 2, 1e-9), 0, 1)
    proj = a + t[:, None] * ab
    i = int(np.argmin(np.sum((proj - p) ** 2, axis=1)))
    tangent = ab[i] / max(seg_len[i], 1e-9)
    rel = p - proj[i]
    cross = tangent[0] * rel[1] - tangent[1] * rel[0]
    return cum[i] + t[i] * seg_len[i], float(cross), tangent


def wall_points(trace_csv, route):
    trace_csv = Path(trace_csv)
    with open(trace_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["contact"] == "1"]
    pts = []
    for r in rows:
        p = np.array([float(r["x"]), float(r["z"])])
        s, lateral, tangent = project(route, p)
        if MIN_LATERAL_M <= abs(lateral) <= MAX_LATERAL_M:
            pts.append((s, lateral, float(r["x"]), float(r["y"]), float(r["z"]), tangent))
    if not pts:
        raise ValueError(f"{trace_csv}: no wall contact recorded")
    # The wall is on whichever side most contacts were.
    side_sign = 1.0 if sum(np.sign(p[1]) for p in pts) > 0 else -1.0
    kept = [p for p in pts if np.sign(p[1]) == side_sign]
    print(f"  {trace_csv.name}: {len(rows)} contact steps, {len(pts) - len(kept)} on the wrong side dropped")

    kept.sort(key=lambda p: p[0])
    thinned, bin_start, bucket = [], None, []
    for p in kept:
        if bin_start is None or p[0] - bin_start >= POINT_SPACING_M:
            if bucket:
                thinned.append(bucket)
            bin_start, bucket = p[0], []
        bucket.append(p)
    if bucket:
        thinned.append(bucket)

    raw, offset = [], []
    for bucket in thinned:
        x = float(np.mean([p[2] for p in bucket]))
        y = float(np.mean([p[3] for p in bucket]))
        z = float(np.mean([p[4] for p in bucket]))
        tx, tz = bucket[len(bucket) // 2][5]
        # Outward normal: the side the wall is on.
        nx, nz = -tz * side_sign, tx * side_sign
        raw.append((x, y, z))
        offset.append((x + nx * WALL_SHIFT_M, y, z + nz * WALL_SHIFT_M))
    return raw, offset, side_sign


# The left wall is on the negative side of the route (cross product < 0), as
# tracing A03 shows: the game's x-z axes make "left" a clockwise turn there.
LEFT_SIGN = -1.0
MID_SPACING_M = 2.5      # A03's centre-line points are ~2.3 m apart


GAP_M = 6.0              # wall points further apart than this along the route: fill the gap
# Before the first and after the last traced point (start and finish lanes, which
# the tracer can't reach while it's getting going or taking the final jump) a
# wall's distance from the route can't be interpolated. Stadium road blocks are
# about 20 m wide: on A01 the start lane's walls are ~10 m either side.
ROAD_HALF_WIDTH_M = 10.0


def fill_gaps(points, route_xyz, side_sign, end_m):
    """Fill stretches with no traced wall (start pockets, jumps, bridges) with
    points at the wall's distance from the route, interpolated between the
    gap's two ends. Interpolating the wall itself straight across a gap can
    cut across the road; following the route can't."""
    route = load_route_xyz(route_xyz)
    xz = route_xyz[:, [0, 2]]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))])
    end_m = min(end_m, cum[-1]) if end_m else cum[-1]
    traced = []
    for p in points:
        s_m, lateral, _ = project(route, np.array([p[0], p[2]]))
        traced.append((s_m, abs(lateral), p))
    traced.sort(key=lambda t: t[0])
    ts = np.array([t[0] for t in traced])
    widths = np.array([t[1] for t in traced])
    merged = [(t[0], t[2]) for t in traced]
    filled = 0
    for s_m in np.arange(0.0, end_m + 1e-9, POINT_SPACING_M):
        if np.min(np.abs(ts - s_m)) <= GAP_M:
            continue
        i = min(int(np.searchsorted(cum, s_m, side="right")) - 1, len(cum) - 2)
        t = (s_m - cum[i]) / max(cum[i + 1] - cum[i], 1e-9)
        cx, cy, cz = route_xyz[i] + t * (route_xyz[i + 1] - route_xyz[i])
        tx, tz = (xz[i + 1] - xz[i]) / max(cum[i + 1] - cum[i], 1e-9)
        if s_m < ts[0] or s_m > ts[-1]:
            width = ROAD_HALF_WIDTH_M
        else:
            width = float(np.interp(s_m, ts, widths))
        merged.append((s_m, (cx - tz * side_sign * width, cy, cz + tx * side_sign * width)))
        filled += 1
    merged.sort(key=lambda m: m[0])
    return [m[1] for m in merged], filled


def load_route_xyz(route_xyz):
    xz = route_xyz[:, [0, 2]]
    keep = np.concatenate([[True], np.linalg.norm(np.diff(xz, axis=0), axis=1) > 0.05])
    return xz[keep]


def route_centre_line(route_xyz, end_m):
    """The route resampled every MID_SPACING_M up to end_m (the finish)."""
    xz = route_xyz[:, [0, 2]]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))])
    end_m = min(end_m, cum[-1]) if end_m else cum[-1]
    s = np.append(np.arange(0.0, end_m, MID_SPACING_M), end_m)
    return [(float(np.interp(v, cum, route_xyz[:, 0])), float(np.interp(v, cum, route_xyz[:, 1])),
             float(np.interp(v, cum, route_xyz[:, 2]))) for v in s]


def corridor(centre, half_width_m):
    """Virtual walls half_width_m either side of a centre line."""
    pts = np.array(centre)
    xz = pts[:, [0, 2]]
    tangent = np.gradient(xz, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])  # cross > 0 side
    walls = {}
    for side, sign in (("left", LEFT_SIGN), ("right", -LEFT_SIGN)):
        off = xz + normal * sign * half_width_m
        walls[side] = [(float(x), float(y), float(z)) for (x, z), y in zip(off, pts[:, 1])]
    return walls


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("map", help="map folder name under Tracks/")
    parser.add_argument("traces", nargs="?", help="folder with <map>-left.csv and <map>-right.csv")
    parser.add_argument("--end-m", type=float, default=None,
                        help="route distance of the finish line: the centre line stops there "
                             "(the route is a replay, which keeps going after the finish)")
    parser.add_argument("--corridor", type=float, default=None, metavar="HALF_WIDTH_M",
                        help="no traced walls: write virtual walls this far either side of the route, "
                             "for maps that aren't a walled corridor (A02's open dirt fields)")
    args = parser.parse_args()

    map_dir = Path(__file__).parent / args.map
    route_xyz = np.loadtxt(map_dir / "route.csv", delimiter=",", skiprows=1)
    route = load_route(map_dir / "route.csv")
    # The centre line is the route itself: unlike midpoints between traced
    # walls, it has no gaps where a section has no wall (jumps, open ground).
    mids = route_centre_line(route_xyz, args.end_m)

    if args.corridor is not None:
        walls = {side: (pts, pts, None) for side, pts in corridor(mids, args.corridor).items()}
    else:
        if not args.traces:
            raise SystemExit("give the traces folder, or --corridor")
        traces = Path(args.traces)
        walls = {side: wall_points(traces / f"{args.map}-{side}.csv", route) for side in ("left", "right")}
        for side, sign in (("left", LEFT_SIGN), ("right", -LEFT_SIGN)):
            if walls[side][2] != sign:
                raise ValueError(f"the {side} trace touched the wrong side of the route: check it")
            offset, filled = fill_gaps(walls[side][1], route_xyz, sign, args.end_m)
            print(f"  {side}: {len(walls[side][1])} traced points, {filled} filled in gaps")
            walls[side] = (walls[side][0], offset, sign)

    with open(map_dir / "points.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "type"])
        for side in ("left", "right"):
            w.writerows((*p, side) for p in walls[side][0])

    out = map_dir / "points-offset.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "type"])
        for side in ("left", "right"):
            w.writerows((*p, side) for p in walls[side][1])
        w.writerows((*p, "mid") for p in mids)

    mid_xz = np.array([[p[0], p[2]] for p in mids])
    length = float(np.sum(np.linalg.norm(np.diff(mid_xz, axis=0), axis=1)))
    kind = f"virtual walls {args.corridor:g} m either side" if args.corridor is not None else "traced walls"
    print(f"Wrote {out}: {len(walls['left'][1])} left, {len(walls['right'][1])} right ({kind}), "
          f"{len(mids)} mid points; centre line {length:.0f} m")


if __name__ == "__main__":
    main()
