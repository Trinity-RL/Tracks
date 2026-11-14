# Tracks

Wall and centre-line data for each TMRC map, and the scripts that make it.
The TMRC framework (`TMRC/agent/track_features.py`) and the competition harness
cast the car's rays against these walls and measure progress along the centre line.

```
<map>/
  points-offset.csv   the data everything reads: x,y,z,type with type
                      left / right (walls, start to finish) and mid (centre line)
  points.csv          wall points before the half-car-width offset
  route.csv           a start-to-finish driven line (the game's author replay)
  topdown.png         plot of points-offset.csv: check it after regenerating
interpolate_track.py  TrackInterpolator: wall interpolation and ray casting
```

Maps: **A01-Race**, **A02-Race**, **A03-Race**.

## Making a map's data

Python with `numpy scipy matplotlib pygbx`.

### From two hand-driven replays (how A03-Race was made)

Drive one lap hugging the left wall and one hugging the right, save them as
`<map>/Left.Replay.Gbx` and `<map>/Right.Replay.Gbx`, then:

```bash
python extract_points.py <map>        # replays -> points.csv
python offset_boundaries.py <map>     # -> points-offset.csv (walls)
python compute_midpoints.py <map>     # appends the centre line
python visualize_points.py <map>      # -> topdown.png
```

### Traced by the game (how A01-Race was made)

`TMRC/agent/trace_walls.py` drives both laps itself. It follows a route from
start to finish, aimed well to one side so the car rubs along that wall, and
records where it touched. It rewinds past gaps and jumps, which have no walls.

```bash
# 1. a route: any finishing replay; the game ships an author replay for each
#    campaign map (GameData/Tracks/Campaigns/Nations/White/<map>.Replay.gbx)
python route_from_replay.py <map> path/to/<map>.Replay.Gbx
# 2. drive along both walls (from TMNF-Docker/; takes a few minutes)
TM_MODE=script TM_SCRIPT=agent/trace_walls.py TM_MAP_FILE=<map>.Challenge.Gbx \
    TM_NUM_INSTANCES=1 docker compose up --build
# 3. traces -> points.csv, points-offset.csv (walls + centre line)
python traces_to_points.py <map> ../TMNF-Docker/TMRC/results/traces --end-m <finish>
python visualize_points.py <map>
```

`--end-m` is the route distance of the finish line (the trace log prints it
when the car finishes); the route is a replay, which keeps going after it.
A01-Race: `--end-m 2360`.

### Maps without corridor walls (A02-Race)

A02-Race is open dirt fields: nothing borders the course except the stadium,
hundreds of metres away. Its walls are **virtual**, a corridor either side of
the author's line, so rays still guide a driver along the course:

```bash
python traces_to_points.py A02-Race --corridor 12 --end-m 841
```

Where a traced map has a section with no wall to touch (a jump, a bridge
without barriers, the start lane), `traces_to_points.py` fills it at the wall's
distance from the route, interpolated between the gap's ends (10 m before the
first and after the last traced point). Hand-made maps have no such filling:
there the interpolated wall is a straight line across any gap.

After changing a map's data, re-sync the competition copy:
`competition-admin/scripts/sync_tracks.sh`.
