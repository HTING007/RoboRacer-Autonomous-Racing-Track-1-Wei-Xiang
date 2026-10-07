#!/usr/bin/env python3
"""Validate the hackathon track and derive a centreline from its map.

Standard library plus PyYAML only, so it runs on the host as well as inside the
container.

    # Check maps/tracks.yaml against the map image
    ./scripts/track_tool.py validate

    # Write a centreline CSV that pure_pursuit can follow
    ./scripts/track_tool.py centerline

`validate` is the important one. It renders the start pose and the finish line
onto the occupancy grid and fails if either is inside a wall, if the finish line
does not span the corridor, or if no closed lap exists through the line at all.
A finish line that stops short of a wall silently misses laps, and one that
pokes through a wall counts phantom ones; neither is something to discover
while a team is being scored.
"""

from __future__ import annotations

import argparse
import heapq
import math
import os
import struct
import sys
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "maps", "tracks.yaml")

sys.path.insert(0, os.path.join(REPO_ROOT, "race_ws", "src", "roboracer_referee"))

INF = float("inf")
Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Image readers
#
# Map images are 8-bit greyscale PGM or PNG. Reading them here rather than via
# Pillow keeps this runnable with a bare Python on any machine.
# ---------------------------------------------------------------------------

def read_image_gray(path: str) -> Tuple[int, int, bytearray]:
    with open(path, "rb") as handle:
        head = handle.read(8)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return _read_png_gray(path)
    if head[:2] in (b"P5", b"P2"):
        return _read_pgm_gray(path)
    raise ValueError(f"{path}: not a PNG or PGM image")


def _read_pgm_gray(path: str) -> Tuple[int, int, bytearray]:
    data = open(path, "rb").read()
    magic = data[:2]
    pos, fields = 2, []
    while len(fields) < 3:
        while pos < len(data) and data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":                       # comment to end of line
            while pos < len(data) and data[pos:pos + 1] != b"\n":
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos:pos + 1].isspace():
            pos += 1
        fields.append(int(data[start:pos]))
    width, height, maxval = fields
    if maxval > 255:
        raise ValueError(f"{path}: only 8-bit PGM is supported (maxval {maxval})")

    if magic == b"P5":
        pos += 1                                            # exactly one whitespace byte
        pixels = bytearray(data[pos:pos + width * height])
    else:                                                   # P2, ASCII
        values = data[pos:].split()
        pixels = bytearray(int(v) for v in values[:width * height])
    if len(pixels) != width * height:
        raise ValueError(f"{path}: expected {width * height} pixels, found {len(pixels)}")
    return width, height, pixels


def _read_png_gray(path: str) -> Tuple[int, int, bytearray]:
    data = open(path, "rb").read()
    pos, idat = 8, bytearray()
    width = height = depth = color = 0
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if ctype == b"IHDR":
            width, height, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if depth != 8:
                raise ValueError(f"{path}: only 8-bit PNGs are supported (got {depth}-bit)")
            if interlace:
                raise ValueError(f"{path}: interlaced PNGs are not supported")
            if color not in (0, 2, 4, 6):
                raise ValueError(f"{path}: unsupported PNG colour type {color}")
        elif ctype == b"IDAT":
            idat += body
        elif ctype == b"IEND":
            break
        pos += 12 + length

    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color]
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    out = bytearray(width * height)
    prev = bytearray(stride)
    offset = 0
    for row in range(height):
        filter_type = raw[offset]
        offset += 1
        line = bytearray(raw[offset:offset + stride])
        offset += stride
        _unfilter(filter_type, line, prev, channels)
        for col in range(width):
            out[row * width + col] = line[col * channels]
        prev = line
    return width, height, out


def _unfilter(filter_type: int, line: bytearray, prev: bytearray, bpp: int) -> None:
    if filter_type == 0:
        return
    for i in range(len(line)):
        a = line[i - bpp] if i >= bpp else 0
        b = prev[i]
        c = prev[i - bpp] if i >= bpp else 0
        if filter_type == 1:
            line[i] = (line[i] + a) & 0xFF
        elif filter_type == 2:
            line[i] = (line[i] + b) & 0xFF
        elif filter_type == 3:
            line[i] = (line[i] + (a + b) // 2) & 0xFF
        elif filter_type == 4:
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            line[i] = (line[i] + pred) & 0xFF
        else:
            raise ValueError(f"unknown PNG filter type {filter_type}")


# ---------------------------------------------------------------------------
# Occupancy grid
# ---------------------------------------------------------------------------

class OccupancyMap:
    """A map_server style grid, queried in world coordinates.

    Occupancy follows the same rule the simulator uses: probability is
    ``(255 - pixel) / 255`` (or ``pixel / 255`` when ``negate``), and a cell is
    an obstacle when that exceeds ``occupied_thresh``. The ROS "unknown" band
    counts as free. Validating against anything else would be checking a map
    the simulator is not using.
    """

    def __init__(self, yaml_path: str):
        with open(yaml_path, "r") as handle:
            meta = yaml.safe_load(handle) or {}
        image = meta.get("image")
        if not image:
            raise ValueError(f"{yaml_path}: missing 'image'")
        if not os.path.isabs(image):
            image = os.path.join(os.path.dirname(yaml_path), image)
        if not os.path.isfile(image):
            raise ValueError(
                f"{yaml_path}: 'image: {meta['image']}' does not exist. "
                f"Expected {image}."
            )
        self.yaml_path = yaml_path
        self.width, self.height, self.pixels = read_image_gray(image)
        self.resolution = float(meta["resolution"])
        self.origin = [float(v) for v in meta["origin"]]
        self.negate = bool(int(meta.get("negate", 0)))
        self.occupied_thresh = float(meta.get("occupied_thresh", 0.65))
        self._clearance: Optional[List[float]] = None

    def _is_obstacle(self, value: int) -> bool:
        occupancy = value / 255.0 if self.negate else (255 - value) / 255.0
        return occupancy > self.occupied_thresh

    # -- basic queries --------------------------------------------------

    def to_cell(self, x: float, y: float) -> Tuple[int, int]:
        col = int((x - self.origin[0]) / self.resolution)
        # Image rows run top-down while the map frame runs bottom-up.
        row = self.height - 1 - int((y - self.origin[1]) / self.resolution)
        return row, col

    def to_world(self, row: int, col: int) -> Point:
        return (self.origin[0] + col * self.resolution,
                self.origin[1] + (self.height - 1 - row) * self.resolution)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_free(self, x: float, y: float) -> bool:
        row, col = self.to_cell(x, y)
        if not self.in_bounds(row, col):
            return False
        return not self._is_obstacle(self.pixels[row * self.width + col])

    def segment_is_clear(self, a: Point, b: Point) -> Tuple[bool, Optional[Point]]:
        step = self.resolution / 2.0
        samples = max(2, int(math.dist(a, b) / step) + 1)
        for i in range(samples + 1):
            t = i / samples
            point = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if not self.is_free(*point):
                return False, point
        return True, None

    def free_distance(self, x: float, y: float, dx: float, dy: float,
                      limit: float = 40.0) -> float:
        """March from (x, y) along a unit direction until the first occupied cell."""
        step = self.resolution / 2.0
        travelled = 0.0
        while travelled < limit:
            travelled += step
            if not self.is_free(x + dx * travelled, y + dy * travelled):
                return travelled - step
        return limit

    # -- clearance ------------------------------------------------------

    # -- cones ----------------------------------------------------------

    def cone_components(self, max_span: float = 0.9,
                        max_cells: int = 400) -> List[Tuple[float, float]]:
        """Centres of the small free-standing obstacles, i.e. the cones.

        Anything occupied, isolated, and smaller than max_span across counts.
        The track walls are one huge component and never qualify.
        """
        from collections import deque                              # noqa: PLC0415

        w, h = self.width, self.height
        occupied = [self._is_obstacle(v) for v in self.pixels]
        seen = bytearray(w * h)
        centres = []
        for start in range(w * h):
            if not occupied[start] or seen[start]:
                continue
            queue = deque([start])
            seen[start] = 1
            cells = []
            while queue:
                i = queue.popleft()
                cells.append(i)
                row, col = divmod(i, w)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = row + dr, col + dc
                        if self.in_bounds(rr, cc):
                            j = rr * w + cc
                            if occupied[j] and not seen[j]:
                                seen[j] = 1
                                queue.append(j)
            if len(cells) > max_cells:
                continue
            rows = [i // w for i in cells]
            cols = [i % w for i in cells]
            span = max(max(rows) - min(rows) + 1, max(cols) - min(cols) + 1) * self.resolution
            if not 0.1 < span < max_span:
                continue
            centres.append(self.to_world((min(rows) + max(rows)) // 2,
                                         (min(cols) + max(cols)) // 2))
        return centres

    def link_cones(self, max_distance: float = 1.1) -> int:
        """Join neighbouring cones into solid barriers, and return how many.

        A line of cones is a wall in every sense that matters to a race, but to
        an occupancy grid it is a row of small islands with drivable gaps
        between them. A shortest-path search will happily thread one, and a car
        following that line is cutting the course. Filling the gaps in makes
        threading geometrically impossible rather than merely against the rules.

        max_distance must sit between the spacing within a row and the width of
        the lane between rows, or the lanes get sealed too. On this circuit
        those are about 0.95 m and 1.9 m.
        """
        centres = self.cone_components()
        links = 0
        for i, a in enumerate(centres):
            for b in centres[i + 1:]:
                if math.dist(a, b) <= max_distance:
                    self._draw_barrier(a, b)
                    links += 1
        if links:
            self._clearance = None          # the geometry changed
        return links

    def _draw_barrier(self, a: Point, b: Point) -> None:
        steps = max(2, int(math.dist(a, b) / (self.resolution / 2.0)))
        for i in range(steps + 1):
            t = i / steps
            row, col = self.to_cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if self.in_bounds(row + dr, col + dc):
                        self.pixels[(row + dr) * self.width + col + dc] = 0

    def clearance(self) -> List[float]:
        """Distance in metres from each cell to the nearest obstacle."""
        if self._clearance is None:
            self._clearance = _euclidean_distance_transform(
                [not self._is_obstacle(self.pixels[i])
                 for i in range(self.width * self.height)],
                self.width, self.height,
            )
            self._clearance = [d * self.resolution for d in self._clearance]
        return self._clearance

    def clearance_at(self, x: float, y: float) -> float:
        row, col = self.to_cell(x, y)
        if not self.in_bounds(row, col):
            return 0.0
        return self.clearance()[row * self.width + col]


def _euclidean_distance_transform(free: Sequence[bool], w: int, h: int) -> List[float]:
    """Exact EDT (Felzenszwalb & Huttenlocher), distance to the nearest False cell."""
    f = [0.0 if not free[i] else INF for i in range(w * h)]
    for c in range(w):
        d = _dt_1d([f[r * w + c] for r in range(h)])
        for r in range(h):
            f[r * w + c] = d[r]
    for r in range(h):
        f[r * w:(r + 1) * w] = _dt_1d(f[r * w:(r + 1) * w])
    return [x ** 0.5 for x in f]


def _dt_1d(f: Sequence[float]) -> List[float]:
    n = len(f)
    v = [0] * n
    z = [0.0] * (n + 1)
    k = 0
    z[0], z[1] = -INF, INF
    for q in range(1, n):
        while True:
            if f[v[k]] == INF and f[q] == INF:
                s = INF
            elif f[v[k]] == INF:
                s = -INF
            elif f[q] == INF:
                s = INF
            else:
                s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
            if s <= z[k] and k > 0:
                k -= 1
                continue
            break
        k += 1
        v[k], z[k], z[k + 1] = q, s, INF
    out = [0.0] * n
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        out[q] = (q - v[k]) ** 2 + f[v[k]] if f[v[k]] != INF else INF
    return out


# ---------------------------------------------------------------------------
# Lap finding
# ---------------------------------------------------------------------------

NEIGHBOURS = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, 1.41421356), (-1, 1, 1.41421356),
              (1, -1, 1.41421356), (1, 1, 1.41421356))


def find_lap(grid: OccupancyMap, track, half_width: float = 0.35,
             link_cones: float = 1.1) -> Optional[List[Point]]:
    """Cheapest closed lap that crosses the finish line in the racing direction.

    The finish line is cut out of the grid, then a shortest path is searched
    from the cells just after it back round to the cells just before it. That
    path plus the crossing is one lap, which both proves the circuit is
    lappable and gives a usable centreline. The cost prefers open track, so the
    result hugs the middle of the corridor rather than scraping the walls.
    """
    from roboracer_referee.geometry import signed_side          # noqa: PLC0415

    w = grid.width
    if link_cones:
        joined = grid.link_cones(link_cones)
        if joined:
            print(f"  sealed {joined} gap(s) between neighbouring cones")
    clear = grid.clearance()
    drivable = [c >= half_width for c in clear]

    line_a, line_b = track.finish_line
    direction = track.crossing_direction

    # Rasterise the finish line, thick enough that no diagonal step can slip
    # through it without being counted.
    cut = set()
    samples = max(2, int(math.dist(line_a, line_b) / (grid.resolution / 3.0)))
    for i in range(samples + 1):
        t = i / samples
        px = line_a[0] + t * (line_b[0] - line_a[0])
        py = line_a[1] + t * (line_b[1] - line_a[1])
        row, col = grid.to_cell(px, py)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if grid.in_bounds(row + dr, col + dc):
                    cut.add((row + dr) * w + col + dc)

    def side_of(index: int) -> float:
        row, col = divmod(index, w)
        return signed_side(line_a, line_b, grid.to_world(row, col)) * direction

    # After a legal crossing the car is on the side whose sign matches the
    # crossing direction; a lap ends by arriving back on the other side.
    starts, goals = [], set()
    for index in cut:
        row, col = divmod(index, w)
        for dr, dc, _ in NEIGHBOURS:
            j = (row + dr) * w + col + dc
            if not grid.in_bounds(row + dr, col + dc) or j in cut or not drivable[j]:
                continue
            if side_of(j) > 0:
                starts.append(j)
            elif side_of(j) < 0:
                goals.add(j)
    if not starts or not goals:
        return None

    best: Dict[int, float] = {}
    prev: Dict[int, Optional[int]] = {}
    queue: List[Tuple[float, int]] = []
    for s in set(starts):
        best[s] = 0.0
        prev[s] = None
        heapq.heappush(queue, (0.0, s))

    found = None
    while queue:
        cost, i = heapq.heappop(queue)
        if cost > best.get(i, INF):
            continue
        if i in goals:
            found = i
            break
        row, col = divmod(i, w)
        for dr, dc, step in NEIGHBOURS:
            rr, cc = row + dr, col + dc
            if not grid.in_bounds(rr, cc):
                continue
            j = rr * w + cc
            if j in cut or not drivable[j]:
                continue
            nxt = cost + step * grid.resolution
            if nxt < best.get(j, INF):
                best[j] = nxt
                prev[j] = i
                heapq.heappush(queue, (nxt, j))

    if found is None:
        return None

    path, i = [], found
    while i is not None:
        path.append(grid.to_world(*divmod(i, w)))
        i = prev[i]
    path.reverse()
    return path


def path_length(points: Sequence[Point]) -> float:
    return sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))


def smooth_closed(points: List[Point], window: int) -> List[Point]:
    """Moving average around a closed loop, to take the staircase off the grid path."""
    if window < 2 or len(points) < window:
        return points
    n = len(points)
    half = window // 2
    out = []
    for i in range(n):
        sx = sy = 0.0
        for k in range(-half, half + 1):
            px, py = points[(i + k) % n]
            sx += px
            sy += py
        out.append((sx / (2 * half + 1), sy / (2 * half + 1)))
    return out


def enforce_clearance(grid: "OccupancyMap", points: List[Point],
                      target: float, iterations: int = 60) -> List[Point]:
    """Push the line off the walls after smoothing, then re-smooth lightly.

    Averaging a grid path straightens it, which is what we want, but it also
    cuts corners - and on a 1.8 m corridor a cut corner is a wall. Each pass
    nudges any point that is too close to a wall towards more open track and
    then lightly re-smooths, so the line stays continuous.
    """
    step = grid.resolution
    probes = [(math.cos(k * math.pi / 8), math.sin(k * math.pi / 8)) for k in range(16)]
    points = list(points)
    for _ in range(iterations):
        worst = INF
        moved = False
        for i, (x, y) in enumerate(points):
            room = grid.clearance_at(x, y)
            worst = min(worst, room)
            if room >= target:
                continue
            # Step towards whichever direction opens the most room.
            best_dir, best_room = None, room
            for dx, dy in probes:
                probe = grid.clearance_at(x + dx * step, y + dy * step)
                if probe > best_room:
                    best_dir, best_room = (dx, dy), probe
            if best_dir is not None:
                points[i] = (x + best_dir[0] * step, y + best_dir[1] * step)
                moved = True
        if not moved and worst >= target:
            break
        points = smooth_closed(points, 5)
    return points


def resample_closed(points: List[Point], spacing: float) -> List[Point]:
    """Even spacing around a closed loop."""
    loop = points + [points[0]]
    total = path_length(loop)
    count = max(8, int(total / spacing))
    out, target, travelled, i = [], 0.0, 0.0, 0
    for _ in range(count):
        while i < len(loop) - 2 and travelled + math.dist(loop[i], loop[i + 1]) < target:
            travelled += math.dist(loop[i], loop[i + 1])
            i += 1
        seg = math.dist(loop[i], loop[i + 1])
        t = (target - travelled) / seg if seg > 1e-9 else 0.0
        out.append((loop[i][0] + t * (loop[i + 1][0] - loop[i][0]),
                    loop[i][1] + t * (loop[i + 1][1] - loop[i][1])))
        target += total / count
    return out


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _resolve_map_yaml(map_path: str, ext: str) -> Optional[str]:
    """Find a map yaml on the host, though map_path is an in-container path."""
    candidates = [map_path + ".yaml"]
    # map_path is an in-container path; on the host the same file lives under
    # the repository's maps/ directory.
    candidates.append(os.path.join(REPO_ROOT, "maps", os.path.basename(map_path)) + ".yaml")
    marker = "/maps/"
    if marker in map_path:
        tail = map_path.split(marker, 1)[1]
        candidates.append(
            os.path.join(REPO_ROOT, "external", "f1tenth_gym_ros", "maps", tail) + ".yaml")
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _load(config: str, name: Optional[str] = None):
    from roboracer_referee.tracks import TrackError, load_tracks     # noqa: PLC0415
    try:
        tracks, default = load_tracks(config)
    except TrackError as exc:
        raise SystemExit(f"{config}: {exc}")
    chosen = name or default
    if chosen not in tracks:
        raise SystemExit(f"Unknown track '{chosen}'. Available: {', '.join(sorted(tracks))}")
    return tracks, chosen


def cmd_validate(args) -> int:
    tracks, _ = _load(args.config)
    print(f"Checking {len(tracks)} track(s) from {args.config}\n")

    failures = 0
    for name in sorted(tracks):
        track = tracks[name]
        problems: List[str] = []
        notes: List[str] = []

        yaml_path = _resolve_map_yaml(track.map_path, track.map_image_ext)
        grid: Optional[OccupancyMap] = None
        if yaml_path is None:
            problems.append(f"no map found for map_path '{track.map_path}'")
        else:
            try:
                grid = OccupancyMap(yaml_path)
            except Exception as exc:                              # noqa: BLE001
                problems.append(f"could not read {yaml_path}: {exc}")

        if grid is not None:
            sx, sy, _ = track.start_pose
            if not grid.is_free(sx, sy):
                problems.append(f"start pose ({sx:.2f}, {sy:.2f}) is inside a wall or off the map")
            else:
                room = grid.clearance_at(sx, sy)
                if room < args.half_width:
                    problems.append(
                        f"start pose has only {room:.2f} m of clearance; "
                        f"the car needs at least {args.half_width:.2f} m")
                else:
                    notes.append(f"start clearance {room:.2f} m")

            a, b = track.finish_line
            clear, hit = grid.segment_is_clear(a, b)
            if not clear:
                problems.append(
                    f"finish line crosses an obstacle at ({hit[0]:.2f}, {hit[1]:.2f}); "
                    f"shorten it or move it")
            else:
                # Extending past each end must hit a wall, otherwise a car can
                # drive round the end of the line and never complete a lap.
                length = math.dist(a, b)
                ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
                past_b = grid.free_distance(b[0], b[1], ux, uy, limit=2.0)
                past_a = grid.free_distance(a[0], a[1], -ux, -uy, limit=2.0)
                if max(past_a, past_b) > args.half_width:
                    problems.append(
                        f"finish line stops {past_a:.2f} m / {past_b:.2f} m short of the walls; "
                        f"a car could pass an end without crossing it")
                else:
                    notes.append(f"line {length:.2f} m, gaps {past_a:.2f}/{past_b:.2f} m")

            if not problems:
                lap = find_lap(grid, track, args.half_width, args.link_cones)
                if lap is None:
                    problems.append(
                        "no closed lap exists through this finish line for a car "
                        f"{2 * args.half_width:.2f} m wide - the circuit is not lappable")
                else:
                    notes.append(f"lap {path_length(lap):.1f} m")

        if problems:
            failures += 1
            print(f"FAIL  {name}")
            for problem in problems:
                print(f"      - {problem}")
        else:
            print(f"ok    {name:<12} " + ", ".join(notes)
                  + f", direction {track.crossing_direction:+d}")

    print()
    if failures:
        print(f"{failures} of {len(tracks)} track(s) failed validation.", file=sys.stderr)
        return 1
    print(f"All {len(tracks)} track(s) valid.")
    return 0


def cmd_centerline(args) -> int:
    tracks, name = _load(args.config, args.track)
    track = tracks[name]

    yaml_path = _resolve_map_yaml(track.map_path, track.map_image_ext)
    if yaml_path is None:
        raise SystemExit(f"No map found for map_path '{track.map_path}'")
    grid = OccupancyMap(yaml_path)

    print(f"Tracing a lap of '{name}' for a car {2 * args.half_width:.2f} m wide...")
    lap = find_lap(grid, track, args.half_width, args.link_cones)
    if lap is None:
        raise SystemExit(
            "No closed lap found. Either the finish line is wrong, or the corridor "
            f"is narrower than {2 * args.half_width:.2f} m somewhere. "
            "Run 'validate' for details.")

    raw_length = path_length(lap)
    lap = smooth_closed(lap, args.smooth)
    lap = enforce_clearance(grid, lap, args.half_width)
    lap = resample_closed(lap, args.spacing)
    print(f"  {raw_length:.1f} m raw -> {path_length(lap + [lap[0]]):.1f} m smoothed, "
          f"{len(lap)} points at {args.spacing:.2f} m spacing")

    tightest = min(grid.clearance_at(x, y) for x, y in lap)
    print(f"  tightest point on the line: {tightest:.2f} m of clearance")
    if tightest < args.half_width:
        print(f"  WARNING: that is below the {args.half_width:.2f} m the car needs. "
              f"Re-run with a smaller --smooth.", file=sys.stderr)

    output = args.output or os.path.join(REPO_ROOT, "maps", f"{name}_centerline.csv")
    with open(output, "w") as handle:
        # The header has to be the FIRST line: the gym reads line 0 as the
        # column names, so anything above it breaks track loading outright.
        handle.write("# x_m, y_m\n")
        handle.write(f"# Centreline of '{name}', traced from {os.path.basename(yaml_path)}\n")
        handle.write(f"# by scripts/track_tool.py centerline. Lap length "
                     f"{path_length(lap + [lap[0]]):.2f} m.\n")
        handle.write("# This is the middle of the track, NOT a racing line: a starting\n")
        handle.write("# point to shift towards the apexes and add a speed profile to.\n")
        for x, y in lap:
            handle.write(f"{x:.4f}, {y:.4f}\n")
    print(f"Wrote {output}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to tracks.yaml")
    # Not the car's half width (0.155 m) - the radius its corners sweep when
    # turning, hypot(half_length, half_width) = 0.33 m. Clearance below that
    # lets the search thread a gap the car cannot actually take, which shows up
    # as a car that scrapes the same three places on every lap.
    parser.add_argument("--half-width", type=float, default=0.35,
                        help="clearance the car needs along the path, in metres "
                             "(default 0.35: its turning envelope, not its half width)")
    parser.add_argument("--link-cones", type=float, default=1.1,
                        help="join cones closer together than this into a solid "
                             "barrier, so no path can thread a cone row. 0 disables.")
    sub = parser.add_subparsers(dest="command", required=True)

    val = sub.add_parser("validate", help="check tracks.yaml against the map image")
    val.set_defaults(func=cmd_validate)

    cen = sub.add_parser("centerline", help="trace a centreline CSV from the map")
    cen.add_argument("--track", default=None, help="track name (default: the default track)")
    cen.add_argument("--output", default=None, help="CSV path to write")
    cen.add_argument("--spacing", type=float, default=0.20, help="point spacing in metres")
    cen.add_argument("--smooth", type=int, default=21,
                     help="moving-average window, in grid path points")
    cen.set_defaults(func=cmd_centerline)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
