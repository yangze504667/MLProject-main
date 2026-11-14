"""Spatially pruned interference queries for cable polylines and surface voxels."""

import numpy as np


def _segment_distance(a, b, c, d):
    u, v = b - a, d - c
    uu, vv, uv = float(u @ u), float(v @ v), float(u @ v)
    w = a - c
    uw, vw = float(u @ w), float(v @ w)
    candidates = [(0.0, np.clip(vw / vv, 0, 1) if vv else 0.0),
                  (1.0, np.clip((vw + uv) / vv, 0, 1) if vv else 0.0),
                  (np.clip(-uw / uu, 0, 1) if uu else 0.0, 0.0),
                  (np.clip((uv - uw) / uu, 0, 1) if uu else 0.0, 1.0)]
    den = uu * vv - uv * uv
    if den > 1e-14 * uu * vv:
        s, t = (uv * vw - vv * uw) / den, (uu * vw - uv * uw) / den
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append((s, t))
    s, t = min(candidates, key=lambda st: np.dot(w + st[0] * u - st[1] * v,
                                                  w + st[0] * u - st[1] * v))
    pa, pb = a + s * u, c + t * v
    return float(np.linalg.norm(pa - pb)), pa, pb


def _segment_box_closest(a, b, lo, hi):
    """Minimum segment-box distance and the parameter of a closest point."""
    direction = b - a
    breaks = [0.0, 1.0]
    for axis in range(3):
        if direction[axis] == 0:
            continue
        for bound in (lo[axis], hi[axis]):
            t = (bound - a[axis]) / direction[axis]
            if 0 < t < 1:
                breaks.append(float(t))
    breaks = sorted(set(breaks))
    best, best_t = float("inf"), 0.0
    for left, right in zip(breaks[:-1], breaks[1:]):
        middle = a + (left + right) * 0.5 * direction
        below, above = middle < lo, middle > hi
        active = below | above
        slope = direction * active
        offset = (a - np.where(below, lo, hi)) * active
        quadratic = float(slope @ slope)
        t = np.clip(-float(slope @ offset) / quadratic, left, right) if quadratic else left
        point = a + t * direction
        gap = point - np.clip(point, lo, hi)
        distance2 = float(gap @ gap)
        if distance2 < best:
            best, best_t = distance2, float(t)
    return best ** 0.5, best_t


def _segment_box_distance(a, b, lo, hi):
    return _segment_box_closest(a, b, lo, hi)[0]


def _segment_box_interval(a, b, lo, hi, radius, tolerance):
    distance, middle = _segment_box_closest(a, b, lo, hi)
    if distance >= radius - tolerance:
        return None
    direction = b - a
    radius2 = radius * radius

    def inside(t):
        point = a + t * direction
        gap = point - np.clip(point, lo, hi)
        return float(gap @ gap) < radius2

    entry, exit_ = 0.0, 1.0
    if not inside(0.0):
        left, right = 0.0, middle
        for _ in range(48):
            t = (left + right) * 0.5
            if inside(t):
                right = t
            else:
                left = t
        entry = right
    if not inside(1.0):
        left, right = middle, 1.0
        for _ in range(48):
            t = (left + right) * 0.5
            if inside(t):
                left = t
            else:
                right = t
        exit_ = left
    return entry, exit_


def _segment_capsule_interval(a, b, c, d, radius, tolerance):
    """Parameter interval of segment AB strictly inside the capsule around CD."""
    if _segment_distance(a, b, c, d)[0] >= radius - tolerance:
        return None
    u, v, w = b - a, d - c, a - c
    vv = float(v @ v)
    projection, rate = (float(w @ v) / vv, float(u @ v) / vv) if vv else (0.0, 0.0)
    breaks = [0.0, 1.0]
    if rate:
        breaks.extend(t for bound in (0.0, 1.0) if 0 < (t := (bound - projection) / rate) < 1)
    intervals = []
    breaks = sorted(set(breaks))
    for left, right in zip(breaks[:-1], breaks[1:]):
        middle_projection = projection + rate * (left + right) * 0.5
        if not vv or middle_projection <= 0:
            offset, slope = w, u
        elif middle_projection >= 1:
            offset, slope = a - d, u
        else:
            offset, slope = w - projection * v, u - rate * v
        aa, bb = float(slope @ slope), float(slope @ offset)
        cc = float(offset @ offset) - radius * radius
        if aa == 0:
            if cc < 0:
                intervals.append((left, right))
            continue
        discriminant = bb * bb - aa * cc
        if discriminant <= 0:
            continue
        q = -bb - np.copysign(np.sqrt(discriminant), bb)
        first, last = sorted((q / aa, cc / q))
        first, last = max(left, first), min(right, last)
        if first < last:
            intervals.append((first, last))
    return (min(item[0] for item in intervals), max(item[1] for item in intervals)) if intervals else None


class InterferenceIndex:
    """Reusable segment BVH for cables and a dense voxel-grid index for chassis."""

    def __init__(self, cables, occupancy=None, scene=None, tolerance=1e-9):
        self.cables = {}
        self.order = []
        self._nodes = None
        self.tolerance = float(tolerance)
        if self.tolerance < 0 or not np.isfinite(self.tolerance):
            raise ValueError("tolerance must be finite and nonnegative")
        for raw in cables:
            name = str(raw["id"])
            points = np.asarray(raw["points"], dtype=np.float64)
            radius = float(raw["radius"])
            if name in self.cables or points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
                raise ValueError(f"invalid or duplicate cable: {name}")
            if not np.isfinite(points).all() or not np.isfinite(radius) or radius <= 0:
                raise ValueError(f"invalid geometry: {name}")
            self.cables[name] = {"points": points, "radius": radius,
                                 "lo": np.minimum(points[:-1], points[1:]) - radius,
                                 "hi": np.maximum(points[:-1], points[1:]) + radius}
            self.order.append(name)
        self.occupancy = None
        if occupancy is None and scene is None:
            return
        if occupancy is None or scene is None:
            raise ValueError("occupancy and scene must be supplied together")
        self.occupancy = np.asarray(occupancy)
        self.origin = np.asarray(scene["origin"], dtype=np.float64)
        self.pitch = float(scene["voxel_size"])
        self.occupied_value = int(scene.get("occupied_value", 1))
        if self.occupancy.ndim != 3 or self.occupancy.shape != tuple(scene["dimensions"]):
            raise ValueError("occupancy dimensions do not match scene")
        if self.origin.shape != (3,) or not np.isfinite(self.origin).all() or not np.isfinite(self.pitch) or self.pitch <= 0:
            raise ValueError("invalid voxel origin or size")
        if scene.get("axis_order") != "xyz":
            raise ValueError("only xyz axis order is supported")

    @staticmethod
    def _limit(max_hits):
        if max_hits is not None and (not isinstance(max_hits, int) or max_hits < 1):
            raise ValueError("max_hits must be a positive integer or None")

    def _candidate(self, name, points):
        cable = self.cables[name]
        if points is None:
            return cable
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("candidate path must contain at least two finite 3D points")
        radius = cable["radius"]
        return {"points": points, "radius": radius,
                "lo": np.minimum(points[:-1], points[1:]) - radius,
                "hi": np.maximum(points[:-1], points[1:]) + radius}

    def _build_bvh(self):
        starts, ends, lows, highs, owners, locals_ = [], [], [], [], [], []
        for owner, name in enumerate(self.order):
            cable = self.cables[name]
            count = len(cable["lo"])
            starts.append(cable["points"][:-1])
            ends.append(cable["points"][1:])
            lows.append(cable["lo"])
            highs.append(cable["hi"])
            owners.append(np.full(count, owner, dtype=np.int32))
            locals_.append(np.arange(count, dtype=np.int32))
        self._starts = np.concatenate(starts) if starts else np.empty((0, 3))
        self._ends = np.concatenate(ends) if ends else np.empty((0, 3))
        self._lows = np.concatenate(lows) if lows else np.empty((0, 3))
        self._highs = np.concatenate(highs) if highs else np.empty((0, 3))
        self._owners = np.concatenate(owners) if owners else np.empty(0, dtype=np.int32)
        self._locals = np.concatenate(locals_) if locals_ else np.empty(0, dtype=np.int32)
        self._indices = np.arange(len(self._owners), dtype=np.int32)
        self._nodes = []
        if len(self._indices):
            self._split(0, len(self._indices))

    def _split(self, start, end):
        ids = self._indices[start:end]
        lo, hi = self._lows[ids].min(axis=0), self._highs[ids].max(axis=0)
        node = len(self._nodes)
        self._nodes.append(None)
        if end - start <= 64:
            self._nodes[node] = (lo, hi, -1, -1, start, end)
            return node
        axis = int(np.argmax(hi - lo))
        centers = self._lows[ids, axis] + self._highs[ids, axis]
        middle = (start + end) // 2
        self._indices[start:end] = ids[np.argpartition(centers, middle - start)]
        left, right = self._split(start, middle), self._split(middle, end)
        self._nodes[node] = (lo, hi, left, right, start, end)
        return node

    def _overlaps(self, lo, hi):
        if not self._nodes:
            return
        stack = [0]
        while stack:
            node_lo, node_hi, left, right, start, end = self._nodes[stack.pop()]
            if np.any(node_lo > hi) or np.any(node_hi < lo):
                continue
            if left >= 0:
                stack.extend((left, right))
                continue
            ids = self._indices[start:end]
            mask = np.all(self._lows[ids] <= hi, axis=1) & np.all(self._highs[ids] >= lo, axis=1)
            yield from ids[mask]

    def query_cables(self, cable_id, other_id=None, max_hits=1, points=None):
        """Return penetrating segment pairs, optionally for a candidate path."""
        self._limit(max_hits)
        name = str(cable_id)
        a = self._candidate(name, points)
        other = str(other_id) if other_id is not None else None
        if other is not None and other not in self.cables:
            raise KeyError(other)
        if other == name:
            return []
        if self._nodes is None:
            self._build_bvh()
        hits = []
        counts = {}
        for i, (start, end) in enumerate(zip(a["points"][:-1], a["points"][1:])):
            for j in self._overlaps(a["lo"][i], a["hi"][i]):
                peer = self.order[self._owners[j]]
                if peer == name or (other is not None and peer != other):
                    continue
                if max_hits is not None and counts.get(peer, 0) >= max_hits:
                    continue
                b = self.cables[peer]
                distance, pa, pb = _segment_distance(start, end, self._starts[j], self._ends[j])
                penetration = a["radius"] + b["radius"] - distance
                if penetration <= self.tolerance:
                    continue
                hits.append({"cable_a": name, "cable_b": peer, "segment_a": i,
                             "segment_b": int(self._locals[j]), "distance": distance,
                             "penetration": penetration, "point_a": pa.tolist(), "point_b": pb.tolist()})
                counts[peer] = counts.get(peer, 0) + 1
                if other is not None and max_hits is not None and counts[peer] >= max_hits:
                    return hits
        return hits

    def _occupied_voxels(self, start, end, radius):
        shape = np.array(self.occupancy.shape)
        lo = np.maximum(np.floor((np.minimum(start, end) - radius - self.origin) / self.pitch).astype(int) - 1, 0)
        hi = np.minimum(np.floor((np.maximum(start, end) + radius - self.origin) / self.pitch).astype(int) + 2, shape)
        if np.any(lo >= hi):
            return
        block = self.occupancy[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        for offset in np.argwhere(block == self.occupied_value):
            voxel = lo + offset
            yield voxel, self.origin + voxel * self.pitch

    def cable_collision_length(self, cable_id, other_id=None, points=None):
        """Arc length of this cable within summed radii of any other centerline."""
        name = str(cable_id)
        cable = self._candidate(name, points)
        other = str(other_id) if other_id is not None else None
        if other is not None and other not in self.cables:
            raise KeyError(other)
        if other == name:
            return 0.0
        if self._nodes is None:
            self._build_bvh()
        total = 0.0
        for i, (start, end) in enumerate(zip(cable["points"][:-1], cable["points"][1:])):
            length = float(np.linalg.norm(end - start))
            if length == 0:
                continue
            intervals = []
            for j in self._overlaps(cable["lo"][i], cable["hi"][i]):
                peer = self.order[self._owners[j]]
                if peer == name or (other is not None and peer != other):
                    continue
                interval = _segment_capsule_interval(start, end, self._starts[j], self._ends[j],
                                                     cable["radius"] + self.cables[peer]["radius"], self.tolerance)
                if interval is not None:
                    intervals.append(interval)
            right = 0.0
            for left, end_t in sorted(intervals):
                total += max(0.0, end_t - max(left, right)) * length
                right = max(right, end_t)
        return float(total)

    def evaluate_cables(self, cable_id, points):
        """Candidate versus the original peer paths; rebuild the index for joint paths."""
        before = self.cable_collision_length(cable_id)
        after = self.cable_collision_length(cable_id, points=points)
        return {"before_length_mm": before, "after_length_mm": after,
                "improvement_pct": None if before <= self.tolerance else 100 * (before - after) / before,
                "before_collides": bool(self.query_cables(cable_id)),
                "after_collides": bool(self.query_cables(cable_id, points=points))}

    def query_chassis(self, cable_id, max_hits=1, points=None):
        """Return penetrations against occupied cubes, optionally for a candidate path."""
        self._limit(max_hits)
        if self.occupancy is None:
            raise ValueError("no chassis occupancy was supplied")
        name = str(cable_id)
        cable = self._candidate(name, points)
        points, radius = cable["points"], cable["radius"]
        hits = []
        for i, (start, end) in enumerate(zip(points[:-1], points[1:])):
            for voxel, box_lo in self._occupied_voxels(start, end, radius):
                distance = _segment_box_distance(start, end, box_lo, box_lo + self.pitch)
                penetration = radius - distance
                if penetration <= self.tolerance:
                    continue
                hits.append({"cable_id": name, "segment": i, "voxel": voxel.tolist(),
                             "distance": distance, "penetration": penetration})
                if max_hits is not None and len(hits) >= max_hits:
                    return hits
        return hits

    def chassis_collision_length(self, cable_id, points=None):
        """Centerline length within radius of occupied surface voxels, in millimeters."""
        if self.occupancy is None:
            raise ValueError("no chassis occupancy was supplied")
        cable = self._candidate(str(cable_id), points)
        total = 0.0
        for start, end in zip(cable["points"][:-1], cable["points"][1:]):
            length = float(np.linalg.norm(end - start))
            if length <= 0:
                continue
            intervals = []
            for _, box_lo in self._occupied_voxels(start, end, cable["radius"]):
                interval = _segment_box_interval(start, end, box_lo, box_lo + self.pitch,
                                                 cable["radius"], self.tolerance)
                if interval is not None:
                    intervals.append(interval)
            right = 0.0
            for left, end_t in sorted(intervals):
                total += max(0.0, end_t - max(left, right)) * length
                right = max(right, end_t)
        return total

    def evaluate_chassis(self, cable_id, points):
        """Compare a candidate to the original path by collided centerline length."""
        before = self.chassis_collision_length(cable_id)
        after = self.chassis_collision_length(cable_id, points=points)
        return {"before_length_mm": before, "after_length_mm": after,
                "improvement_pct": None if before <= self.tolerance else 100 * (before - after) / before,
                "before_collides": bool(self.query_chassis(cable_id)),
                "after_collides": bool(self.query_chassis(cable_id, points=points))}
