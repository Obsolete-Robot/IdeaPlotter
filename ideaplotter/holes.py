"""Holes in the map, and how to turn one into a point in the full embedding space.

A hole is an empty ball in the 3-D map that has games on (nearly) every side. The
map is only a projection, so the hole itself has no coordinates in the 1024-d space.
It gets one from the games around its rim: rim games are grouped by direction (sector)
around the centre, anchors are drawn from *different* sectors so they surround the gap,
and the usual void optimisation pushes their blend as far from every game as it can go.
Rim games come from different regions, so these blends are allowed to be less coherent
than the within-genre voids of `voids.find`: that is what "between regions" means.
"""
import numpy as np
from scipy import ndimage

from .space import Space
from .voids import describe_target, optimize, pick_best


def _directions() -> np.ndarray:
    """26 unit vectors: toward the faces, edges and corners of a cube around the centre."""
    d = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)], float)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def find_holes(space: Space, view: list[float], grid: int = 128, max_holes: int = 40,
               min_coverage: float = 0.8, min_r: float = 2.0) -> list[dict]:
    """Largest empty balls inside the 3-D cloud (distance transform + coverage test over 26 directions)."""
    lo, hi = np.array(view[:3]), np.array(view[3:])
    g = np.round((space.xyz_q - lo) / (hi - lo) * (grid - 1)).astype(int)
    ok = ((g >= 0) & (g < grid)).all(1)
    counts = np.zeros((grid, grid, grid))
    np.add.at(counts, tuple(g[ok].T), 1)
    occupied = ndimage.uniform_filter(counts, 3) * 27 >= 2  # ignore lone strays inside a gap
    dist = ndimage.distance_transform_edt(~occupied)
    peaks = (dist == ndimage.maximum_filter(dist, size=7)) & (dist >= min_r)
    cand = sorted(zip(*np.nonzero(peaks)), key=lambda p: -dist[p])
    dirs = _directions()
    holes = []
    for c in cand:
        c, r = np.array(c, float), float(dist[tuple(c)])
        if any(np.linalg.norm(c - h["_c"]) < (r + h["_r"]) * 0.9 for h in holes):
            continue
        hits = 0
        for u in dirs:
            for t in np.arange(r, 4 * r + 1):
                p = np.round(c + t * u).astype(int)
                if not ((p >= 0) & (p < grid)).all():
                    break
                if occupied[tuple(p)]:
                    hits += 1
                    break
        if hits / len(dirs) < min_coverage:
            continue  # open to the outside: an edge of the cloud, not a hole
        holes.append({"_c": c, "_r": r, "coverage": hits / len(dirs)})
        if len(holes) == max_holes:
            break
    step = (hi - lo) / (grid - 1)
    return [{"id": f"h{n:02d}", "xyz": [round(float(v), 1) for v in lo + h["_c"] * step],
             "r": round(float(h["_r"] * step.mean()), 1), "coverage": round(h["coverage"], 2)}
            for n, h in enumerate(holes, 1)]


def find_holes_2d(space: Space, view: list[float], grid: int = 360, max_holes: int = 40,
               min_coverage: float = 0.8) -> list[dict]:
    """Largest empty discs inside the 2-D map (distance transform + angular coverage test)."""
    x0, y0, x1, y1 = view
    gx = np.round((space.xy_q[:, 0] - x0) / (x1 - x0) * (grid - 1)).astype(int)
    gy = np.round((space.xy_q[:, 1] - y0) / (y1 - y0) * (grid - 1)).astype(int)
    ok = (gx >= 0) & (gx < grid) & (gy >= 0) & (gy < grid)
    counts = np.zeros((grid, grid))
    np.add.at(counts, (gy[ok], gx[ok]), 1)
    occupied = ndimage.uniform_filter(counts, 3) * 9 >= 2  # ignore lone strays inside a gap
    dist = ndimage.distance_transform_edt(~occupied)
    peaks = (dist == ndimage.maximum_filter(dist, size=9)) & (dist >= 3.5)
    cand = sorted(zip(*np.nonzero(peaks)), key=lambda p: -dist[p])
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    holes = []
    for cy, cx in cand:
        r = float(dist[cy, cx])
        if any(np.hypot(cx - h["_c"][0], cy - h["_c"][1]) < (r + h["_r"]) * 0.9 for h in holes):
            continue
        hits = 0
        for a in angles:
            for t in np.arange(r, 4 * r + 1):
                px, py = int(round(cx + t * np.cos(a))), int(round(cy + t * np.sin(a)))
                if not (0 <= px < grid and 0 <= py < grid):
                    break
                if occupied[py, px]:
                    hits += 1
                    break
        if hits / len(angles) < min_coverage:
            continue  # open to the outside: an edge of the cloud, not a hole
        holes.append({"_c": (cx, cy), "_r": r, "coverage": hits / len(angles)})
        if len(holes) == max_holes:
            break
    sx, sy = (x1 - x0) / (grid - 1), (y1 - y0) / (grid - 1)
    out = []
    for n, h in enumerate(holes, 1):
        out.append({"id": f"h{n:02d}", "xy": [round(x0 + h["_c"][0] * sx, 1), round(y0 + h["_c"][1] * sy, 1)],
                    "r": round(h["_r"] * (sx + sy) / 2, 1), "coverage": round(h["coverage"], 2)})
    return out


def rim(space: Space, center, radius: float, sectors: int = 8, per_sector: int = 8) -> list[np.ndarray]:
    """Nearest SFW games around the gap, grouped by direction from the centre.

    A 3-D centre uses the 3-D map and 14 directions (cube faces and corners); a 2-D one,
    saved by older versions, uses the 2-D map and `sectors` angular slices."""
    center = np.asarray(center, dtype=np.float32)
    P = space.xyz_q if len(center) == 3 else space.xy_q
    off = P - center
    d = np.linalg.norm(off, axis=1)
    sfw = ~space.games["nsfw"].to_numpy()
    ring = np.flatnonzero((d >= radius * 0.7) & (d <= radius * 3.5) & sfw)
    if len(ring) < 8:  # tiny or open spot: just take the nearest games
        ring = np.flatnonzero(sfw)[np.argsort(d[sfw])[:64]]
    if len(center) == 3:
        dirs = _directions()[[0, 2, 4, 6, 8, 10, 12, 13, 15, 17, 19, 21, 23, 25]]  # 8 corners + 6 faces
        sec = np.argmax(off[ring] / np.maximum(d[ring, None], 1e-6) @ dirs.T, axis=1)
        n_sec = len(dirs)
    else:
        ang = np.arctan2(off[ring, 1], off[ring, 0])
        sec = ((ang + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
        n_sec = sectors
    groups = []
    for s in range(n_sec):
        rows = ring[sec == s]
        if len(rows):
            groups.append(rows[np.argsort(d[rows])[:per_sector]])
    return groups


def hole_target(space: Space, center, radius: float, exclude=None, anchors=(3, 5), n_candidates: int = 512,
                min_coherence: float = 0.7, seed=None) -> tuple[dict, np.ndarray]:
    groups = rim(space, center, radius)
    if len(groups) < 2:
        raise ValueError("There are not enough games around this spot to triangulate it. Pick a spot inside the cloud.")
    rng = np.random.default_rng(seed)
    lo, hi = min(anchors[0], len(groups)), min(anchors[1], len(groups))
    idx = np.zeros((n_candidates, hi), dtype=np.int64)
    mask = np.zeros((n_candidates, hi), dtype=bool)
    for r in range(n_candidates):
        m = int(rng.integers(lo, hi + 1))
        for k, g in enumerate(rng.choice(len(groups), size=m, replace=False)):
            idx[r, k] = groups[g][rng.integers(len(groups[g]))]
            mask[r, k] = True
    # min_share keeps every side of the hole in the blend, so the target stays *between* them
    X, W, C = optimize(space, idx, mask, steps=150, seed=int(rng.integers(1 << 30)), batch=512,
                       min_coherence=min_coherence, min_share=0.5, verbose=False)
    r = pick_best(space, X, C, min_coherence, exclude)
    anc = sorted([(int(a), float(w)) for a, w, m in zip(idx[r], W[r], mask[r]) if m], key=lambda t: -t[1])
    v = describe_target(space, X[r], anc, C[r])
    regions = list(dict.fromkeys(space.cluster_names[space.cluster[g[0]]] for g in groups))
    v["context"] = ("This point fills a gap in the map of Steam: an empty area surrounded on all sides by "
                    "these regions: " + "; ".join(regions) + ". The blend recipe below takes one game from "
                    "different sides of the gap. The game should belong between those regions, not inside any one of them.")
    v["surrounding_regions"] = regions
    return v, X[r]
