"""Procedural aerial-scene simulator for the RS-12K / LULC task sets.

Every patch is rendered at a nominal 0.5 m ground sampling distance with a
per-pixel land-cover map over five classes (water, vegetation, built-up,
barren, cropland). The design encodes the property the paper relies on:

* water is separable on colour alone;
* vegetation vs cropland share a green palette and differ mainly by parcel
  geometry (straight field edges) and crop-row texture (period 4.5-11 px);
* barren vs built-up share a grey-brown palette and differ mainly by
  rectilinear roof/road footprints and cast shadows versus isotropic soil
  texture with curved gullies and tracks.

Those cues sit at spatial frequencies that the x4 degradation of Eq. (1)
removes, so a classifier trained on HR patches loses accuracy on LR input
exactly in the spectrally adjacent pairs.

Splits are made "geographically" disjoint through region styles: each
region fixes soil/vegetation/water/roof palettes, sun geometry, haze, road
orientation and crop calendar, and regions never cross splits.

Only NumPy and OpenCV are used, so the generator runs anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

CLASSES = ["water", "vegetation", "built_up", "barren", "cropland"]
CLASS_TITLES = ["Water", "Vegetation", "Built-up", "Barren", "Cropland"]
W, V, B, R, C = range(5)
SCENE_TYPES = ["harbour", "urban_grid", "farmland", "river_delta", "quarry"]
SCENE_TITLES = ["Harbour", "Urban-Grid", "Farmland", "River-Delta", "Quarry"]

# P(scene type | dominant class)
SCENE_GIVEN_CLASS = {
    W: {"harbour": 0.45, "river_delta": 0.45, "quarry": 0.10},
    V: {"river_delta": 0.35, "farmland": 0.35, "urban_grid": 0.15, "quarry": 0.15},
    B: {"urban_grid": 0.60, "harbour": 0.30, "quarry": 0.10},
    R: {"quarry": 0.50, "river_delta": 0.30, "urban_grid": 0.20},
    C: {"farmland": 0.80, "river_delta": 0.20},
}
# secondary land cover that each scene type can contain
SCENE_SECONDARY = {
    "harbour": [W, B, R],
    "urban_grid": [B, V, R],
    "farmland": [C, V, B, W],
    "river_delta": [W, R, V, C],
    "quarry": [R, W, B, V],
}


# ----------------------------------------------------------------------------
# region style
# ----------------------------------------------------------------------------
@dataclass
class RegionStyle:
    region_id: int
    soil: np.ndarray
    green: np.ndarray
    water: np.ndarray
    asphalt: np.ndarray
    roofs: list
    crop_states: list
    sun_az: float
    sun_el: float
    haze: float
    haze_rgb: np.ndarray
    gain: np.ndarray
    grid_angle: float
    row_period: tuple
    urban_density: float
    extras: dict = field(default_factory=dict)


def make_region_style(region_id: int, seed: int = 0) -> RegionStyle:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7919, region_id]))
    j = lambda s: rng.normal(0, s, 3)  # noqa: E731
    soil_bases = [np.array([0.52, 0.44, 0.36]), np.array([0.58, 0.40, 0.30]),
                  np.array([0.50, 0.48, 0.42]), np.array([0.62, 0.55, 0.43])]
    soil = soil_bases[rng.integers(len(soil_bases))] + j(0.025)
    green = np.array([0.24, 0.34, 0.17]) + j(0.025)
    if rng.random() < 0.7:
        water = np.array([0.10, 0.20, 0.30]) + j(0.02)
    else:
        water = np.array([0.20, 0.28, 0.27]) + j(0.02)
    asphalt = np.array([0.36, 0.36, 0.37]) + rng.normal(0, 0.03)
    roof_pool = [
        np.array([0.62, 0.61, 0.58]), np.array([0.56, 0.34, 0.27]),
        np.array([0.30, 0.30, 0.32]), np.array([0.74, 0.73, 0.70]),
        soil * 1.05, np.array([0.47, 0.45, 0.42]),
    ]
    roofs = [roof_pool[i] + j(0.02) for i in rng.choice(len(roof_pool), 4, replace=False)]
    crop_states = [
        ("green", green * np.array([1.05, 1.08, 1.0]) + j(0.02)),
        ("green", green * np.array([0.92, 1.12, 0.95]) + j(0.02)),
        ("ripe", np.array([0.64, 0.57, 0.36]) + j(0.03)),
        ("plough", soil * 0.92 + j(0.015)),
        ("fallow", 0.5 * green + 0.5 * soil + j(0.02)),
    ]
    return RegionStyle(
        region_id=region_id, soil=soil, green=green, water=water, asphalt=asphalt,
        roofs=roofs, crop_states=crop_states,
        sun_az=rng.uniform(0, 2 * np.pi), sun_el=np.deg2rad(rng.uniform(35, 65)),
        haze=rng.uniform(0.0, 0.08), haze_rgb=np.array([0.70, 0.72, 0.76]) + j(0.02),
        gain=rng.uniform(0.9, 1.1) * (1 + rng.normal(0, 0.03, 3)),
        grid_angle=rng.uniform(0, np.pi / 2), row_period=(rng.uniform(4.5, 6.5), rng.uniform(8.0, 11.0)),
        urban_density=rng.uniform(0.45, 0.9),
    )


# ----------------------------------------------------------------------------
# noise primitives
# ----------------------------------------------------------------------------
def value_noise(h, w, cell, rng):
    gh, gw = int(np.ceil(h / cell)) + 3, int(np.ceil(w / cell)) + 3
    g = rng.random((gh, gw)).astype(np.float32)
    up = cv2.resize(g, (int(gw * cell), int(gh * cell)), interpolation=cv2.INTER_CUBIC)
    oy, ox = rng.integers(0, int(cell) + 1, 2)
    return up[oy:oy + h, ox:ox + w]


def fbm(h, w, rng, base=64, octaves=4, gain=0.5):
    out = np.zeros((h, w), np.float32)
    amp, tot, cell = 1.0, 0.0, float(base)
    for _ in range(octaves):
        out += amp * (value_noise(h, w, max(cell, 1.5), rng) - 0.5)
        tot += amp
        amp *= gain
        cell /= 2
    return out / tot  # roughly in [-0.5, 0.5]


def worley_f1(h, w, cell, rng, jitter=0.9):
    """Distance to the nearest jittered seed (tree-crown texture)."""
    img = np.full((h + 2 * cell, w + 2 * cell), 255, np.uint8)
    ys = np.arange(0, h + 2 * cell, cell)
    xs = np.arange(0, w + 2 * cell, cell)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    yy = (yy + rng.uniform(0, jitter * cell, yy.shape)).astype(int).clip(0, img.shape[0] - 1)
    xx = (xx + rng.uniform(0, jitter * cell, xx.shape)).astype(int).clip(0, img.shape[1] - 1)
    img[yy, xx] = 0
    d = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    return d[cell:cell + h, cell:cell + w]


def rotated_coords(h, w, theta):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    u = x * np.cos(theta) + y * np.sin(theta)
    v = -x * np.sin(theta) + y * np.cos(theta)
    return u, v


def irregular_grid(coord, lo, hi, rng):
    """Cell index and distance-to-edge for an irregular 1-D partition."""
    cmin, cmax = coord.min() - hi, coord.max() + hi
    edges = [cmin]
    while edges[-1] < cmax:
        edges.append(edges[-1] + rng.uniform(lo, hi))
    edges = np.asarray(edges, np.float32)
    idx = np.searchsorted(edges, coord, side="right") - 1
    left = coord - edges[idx]
    right = edges[np.minimum(idx + 1, len(edges) - 1)] - coord
    return idx, np.minimum(left, right), edges


def smooth_curve(h, w, rng, n=6, straight=0.3):
    """Random sinuous polyline crossing the patch."""
    ang = rng.uniform(0, np.pi)
    d = np.array([np.cos(ang), np.sin(ang)])
    nrm = np.array([-d[1], d[0]])
    c = np.array([w / 2, h / 2]) + rng.normal(0, 0.2 * min(h, w), 2)
    L = 1.5 * max(h, w)
    t = np.linspace(-L / 2, L / 2, 60)
    off = np.zeros_like(t)
    for _ in range(n):
        off += rng.normal(0, (1 - straight) * 0.08 * max(h, w)) * np.sin(2 * np.pi * t / rng.uniform(0.4, 1.6) / max(h, w) + rng.uniform(0, 6.3))
    pts = c[None] + t[:, None] * d[None] + off[:, None] * nrm[None]
    return pts.astype(np.float32)


def draw_polyline_mask(h, w, pts, width):
    m = np.zeros((h, w), np.uint8)
    cv2.polylines(m, [np.round(pts).astype(np.int32)], False, 1, thickness=max(1, int(round(width))), lineType=cv2.LINE_8)
    return m.astype(bool)


# ----------------------------------------------------------------------------
# secondary-feature masks (area-controlled)
# ----------------------------------------------------------------------------
def blob_mask(h, w, area, rng, base=None):
    if area <= 0:
        return np.zeros((h, w), bool)
    n = fbm(h, w, rng, base=base or rng.uniform(40, 110), octaves=3)
    thr = np.quantile(n, 1 - area)
    return n > thr


def river_mask(h, w, area, rng):
    pts = smooth_curve(h, w, rng)
    width = np.clip(area * h * w / (1.3 * max(h, w)), 3, 0.45 * min(h, w))
    return draw_polyline_mask(h, w, pts, width)


def shore_mask(h, w, area, rng):
    ang = rng.uniform(0, 2 * np.pi)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    s = (x - w / 2) * np.cos(ang) + (y - h / 2) * np.sin(ang)
    s = s + 8 * fbm(h, w, rng, base=48, octaves=2)
    # rectilinear quays on harbour coasts
    if rng.random() < 0.6:
        s = np.round(s / 6) * 6
    thr = np.quantile(s, 1 - area)
    return s > thr


def parcel_mask(h, w, area, rng, theta):
    u, v = rotated_coords(h, w, theta)
    iu, _, _ = irregular_grid(u, 35, 90, rng)
    iv, _, _ = irregular_grid(v, 35, 90, rng)
    cid = iu * 1000 + iv
    ids = np.unique(cid)
    rng.shuffle(ids)
    m = np.zeros((h, w), bool)
    for k in ids:
        if m.mean() >= area:
            break
        m |= cid == k
    return m


def rect_cluster_mask(h, w, area, rng, theta):
    u, v = rotated_coords(h, w, theta)
    cu, cv_ = rng.uniform(u.min(), u.max()), rng.uniform(v.min(), v.max())
    su = sv = np.sqrt(max(area, 1e-3) * h * w)
    su *= rng.uniform(0.7, 1.4)
    sv = area * h * w / su
    m = (np.abs(u - cu) < su / 2) & (np.abs(v - cv_) < sv / 2)
    if m.mean() < 0.5 * area:  # fell off the patch: centre it
        u0, v0 = u[h // 2, w // 2], v[h // 2, w // 2]
        m = (np.abs(u - u0) < su / 2) & (np.abs(v - v0) < sv / 2)
    return m


# ----------------------------------------------------------------------------
# class renderers: each returns (rgb HxWx3, height HxW in metres)
# ----------------------------------------------------------------------------
def render_water(h, w, st, rng, mask):
    img = np.empty((h, w, 3), np.float32)
    img[:] = st.water
    img += 0.05 * fbm(h, w, rng, base=90, octaves=2)[..., None] * np.array([0.6, 0.8, 1.0])
    th = rng.uniform(0, np.pi)
    u, _ = rotated_coords(h, w, th)
    ripple = np.sin(2 * np.pi * u / rng.uniform(5, 12) + 3 * fbm(h, w, rng, base=24, octaves=2))
    img += 0.010 * ripple[..., None]
    glint = (rng.random((h, w)) < 0.0015).astype(np.float32)
    glint = cv2.GaussianBlur(glint, (0, 0), 0.8) * 3.0
    img += 0.25 * glint[..., None] * rng.uniform(0.0, 1.0)
    # shallow / turbid margins
    if mask is not None and (~mask).any():
        d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        shallow = np.exp(-d / rng.uniform(2.5, 6.0))[..., None]
        img = img * (1 - 0.6 * shallow) + 0.6 * shallow * (0.5 * st.soil + 0.5 * st.water + 0.05)
    return img, np.zeros((h, w), np.float32)


def render_vegetation(h, w, st, rng, mask):
    img = np.empty((h, w, 3), np.float32)
    tone = st.green * (1 + rng.normal(0, 0.04, 3))
    img[:] = tone
    forest = fbm(h, w, rng, base=rng.uniform(50, 120), octaves=3) > rng.uniform(-0.15, 0.2)
    # meadow / shrub: organic multi-scale texture
    meadow = 0.07 * fbm(h, w, rng, base=40, octaves=4) + 0.025 * (rng.random((h, w)) - 0.5)
    img += meadow[..., None] * np.array([0.8, 1.0, 0.6])
    # forest: crowns from a Worley field
    cell = int(rng.integers(6, 10))
    d = worley_f1(h, w, cell, rng)
    r = cell * rng.uniform(0.55, 0.75)
    dome = np.clip(1 - (d / r) ** 2, 0, 1).astype(np.float32)
    gy, gx = np.gradient(dome)
    lit = -(gx * np.cos(st.sun_az) + gy * np.sin(st.sun_az))
    crown = 0.75 + 0.45 * dome + 1.2 * lit
    crown_tone = st.green * np.array([0.85, 0.95, 0.85])
    fimg = crown_tone[None, None] * crown[..., None] + 0.03 * fbm(h, w, rng, base=16, octaves=2)[..., None]
    img = np.where(forest[..., None], fimg, img)
    height = np.where(forest, dome * rng.uniform(7, 14), dome * 0.3).astype(np.float32)
    if rng.random() < 0.25:  # confuser: straight tracks / hedgerows crossing natural vegetation
        t = np.zeros((h, w), np.float32)
        for _ in range(rng.integers(1, 4)):
            pts = smooth_curve(h, w, rng, n=1, straight=0.95)
            cv2.polylines(t, [np.round(pts).astype(np.int32)], False, 1.0, int(rng.integers(1, 3)), cv2.LINE_AA)
        img = img * (1 - 0.5 * t[..., None]) + 0.5 * t[..., None] * (0.6 * st.soil + 0.4 * st.green)
    return img, height


def render_cropland(h, w, st, rng, mask, theta=None):
    theta = st.grid_angle + rng.normal(0, 0.15) if theta is None else theta
    u, v = rotated_coords(h, w, theta)
    strip = rng.random() < 0.35
    big = rng.random() < 0.2  # confuser: one large, weakly-rowed parcel
    if big:
        iu, du, _ = irregular_grid(u, 1.5 * h, 2.5 * h, rng)
        iv, dv, _ = irregular_grid(v, 1.5 * h, 2.5 * h, rng)
    else:
        iu, du, _ = irregular_grid(u, 30 if strip else 45, 60 if strip else 130, rng)
        iv, dv, _ = irregular_grid(v, 45, 140, rng)
    img = np.zeros((h, w, 3), np.float32)
    ids = iu * 1000 + iv
    lo, hi = st.row_period
    for k in np.unique(ids):
        sel = ids == k
        name, col = st.crop_states[rng.integers(len(st.crop_states))]
        col = col * (1 + rng.normal(0, 0.035, 3))
        period = rng.uniform(lo, hi)
        along_u = rng.random() < 0.5
        coord = v if along_u else u
        phase = rng.uniform(0, 2 * np.pi)
        amp = {"green": 0.07, "ripe": 0.06, "plough": 0.10, "fallow": 0.035}[name] * rng.uniform(0.4, 1.3) * (0.35 if big else 1.0)
        rows = np.sin(2 * np.pi * coord[sel] / period + phase)
        rows = np.sign(rows) * np.abs(rows) ** 0.6
        tram = (np.mod(coord[sel] + phase * 10, rng.uniform(22, 36)) < 1.0).astype(np.float32)
        val = col[None] * (1 + amp * rows[:, None]) + 0.05 * tram[:, None]
        img[sel] = val
    img += 0.02 * fbm(h, w, rng, base=64, octaves=2)[..., None]
    # field margins / tracks along parcel edges (straight lines)
    edge = (np.minimum(du, dv) < rng.uniform(0.8, 1.8)).astype(np.float32)
    edge = cv2.GaussianBlur(edge, (0, 0), 0.5)[..., None]
    margin = 0.55 * st.green + 0.45 * st.soil + 0.06
    img = img * (1 - edge) + edge * margin
    return img, np.zeros((h, w), np.float32)


def render_barren(h, w, st, rng, mask, scene_type=None):
    img = np.empty((h, w, 3), np.float32)
    soil = st.soil * (1 + rng.normal(0, 0.03, 3))
    if scene_type == "river_delta":
        soil = 0.75 * soil + 0.25 * np.array([0.75, 0.70, 0.60])
    img[:] = soil
    img += (0.09 * fbm(h, w, rng, base=90, octaves=3) + 0.05 * fbm(h, w, rng, base=12, octaves=2))[..., None]
    img += 0.03 * (rng.random((h, w, 1)) - 0.5)
    # curved gullies
    rid = 1 - 2 * np.abs(fbm(h, w, rng, base=rng.uniform(40, 80), octaves=3))
    gully = np.clip((rid - 0.93) / 0.07, 0, 1)
    img *= (1 - 0.22 * gully[..., None])
    height = np.zeros((h, w), np.float32)
    # quarry benches: stepped iso-contours (curved edges)
    if scene_type == "quarry" and rng.random() < 0.8:
        hf = fbm(h, w, rng, base=160, octaves=2) * rng.uniform(40, 70)
        steps = np.floor(hf / 3.0)
        gy, gx = np.gradient(steps)
        e = np.clip(np.hypot(gx, gy), 0, 1)
        shade = -(gx * np.cos(st.sun_az) + gy * np.sin(st.sun_az))
        img *= (1 + 0.20 * shade[..., None] - 0.06 * e[..., None])
        img += 0.03 * (np.mod(steps, 2)[..., None] - 0.5)
    if rng.random() < 0.25:  # confuser: straight haul roads and rectangular stockpiles
        th = st.grid_angle + rng.normal(0, 0.2)
        for _ in range(rng.integers(2, 7)):
            x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
            poly = np.round(_rect_poly(x0, y0, rng.uniform(8, 30), rng.uniform(8, 30), th)).astype(np.int32)
            cv2.fillConvexPoly(img, poly, tuple(float(q) for q in soil * rng.uniform(0.8, 1.2)), cv2.LINE_AA)
            cv2.fillConvexPoly(height, poly, float(rng.uniform(1.5, 5)))
        pts = smooth_curve(h, w, rng, n=1, straight=0.97)
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, tuple(float(q) for q in soil * 1.12), int(rng.integers(3, 7)), cv2.LINE_AA)
    # vehicle tracks
    for _ in range(rng.integers(0, 3)):
        pts = smooth_curve(h, w, rng, n=3, straight=0.5)
        t = np.zeros((h, w), np.float32)
        cv2.polylines(t, [np.round(pts).astype(np.int32)], False, 1.0, 1, cv2.LINE_AA)
        img += rng.choice([-0.06, 0.05]) * t[..., None]
    # boulders
    nb = rng.integers(0, 40)
    ys, xs = rng.integers(0, h, nb), rng.integers(0, w, nb)
    for y0, x0 in zip(ys, xs):
        rr = int(rng.integers(1, 3))
        cv2.circle(img, (int(x0), int(y0)), rr, tuple(float(c) for c in soil * rng.uniform(0.7, 1.25)), -1, cv2.LINE_AA)
        cv2.circle(height, (int(x0), int(y0)), rr, float(rng.uniform(0.5, 1.5)), -1)
    return img, height


def _rect_poly(cx, cy, a, b, theta):
    c, s = np.cos(theta), np.sin(theta)
    pts = np.array([[-a, -b], [a, -b], [a, b], [-a, b]], np.float32) / 2
    rot = np.stack([pts[:, 0] * c - pts[:, 1] * s, pts[:, 0] * s + pts[:, 1] * c], 1)
    return (rot + [cx, cy]).astype(np.float32)


def render_builtup(h, w, st, rng, mask, theta=None, density=None):
    theta = st.grid_angle + rng.normal(0, 0.05) if theta is None else theta
    if density is None:  # 20 % sparse peri-urban patches with mostly bare lots (confuser)
        density = st.urban_density * rng.uniform(0.7, 1.15) if rng.random() > 0.2 else rng.uniform(0.1, 0.3)
    u, v = rotated_coords(h, w, theta)
    iu, du, eu = irregular_grid(u, 45, 115, rng)
    iv, dv, ev = irregular_grid(v, 45, 115, rng)
    road_w = rng.uniform(2.5, 5.0)
    road = (du < road_w) | (dv < road_w)
    # lot ground: soil / paved / grass
    img = np.empty((h, w, 3), np.float32)
    lot_kind = rng.random((iu.max() + 2, iv.max() + 2))
    k = lot_kind[iu, iv]
    ground = np.where((k < 0.45)[..., None], st.soil * 0.97,
                      np.where((k < 0.8)[..., None], np.array([0.46, 0.45, 0.43]), st.green * 1.05))
    img[:] = ground
    img += (0.06 * fbm(h, w, rng, base=30, octaves=3))[..., None]
    img = np.where(road[..., None], st.asphalt + 0.015 * fbm(h, w, rng, base=20, octaves=2)[..., None], img)
    height = np.zeros((h, w), np.float32)
    # buildings: rotated rectangles inside blocks
    c, s = np.cos(theta), np.sin(theta)
    for a0, a1 in zip(eu[:-1], eu[1:]):
        for b0, b1 in zip(ev[:-1], ev[1:]):
            bu, bv = a1 - a0 - 2 * road_w, b1 - b0 - 2 * road_w
            if bu < 10 or bv < 10:
                continue
            nu = max(1, int(bu // rng.uniform(16, 34)))
            nv = max(1, int(bv // rng.uniform(16, 34)))
            for i in range(nu):
                for jj in range(nv):
                    if rng.random() > density:
                        continue
                    lu, lv = bu / nu, bv / nv
                    fu, fv = lu * rng.uniform(0.45, 0.85), lv * rng.uniform(0.45, 0.85)
                    cu = a0 + road_w + (i + 0.5) * lu + rng.normal(0, 1)
                    cvv = b0 + road_w + (jj + 0.5) * lv + rng.normal(0, 1)
                    x = cu * c - cvv * s
                    y = cu * s + cvv * c
                    if not (-20 < x < w + 20 and -20 < y < h + 20):
                        continue
                    poly = _rect_poly(x, y, fu, fv, theta)
                    roof = st.roofs[rng.integers(len(st.roofs))] * (1 + rng.normal(0, 0.04))
                    hgt = rng.uniform(4, 18)
                    pint = np.round(poly).astype(np.int32)
                    cv2.fillConvexPoly(img, pint, tuple(float(q) for q in roof), cv2.LINE_AA)
                    cv2.fillConvexPoly(height, pint, float(hgt))
                    if rng.random() < 0.5:  # gable ridge: shade half the roof
                        half = _rect_poly(x - 0.25 * fu * c, y - 0.25 * fu * s, fu / 2, fv, theta)
                        cv2.fillConvexPoly(img, np.round(half).astype(np.int32), tuple(float(q) for q in roof * 0.82), cv2.LINE_AA)
    img += 0.012 * (rng.random((h, w, 1)) - 0.5)
    return img, height


RENDERERS = {W: render_water, V: render_vegetation, B: render_builtup, R: render_barren, C: render_cropland}


# ----------------------------------------------------------------------------
# composition
# ----------------------------------------------------------------------------
def cast_shadows(img, height, st, max_px=14):
    if height.max() <= 0:
        return img
    dx, dy = -np.cos(st.sun_az), -np.sin(st.sun_az)
    drop = 0.5 * np.tan(st.sun_el)  # metres of height lost per pixel travelled (0.5 m GSD)
    h, w = height.shape
    pad = max_px + 1
    hp = cv2.copyMakeBorder(height, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    shadow = np.zeros((h, w), np.float32)
    for d in range(1, max_px + 1):
        oy, ox = int(round(-dy * d)), int(round(-dx * d))
        src = hp[pad + oy:pad + oy + h, pad + ox:pad + ox + w]
        shadow = np.maximum(shadow, (src - d * drop - height > 0.3).astype(np.float32))
    shadow = cv2.GaussianBlur(shadow, (0, 0), 0.6)[..., None]
    tint = np.array([0.50, 0.53, 0.60], np.float32)
    return img * (1 - shadow) + img * tint * shadow


def compose_labels(size, dominant, scene_type, purity, st, rng):
    h = w = size
    lab = np.full((h, w), dominant, np.uint8)
    cands = [c for c in SCENE_SECONDARY[scene_type] if c != dominant]
    if purity >= 0.999 or not cands:
        return lab
    k = int(rng.integers(1, min(3, len(cands)) + 1))
    chosen = list(rng.choice(cands, k, replace=False))
    rest = 1 - purity
    shares = rng.dirichlet(np.ones(k)) * rest
    for cls, a in zip(chosen, shares):
        if cls == W:
            m = shore_mask(h, w, a, rng) if scene_type == "harbour" else river_mask(h, w, a, rng)
        elif cls == C:
            m = parcel_mask(h, w, a, rng, st.grid_angle)
        elif cls == B:
            if rng.random() < 0.5:
                m = rect_cluster_mask(h, w, a, rng, st.grid_angle)
            else:
                m = parcel_mask(h, w, a, rng, st.grid_angle)
        else:
            m = blob_mask(h, w, a, rng)
        lab[m] = cls
    return lab


def render_from_labels(lab, st, rng, scene_type):
    h, w = lab.shape
    img = np.zeros((h, w, 3), np.float32)
    height = np.zeros((h, w), np.float32)
    present = np.unique(lab)
    soft = np.zeros((len(present), h, w), np.float32)
    for i, c in enumerate(present):
        soft[i] = cv2.GaussianBlur((lab == c).astype(np.float32), (0, 0), 0.7)
    soft /= soft.sum(0, keepdims=True) + 1e-6
    for i, c in enumerate(present):
        mask = lab == c
        if c == R:
            layer, hl = render_barren(h, w, st, rng, mask, scene_type)
        else:
            layer, hl = RENDERERS[int(c)](h, w, st, rng, mask)
        img += soft[i][..., None] * layer
        height += mask * hl
    return img, height


def finish(img, height, st, rng):
    h, w = height.shape
    img = cast_shadows(img, height, st)
    illum = 1 + 0.04 * fbm(h, w, rng, base=max(h, w), octaves=1)[..., None]
    img = img * illum * st.gain
    img = img * (1 - st.haze) + st.haze * st.haze_rgb
    img = cv2.GaussianBlur(img, (0, 0), 0.45)  # HR sensor MTF
    img += rng.normal(0, 0.6 / 255, img.shape).astype(np.float32)
    return np.clip(np.round(img * 255), 0, 255).astype(np.uint8)


def sample_scene_type(dominant, rng):
    d = SCENE_GIVEN_CLASS[dominant]
    keys = list(d)
    p = np.array([d[k] for k in keys])
    return keys[rng.choice(len(keys), p=p / p.sum())]


def generate_patch(dominant: int, region_style: RegionStyle, seed_seq, size: int = 192,
                   scene_type: str | None = None, purity: float | None = None):
    """Render one labelled HR patch.

    Returns (rgb uint8 HxWx3, label map uint8 HxW, info dict).
    """
    rng = np.random.default_rng(seed_seq)
    st = region_style
    if scene_type is None:
        scene_type = sample_scene_type(dominant, rng)
    for _ in range(8):
        p = purity if purity is not None else (1.0 if rng.random() < 0.25 else rng.uniform(0.62, 0.97))
        lab = compose_labels(size, dominant, scene_type, p, st, rng)
        frac = np.bincount(lab.ravel(), minlength=5) / lab.size
        if frac.argmax() == dominant and frac[dominant] >= 0.55:
            break
    img, height = render_from_labels(lab, st, rng, scene_type)
    rgb = finish(img, height, st, rng)
    info = {"scene_type": scene_type, "purity": float(frac[dominant]),
            **{f"frac_{n}": float(f) for n, f in zip(CLASSES, frac)}}
    return rgb, lab, info


def generate_showcase(region_style: RegionStyle, seed_seq, size=384, scene_type="farmland"):
    """Mixed scene for qualitative figures (river + parcels + buildings + roads)."""
    rng = np.random.default_rng(seed_seq)
    st = region_style
    dom = {"harbour": W, "urban_grid": B, "farmland": C, "river_delta": W, "quarry": R}[scene_type]
    lab = compose_labels(size, dom, scene_type, rng.uniform(0.45, 0.6), st, rng)
    if scene_type in ("farmland", "urban_grid"):
        lab[river_mask(size, size, 0.05, rng)] = W
    img, height = render_from_labels(lab, st, rng, scene_type)
    return finish(img, height, st, rng), lab
