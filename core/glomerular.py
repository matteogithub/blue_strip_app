"""
core/glomerular.py
------------------
Post-processes Strategy C blob detections into glomerular metrics.
No GUI imports.

Three outputs per lobe:
  - Automated Radial Glomerular Count (aRGC)
  - Absolute glomerular density relative to cortical area
  - Depth distribution and maturational generation count
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks
from skimage.measure import find_contours
from skimage.morphology import convex_hull_image

from config import Config


# ── Filtering ──────────────────────────────────────────────────────────────────

def filter_glomeruli(blobs, E, lobe_mask, boundary_um, config):
    """
    Apply three sequential filters to the raw blob list from Strategy C.

    Filter 1 — depth:
        Discard blobs with depth_um <= boundary_um (inside the blue strip).
    Filter 2 — size:
        Keep blobs whose r_um is in [sc_min_r_um, sc_max_r_um].
    Filter 3 — circularity:
        Sample eosin channel E on a ring of radius 0.85 * r_px at 24 points.
        Compute CV = std / (mean + 1e-8).  Discard if CV > gl_circ_cv_max.

    Returns the surviving blobs; each gains 'circularity_cv' and 'kept' keys.
    """
    px = config.pixel_size_um
    H, W = E.shape
    effective_boundary = 0.0 if np.isnan(boundary_um) else boundary_um
    result = []

    for b in blobs:
        # Filter 1: must be below the blue strip
        if b["depth_um"] <= effective_boundary:
            continue

        # Filter 2: radius in permitted range
        r_um = b["r_px"] * px
        if not (config.sc_min_r_um <= r_um <= config.sc_max_r_um):
            continue

        # Filter 3: circularity via eosin-ring CV
        cy, cx = float(b["cy"]), float(b["cx"])
        ring_r  = 0.85 * b["r_px"]
        angles  = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        row_coords = np.clip(cy + ring_r * np.sin(angles), 0, H - 1)
        col_coords = np.clip(cx + ring_r * np.cos(angles), 0, W - 1)
        samples = map_coordinates(E, [row_coords, col_coords], order=1)
        cv = float(samples.std() / (samples.mean() + 1e-8))

        if cv > config.gl_circ_cv_max:
            continue

        blob_copy = dict(b)
        blob_copy["circularity_cv"] = cv
        blob_copy["kept"] = True
        result.append(blob_copy)

    return result


# ── aRGC ───────────────────────────────────────────────────────────────────────

def _convex_hull_contour(lobe_mask, n_pts):
    """
    Return (pts, arc) where pts is an (n_pts, 2) array of evenly sampled
    [row, col] coordinates on the convex-hull contour of lobe_mask, and arc
    is the cumulative arc length in pixels.
    Returns (None, None) if the contour cannot be extracted.
    """
    try:
        hull = convex_hull_image(lobe_mask)
    except Exception:
        hull = lobe_mask
    contours = find_contours(hull.astype(float), level=0.5)
    if not contours:
        return None, None
    contour = max(contours, key=len)
    idx  = np.round(np.linspace(0, len(contour) - 1, n_pts)).astype(int)
    pts  = contour[idx]
    diffs = np.diff(pts, axis=0)
    arc  = np.concatenate([[0.0], np.sqrt((diffs ** 2).sum(axis=1)).cumsum()])
    return pts, arc


def compute_argc(glomeruli, lobe_mask, dist_from_bg, config,
                 n_contour_pts=200):
    """
    Automated Radial Glomerular Count (aRGC).

    For each of n_contour_pts evenly spaced capsule points, cast an
    inward-normal ray and count filtered glomeruli whose centres lie within
    a lateral distance of gl_corridor_um / 2 from the ray and on the
    inward side.

    Returns a dict with:
        argc_mean, argc_sd, argc_profile (array), arc_um (array)
    """
    px       = config.pixel_size_um
    half_w_px = (config.gl_corridor_um / 2.0) / px

    pts, arc = _convex_hull_contour(lobe_mask, n_contour_pts)
    if pts is None or len(glomeruli) == 0:
        return {
            "argc_mean":    0.0,
            "argc_sd":      0.0,
            "argc_profile": np.zeros(n_contour_pts),
            "arc_um":       np.zeros(n_contour_pts),
        }

    H, W       = lobe_mask.shape
    g_centres  = np.array([[b["cy"], b["cx"]] for b in glomeruli], dtype=float)
    ray_count  = np.zeros(len(pts))

    for i in range(len(pts)):
        r, c   = pts[i]
        prev   = pts[i - 1]
        nxt    = pts[(i + 1) % len(pts)]
        tangent = nxt - prev
        tangent /= np.linalg.norm(tangent) + 1e-8
        na = np.array([-tangent[1],  tangent[0]])
        nb = np.array([ tangent[1], -tangent[0]])

        # Pick inward normal using the same test as fusion._spatial_profile
        da = db = 0
        for st in [3, 6, 10]:
            ra = int(round(r + st * na[0])); ca = int(round(c + st * na[1]))
            rb = int(round(r + st * nb[0])); cb = int(round(c + st * nb[1]))
            da = dist_from_bg[ra, ca] if 0 <= ra < H and 0 <= ca < W else 0
            db = dist_from_bg[rb, cb] if 0 <= rb < H and 0 <= cb < W else 0
            if da != db:
                break
        inward = na if da > db else nb

        p      = np.array([r, c])
        g_vecs = g_centres - p                               # (N, 2)
        dots   = g_vecs @ inward                             # (N,) along-ray
        # Perpendicular distance to the ray line (cross product magnitude)
        cross  = np.abs(g_vecs[:, 0] * inward[1]
                        - g_vecs[:, 1] * inward[0])          # (N,) pixels

        ray_count[i] = float(((cross <= half_w_px) & (dots > 0)).sum())

    return {
        "argc_mean":    float(np.mean(ray_count)),
        "argc_sd":      float(np.std(ray_count)),
        "argc_profile": ray_count,
        "arc_um":       arc * px,
    }


# ── Density ────────────────────────────────────────────────────────────────────

def compute_density(glomeruli, lobe_mask, strip_area_mm2, config):
    """
    Absolute glomerular density relative to cortical area
    (lobe area minus blue-strip area).

    Returns:
        n_total, cortical_area_mm2, density_per_mm2
    """
    px = config.pixel_size_um
    lobe_area_mm2     = lobe_mask.sum() * px ** 2 / 1e6
    cortical_area_mm2 = max(lobe_area_mm2 - strip_area_mm2, 1e-6)
    n = len(glomeruli)
    return {
        "n_total":           n,
        "cortical_area_mm2": cortical_area_mm2,
        "density_per_mm2":   n / cortical_area_mm2,
    }


# ── Depth profile / generation detection ─────────────────────────────────────

def compute_depth_profile(glomeruli, config):
    """
    Build a depth histogram and detect maturational generations
    (peaks in the Gaussian-smoothed histogram).

    Returns:
        depths_um, hist_counts, hist_smoothed, bin_centres_um,
        generation_count, generation_depths_um
    """
    if not glomeruli:
        return {
            "depths_um":            [],
            "hist_counts":          np.array([], dtype=int),
            "hist_smoothed":        np.array([]),
            "bin_centres_um":       np.array([]),
            "generation_count":     0,
            "generation_depths_um": [],
        }

    depths   = [b["depth_um"] for b in glomeruli]
    bin_w    = config.gl_gen_bin_um
    bins     = np.arange(0, max(depths) + bin_w, bin_w)
    counts, edges = np.histogram(depths, bins=bins)
    centres  = (edges[:-1] + edges[1:]) / 2.0
    smoothed = gaussian_filter1d(counts.astype(float), sigma=1.5)

    peaks, _ = find_peaks(smoothed,
                          height=config.gl_gen_min_count,
                          distance=2)
    return {
        "depths_um":            depths,
        "hist_counts":          counts,
        "hist_smoothed":        smoothed,
        "bin_centres_um":       centres,
        "generation_count":     int(len(peaks)),
        "generation_depths_um": [float(centres[p]) for p in peaks],
    }


# ── Orchestrator ───────────────────────────────────────────────────────────────

@dataclass
class GlomerularResult:
    """All glomerular outputs for one lobe."""
    glomeruli:     list   # filtered blobs (with circularity_cv and kept=True)
    n_raw:         int    # blobs before any filtering
    n_filtered:    int    # blobs surviving all three filters
    argc:          dict   # output of compute_argc
    density:       dict   # output of compute_density
    depth_profile: dict   # output of compute_depth_profile


def analyse_glomeruli(blobs, E, lobe_mask, dist_from_bg,
                      boundary_um, strip_area_mm2, config):
    """
    Orchestrate filtering → density → aRGC → depth profile.
    Returns None if blobs is empty.
    """
    if not blobs:
        return None

    n_raw    = len(blobs)
    filtered = filter_glomeruli(blobs, E, lobe_mask, boundary_um, config)

    density    = compute_density(filtered, lobe_mask, strip_area_mm2, config)
    argc       = compute_argc(filtered, lobe_mask, dist_from_bg, config)
    depth_prof = compute_depth_profile(filtered, config)

    return GlomerularResult(
        glomeruli=filtered,
        n_raw=n_raw,
        n_filtered=len(filtered),
        argc=argc,
        density=density,
        depth_profile=depth_prof,
    )
