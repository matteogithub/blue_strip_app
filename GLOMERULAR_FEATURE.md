# Task: Implement automated glomerular quantification

Read `ARCHITECTURE.md` and all source files before writing any code.

---

## Context

This app measures the **nephrogenic blue strip** depth in neonatal kidney H&E
sections. Strategy C already detects rounded eosin-positive blobs (glomeruli
and similar structures) via LoG blob detection and stores them in
`StrategyResult.extra["blobs"]` — a list of dicts with keys:
`cy`, `cx`, `r_px`, `r_um`, `depth_um`.

The task is to add a **glomerular quantification layer** that post-processes
those blobs (no new detection step) and computes three families of metrics.

---

## What to build

### 1. `core/glomerular.py`  (new file)

A pure analysis module. No GUI imports.

#### 1a. Blob filtering

```python
def filter_glomeruli(
    blobs: list,           # raw list from StrategyResult.extra["blobs"]
    E: np.ndarray,         # eosin channel (H×W float)
    lobe_mask: np.ndarray, # bool (H×W)
    boundary_um: float,    # fused blue strip depth — blobs shallower than
                           # this are inside the strip → discard
    config: Config,
) -> list:
    """
    Apply three sequential filters and return the surviving blobs.
    Each surviving blob gains two extra keys: 'circularity_cv' and 'kept'.

    Filter 1 — depth:
        Discard blobs with depth_um <= boundary_um.
        The blue strip is structure-free; anything shallower is a false positive.

    Filter 2 — size:
        Keep blobs whose r_um is in [config.sc_min_r_um, config.sc_max_r_um].
        (Reuses the Strategy C radius bounds — no new config fields needed here.)

    Filter 3 — circularity:
        For each surviving blob at pixel centre (cy, cx) with radius r_px:
          - Sample the eosin channel E at 24 evenly spaced points on a ring
            of radius 0.85 * r_px around the centre.
          - Clip sample coords to image bounds.
          - Compute CV = std(samples) / (mean(samples) + 1e-8).
          - Discard blobs with CV > config.gl_circ_cv_max.
        Rationale: true glomerular tufts have a fairly uniform eosin ring;
        tubular clusters and vessels are asymmetric (high CV).
    """
```

#### 1b. Approach 1 — Automated Radial Glomerular Count (aRGC)

```python
def compute_argc(
    glomeruli: list,          # filtered blob list
    lobe_mask: np.ndarray,
    dist_from_bg: np.ndarray,
    config: Config,
    n_contour_pts: int = 200,
) -> dict:
    """
    Direct automated equivalent of the Hinchliffe manual RGC.

    Algorithm:
      1. Find the convex-hull contour of lobe_mask (same as fusion.py).
      2. Sample n_contour_pts evenly spaced points along the contour.
      3. For each contour point p_i:
           a. Compute the inward normal (same logic as fusion._spatial_profile).
           b. Cast a ray inward from p_i.
           c. For each filtered glomerulus g, compute its perpendicular
              distance d_perp to the ray line:
                  d_perp = |cross(inward, g_centre - p_i)|
              where inward is the unit normal vector and g_centre = (cy, cx).
           d. Count glomeruli with d_perp * pixel_size_um <= config.gl_corridor_um / 2
              AND that lie on the inward side of p_i (dot product > 0).
           e. ray_count[i] = that count.
      4. Return:
           argc_mean   = mean(ray_count)
           argc_sd     = std(ray_count)
           argc_profile = ray_count array (length n_contour_pts)
           arc_um       = cumulative arc length array (for plotting)

    Note: The corridor half-width is config.gl_corridor_um / 2.
    A blob is assigned to a ray if its centre projects within that lateral
    distance. Blobs near the boundary of adjacent corridors may be counted
    by more than one ray — this matches the manual method where the pathologist
    counts every glomerulus that touches the line.
    """
```

#### 1c. Approach 2 — Absolute glomerular density

```python
def compute_density(
    glomeruli: list,
    lobe_mask: np.ndarray,
    strip_area_mm2: float,   # already in LobeMetrics
    config: Config,
) -> dict:
    """
    Returns:
        n_total          : int   — total filtered glomeruli in this lobe
        cortical_area_mm2: float — lobe area minus strip area
        density_per_mm2  : float — n_total / cortical_area_mm2
    """
    lobe_area_mm2 = lobe_mask.sum() * config.pixel_size_um**2 / 1e6
    cortical_area_mm2 = max(lobe_area_mm2 - strip_area_mm2, 1e-6)
    return {
        "n_total": len(glomeruli),
        "cortical_area_mm2": cortical_area_mm2,
        "density_per_mm2": len(glomeruli) / cortical_area_mm2,
    }
```

#### 1d. Approach 3 — Depth distribution and generation count

```python
def compute_depth_profile(
    glomeruli: list,
    config: Config,
) -> dict:
    """
    Bin glomeruli by depth_um to reveal maturational generations.

    Algorithm:
      1. Build a histogram of depth_um values with bin width
         config.gl_gen_bin_um.
      2. Smooth with a Gaussian (sigma = 1.5 bins).
      3. Find peaks in the smoothed histogram using scipy.signal.find_peaks
         with height >= config.gl_gen_min_count and distance >= 2 bins.
      4. generation_count = number of peaks found.
      5. generation_depths_um = list of peak centre depths (µm).

    Returns:
        depths_um          : list of float — raw blob depths (for scatter)
        hist_counts        : np.ndarray   — raw histogram counts
        hist_smoothed      : np.ndarray   — smoothed histogram
        bin_centres_um     : np.ndarray   — depth of each bin centre (µm)
        generation_count   : int
        generation_depths_um: list of float
    """
```

#### 1e. Top-level function

```python
@dataclass
class GlomerularResult:
    glomeruli: list            # filtered blob list (with circularity_cv added)
    n_raw: int                 # blobs before filtering
    n_filtered: int            # blobs after all filters
    argc: dict                 # output of compute_argc
    density: dict              # output of compute_density
    depth_profile: dict        # output of compute_depth_profile


def analyse_glomeruli(
    blobs: list,
    E: np.ndarray,
    lobe_mask: np.ndarray,
    dist_from_bg: np.ndarray,
    boundary_um: float,
    strip_area_mm2: float,
    config: Config,
) -> GlomerularResult:
    """
    Orchestrates filtering → density → aRGC → depth profile.
    Called by the runner after Strategy C completes.
    Returns GlomerularResult or None if blobs list is empty.
    """
```

---

### 2. `config.py`  (add four new fields to the Config dataclass)

```python
# ── Glomerular analysis ───────────────────────────────────────────────────────
gl_corridor_um:    float = 30.0   # aRGC ray corridor full-width (µm)
gl_circ_cv_max:    float = 0.35   # max eosin-ring CV to accept as glomerulus
gl_gen_bin_um:     float = 25.0   # histogram bin width for generation detection
gl_gen_min_count:  int   = 3      # minimum glomeruli per bin to call a generation
```

---

### 3. `gui/runner.py`  (call the new function after Strategy C)

Inside the per-lobe loop, after `fusion = fuse(...)`:

```python
from core.glomerular import analyse_glomeruli

# Extract raw blobs from Strategy C result
blobs_raw = next(
    (r.extra.get("blobs", []) for r in s_results if r.label == "C"), []
)

glom_result = None
if blobs_raw:
    self._progress(
        f"Lobe {lobe_idx + 1}/{n_lobes} — Glomerular analysis…",
        base_frac + 0.90 / n_lobes,
    )
    glom_result = analyse_glomeruli(
        blobs=blobs_raw,
        E=image_data.E,
        lobe_mask=lobe_mask,
        dist_from_bg=seg.dist_from_bg,
        boundary_um=fusion.boundary_um,
        strip_area_mm2=metrics.strip_area_mm2,
        config=cfg,
    )
```

Add `glom_result` to `LobeAnalysis`:

```python
@dataclass
class LobeAnalysis:
    lobe_label: int
    lobe_mask: np.ndarray
    strategy_results: List[StrategyResult]
    fusion: FusionResult
    metrics: LobeMetrics
    glom_result: "GlomerularResult | None" = None  # new field
```

---

### 4. `core/metrics.py`  (add glomerular fields to LobeMetrics)

Append to the `LobeMetrics` dataclass:

```python
# Glomerular metrics (None if Strategy C found no blobs)
gl_n_raw:              Optional[int]   = None
gl_n_filtered:         Optional[int]   = None
gl_density_per_mm2:    Optional[float] = None
gl_cortical_area_mm2:  Optional[float] = None
gl_argc_mean:          Optional[float] = None
gl_argc_sd:            Optional[float] = None
gl_generation_count:   Optional[int]   = None
```

Populate them in `compute_metrics()` if `glom_result is not None`:

```python
if glom_result is not None:
    metrics.gl_n_raw           = glom_result.n_raw
    metrics.gl_n_filtered      = glom_result.n_filtered
    metrics.gl_density_per_mm2 = glom_result.density["density_per_mm2"]
    metrics.gl_cortical_area_mm2 = glom_result.density["cortical_area_mm2"]
    metrics.gl_argc_mean       = glom_result.argc["argc_mean"]
    metrics.gl_argc_sd         = glom_result.argc["argc_sd"]
    metrics.gl_generation_count = glom_result.depth_profile["generation_count"]
```

Update `compute_metrics()` signature to accept `glom_result`:
```python
def compute_metrics(..., glom_result=None) -> LobeMetrics:
```

---

### 5. `gui/panels/param_panel.py`  (add glomerular parameter section)

Add a new section after "Strategy C — Blob Detection" using the existing
`_add_section` / `_add_row` pattern:

```python
self._add_section(inner, "Glomerular Analysis")
self._add_row(inner, "gl_corridor_um",   "Ray corridor (µm)",   30.0)
self._add_row(inner, "gl_circ_cv_max",   "Circularity CV max",  0.35)
self._add_row(inner, "gl_gen_bin_um",    "Generation bin (µm)", 25.0)
self._add_row(inner, "gl_gen_min_count", "Min per generation",  3)
```

Add tooltips to `PARAM_TIPS`:

```python
"gl_corridor_um": (
    "Full width of the ray corridor used for the\n"
    "automated radial glomerular count (aRGC), in µm.\n\n"
    "A ray is cast inward from each capsule point.\n"
    "Glomeruli within corridor/2 of the ray are counted.\n"
    "30 µm matches the typical manual method.\n"
    "Increase if counts seem too low at this magnification."
),
"gl_circ_cv_max": (
    "Maximum coefficient of variation (CV) of the eosin\n"
    "intensity sampled around the rim of each blob.\n\n"
    "True glomerular tufts have a fairly uniform eosin\n"
    "ring → low CV. Tubules and vessels are asymmetric\n"
    "→ high CV and are rejected.\n\n"
    "Lower = stricter (fewer, cleaner detections).\n"
    "Raise to 0.5 if too few glomeruli are found.\n"
    "Default 0.35 works well for standard H&E."
),
"gl_gen_bin_um": (
    "Bin width of the glomerular depth histogram used\n"
    "to detect distinct maturational generations, in µm.\n\n"
    "Each peak in the smoothed histogram = one generation\n"
    "of glomeruli at a characteristic cortical depth.\n"
    "25 µm is appropriate for 10× images."
),
"gl_gen_min_count": (
    "Minimum number of glomeruli per histogram bin\n"
    "to be recognised as a distinct generation peak.\n\n"
    "Prevents single outlier glomeruli from being\n"
    "counted as a generation. Raise if spurious extra\n"
    "generations appear; lower if real generations\n"
    "are being missed."
),
```

Also add the four new fields to `get_config()`:

```python
gl_corridor_um    = f("gl_corridor_um"),
gl_circ_cv_max    = f("gl_circ_cv_max"),
gl_gen_bin_um     = f("gl_gen_bin_um"),
gl_gen_min_count  = i("gl_gen_min_count"),
```

---

### 6. `gui/panels/result_panel.py`  (new "Glomeruli" tab)

Add the tab in `_build()` alongside the existing four tabs:

```python
for tab_name in ("Overview", "Profiles", "Fusion", "Spatial", "Glomeruli"):
```

Add `_draw_glomeruli(la: LobeAnalysis)` and call it from `_refresh()`.

The tab contains **three subplots** arranged as a 1×3 row:

#### Panel 1 — Filtered glomeruli overlay on H&E

- Start from `self._image_data.img.copy()`.
- Colour the blue strip zone (orange, as in the Fusion tab).
- Draw each **filtered** glomerulus as a circle outline (colour = green, lw=1.5).
- Draw each **rejected** blob (failed circularity or depth filter) as a small
  red cross `×` at its centre, so the user can see what was filtered out.
- Title: `f"Lobe {la.lobe_label} — {gr.n_filtered} glomeruli detected"`

#### Panel 2 — aRGC spatial bar chart

- X axis: arc length along capsule (mm).
- Bar chart: `gr.argc["arc_um"] / 1000` vs `gr.argc["argc_profile"]`.
- Horizontal dashed line at `gr.argc["argc_mean"]`, labelled with mean ± SD.
- This shows how glomerular density varies around the perimeter — focal
  depletion zones appear as dips.

#### Panel 3 — Depth histogram with generation peaks

- Bar chart of `gr.depth_profile["hist_counts"]` vs
  `gr.depth_profile["bin_centres_um"]`.
- Overlay the smoothed curve.
- Mark each generation peak with a vertical dashed line and label
  `f"Gen {i+1}: {depth:.0f} µm"`.
- X label: "Depth from capsule (µm)". Y label: "Glomerulus count".

Below the three subplots, add a one-line metrics summary as a text annotation
or `fig.text()`:
```
n=42  |  density=18.3/mm²  |  aRGC=4.1±0.8  |  generations=3
```

Update `_update_metrics()` to append glomerular rows to the metrics text box:

```python
if la.glom_result is not None:
    gr = la.glom_result
    lines += [
        f"  ── Glomeruli ──────────────────────",
        f"  Detected (filtered)  {gr.n_filtered:>6}",
        f"  Density (N/mm²)     {gr.density['density_per_mm2']:>8.1f}",
        f"  aRGC mean           {gr.argc['argc_mean']:>8.1f}",
        f"  aRGC SD             {gr.argc['argc_sd']:>8.1f}",
        f"  Generations         {gr.depth_profile['generation_count']:>6}",
    ]
```

---

## Testing checklist

After implementation, verify the following manually:

1. **Run on the test image** with default parameters. The Glomeruli tab should
   appear and populate without error.

2. **Filtered overlay (Panel 1)**:
   - Green circles should appear in the cortex, below the orange strip zone.
   - No green circles should be visible inside the orange zone.
   - Red crosses should mark rejected blobs; expect most rejections in the
     near-capsule zone (depth filter) and near the sinus (irregular shapes).

3. **aRGC bar chart (Panel 2)**:
   - Should be roughly flat if the strip is uniform around the perimeter.
   - Mean line should match the visual average of the bars.

4. **Depth histogram (Panel 3)**:
   - The distribution should be broad, starting just beyond the strip boundary
     and extending toward the medulla.
   - Generation peaks should be plausible (2–5 for a neonatal kidney).

5. **CSV export**: Re-run Export CSV and confirm the seven new `gl_*` columns
   are present in the output file.

6. **Edge cases**:
   - If Strategy C finds 0 blobs (e.g. very low `sc_blob_threshold`), the
     Glomeruli tab should display a "No glomeruli detected" message instead
     of crashing.
   - If all blobs fail the circularity filter (e.g. `gl_circ_cv_max = 0.0`),
     same graceful fallback.

7. **Parameter sensitivity**:
   - Increase `gl_circ_cv_max` to 0.7 → more green circles appear (less strict).
   - Decrease `gl_corridor_um` to 10 → aRGC mean should drop.
   - Both should update immediately on re-run without error.

---

## Implementation notes

- The `analyse_glomeruli()` function should return `None` (not raise) if
  `blobs` is empty. The runner and result panel must handle `None` gracefully.
- `core/glomerular.py` must have zero GUI imports.
- All depth values in the module work in µm; pixel conversion uses
  `config.pixel_size_um` wherever needed.
- The convex-hull contour extraction in `compute_argc` can be copied from
  `core/fusion._spatial_profile` — the same utility is needed.
  Consider extracting it to a shared helper in `core/segmentation.py` to
  avoid duplication.
- For the eosin ring sampling in the circularity filter, use
  `scipy.ndimage.map_coordinates` with order=1 interpolation for
  sub-pixel accuracy at all radii.
