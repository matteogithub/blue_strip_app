# Blue Strip App — Architecture

## Purpose

Automated quantification of the **nephrogenic blue strip** depth in neonatal
kidney H&E histology sections.

The blue strip is a subcapsular band of densely packed mesenchymal progenitor
cells that stains intensely with hematoxylin. Its thickness is a direct proxy
for residual nephrogenic potential: a thick strip indicates active nephron
formation; a thin or absent strip indicates depletion. The goal is to measure
this depth automatically, without manual annotation and without deep learning.

---

## Key geometric insight

The blue strip wraps around the **entire outer perimeter** of each renal lobe
as a concentric shell — it is not a horizontal band. All depth measurements
use `scipy.ndimage.distance_transform_edt`, which assigns every tissue pixel
its Euclidean distance to the nearest background pixel, i.e. its depth below
the outer capsule, regardless of lobe shape or orientation.

Internal cavities (renal sinus, large vessels) are filled with
`binary_fill_holes` before computing the transform so that sinus-wall pixels
receive their correct outer-capsule depth rather than a spurious near-zero
depth from the adjacent cavity.

---

## Three strategies

Each strategy detects a different aspect of the same biological transition
(dense progenitor zone → maturing cortex). They are expected to be ordered
**A < B < C** in depth, and are fused by median.

### Strategy A — Hematoxylin intensity profile
Signal: mean hematoxylin optical density per concentric depth shell.
Boundary: first significant negative inflection of the smoothed profile.
The blue strip produces a near-capsule plateau of high H signal that drops
as tubular lumens and glomerular spaces dilute the hematoxylin density.
Typical depth: **65–80 µm**.

### Strategy B — Nuclear area fraction
Signal: fraction of pixels above the Otsu H-threshold per depth shell.
No nucleus segmentation — robust at any magnification.
The dense progenitor zone has minimal cytoplasm so almost all pixels are
nuclear (high fraction); maturing cortex has lumens and cytoplasm (lower
fraction). Boundary: first drop after the near-capsule peak.
Typical depth: **85–100 µm**.

### Strategy C — LoG blob distribution
Signal: eosin-channel LoG blobs (glomeruli and rounded tubular structures),
each assigned a depth from the distance map.
Boundary: the valley between the near-capsule non-glomerular cluster and
the main glomerular depth distribution (histogram valley detection).
The blue strip is definitionally structure-free; the first reliable glomeruli
mark its lower edge.
Typical depth: **100–120 µm**.

---

## File structure

```
blue_strip_app/
│
├── main.py                 Entry point. Sets matplotlib backend to TkAgg,
│                           inserts app root on sys.path, launches App.
│
├── config.py               Single Config dataclass — all parameters with
│                           defaults. Created by the GUI and passed unchanged
│                           through the entire pipeline. No global variables.
│                           Sections: Acquisition, Tissue/Lobe Detection,
│                           Depth-Shell Analysis, Strategy C, Glomerular Analysis.
│
├── requirements.txt        Minimum version pins for all dependencies.
│
├── core/                   Pure analysis layer. Zero GUI imports.
│   │
│   ├── loader.py           load_image() → ImageData
│   │                       Dtype-aware normalisation (uint8/uint16/float).
│   │                       Ruifrok-Johnston colour deconvolution → H, E channels.
│   │
│   ├── segmentation.py     segment() → SegmentationResult
│   │                       Otsu threshold → morphological cleanup →
│   │                       connected-component labelling → interior fragment
│   │                       absorption → binary_fill_holes → distance transform.
│   │
│   ├── fusion.py           fuse() → FusionResult
│   │                       Median of valid strategy estimates.
│   │                       Spatial profile d(arc): convex-hull contour sampled
│   │                       at 200 points, inward-normal ray casting per point.
│   │
│   ├── glomerular.py       analyse_glomeruli() → GlomerularResult (or None)
│   │                       Post-processes Strategy C blobs; no new detection.
│   │                       filter_glomeruli(): depth / size / circularity.
│   │                       compute_argc(): automated radial glomerular count.
│   │                       compute_density(): N / cortical area (mm²).
│   │                       compute_depth_profile(): depth histogram +
│   │                       maturational generation peak detection.
│   │
│   ├── metrics.py          compute_metrics() → LobeMetrics
│   │                       export_csv() — appends one row per lobe per image;
│   │                       writes header only if the file is new/empty.
│   │                       LobeMetrics contains seven optional gl_* fields
│   │                       (None when no blobs were detected by Strategy C).
│   │
│   └── strategies/
│       ├── base.py         BaseStrategy ABC + StrategyResult dataclass.
│       │                   Shared helpers: _find_boundary(), _shell_edges().
│       │                   _find_boundary() is NaN-guarded (safe after region
│       │                   exclusion removes shallow shells).
│       ├── intensity.py    Strategy A
│       ├── density.py      Strategy B
│       └── blobs.py        Strategy C — includes _find_valley() for histogram
│                           bimodality detection.
│                           extra["blobs"] stores {cy, cx, r_px, depth_um}
│                           dicts consumed by core/glomerular.py.
│
└── gui/                    GUI layer. Imports from core only.
    │
    ├── app.py              App(tk.Tk) — main window.
    │                       Assembles ParamPanel + ResultPanel in a PanedWindow.
    │                       Owns the queue and the after()-based polling loop.
    │                       on_run() → locks UI → starts AnalysisRunner →
    │                       polls queue every 100 ms → dispatches result/error.
    │
    ├── runner.py           AnalysisRunner(threading.Thread).
    │                       Runs the full pipeline in a daemon thread.
    │                       Communicates via queue.Queue of typed message
    │                       objects (ProgressMessage, ResultMessage, ErrorMessage).
    │                       Per-lobe loop order: strategies → fusion →
    │                       analyse_glomeruli → compute_metrics → LobeAnalysis.
    │                       LobeAnalysis dataclass holds lobe_label, lobe_mask,
    │                       strategy_results, fusion, metrics, glom_result.
    │                       Registered strategies list — adding Strategy D
    │                       requires only one new file and one list entry here.
    │
    └── panels/
        ├── param_panel.py  ParamPanel — scrollable left panel.
        │                   All parameter inputs grouped by section.
        │                   Sections: Image, Acquisition, Tissue Detection,
        │                   Analysis, Strategy C — Blob Detection,
        │                   Glomerular Analysis.
        │                   get_config() → Config. set_enabled() locks/unlocks
        │                   during analysis. _Tooltip class shows hover hints.
        │                   Cross-platform mousewheel: sign-only delta logic +
        │                   Button-4/5 bindings for Linux.
        │                   on_image_selected callback fires after file pick,
        │                   triggering immediate image load + ROI display.
        │
        ├── roi_panel.py    ROIPanel — interactive multi-polygon ROI selector.
        │                   Embeds a matplotlib PolygonSelector.
        │                   set_image() displays the H&E and arms the selector.
        │                   User clicks vertices to build each polygon, closes
        │                   by clicking the first vertex, then presses "New
        │                   polygon" to seal it and start the next.
        │                   get_roi_mask() → bool (H×W) union of all polygons,
        │                   or None (full image used when no ROI is drawn).
        │                   Uses skimage.draw.polygon for fast rasterisation.
        │
        └── result_panel.py ResultPanel — right panel.
                            ttk.Notebook with six embedded tabs:
                            ROI (polygon selector — first tab, shown on load),
                            Overview (H&E · H-channel · depth map),
                            Profiles (A · B · C depth profiles),
                            Fusion (overlay · bar chart),
                            Spatial (d(arc) along capsule perimeter),
                            Glomeruli (filtered overlay · aRGC · depth histogram).
                            Export CSV via core/metrics.export_csv().
                            Save figures via PdfPages (6-page PDF).
                            get_roi_mask() delegates to ROIPanel.
```

---

## Glomerular quantification

`core/glomerular.py` post-processes the blob list produced by Strategy C.
It adds three independent measurements without running any new image analysis.

### Blob filtering (`filter_glomeruli`)

Three sequential filters are applied to the raw blobs:

1. **Depth** — discard blobs with `depth_um ≤ boundary_um`.
   The blue strip is definitionally structure-free; anything shallower than
   the fused boundary estimate is a false positive.

2. **Size** — keep blobs whose radius (in µm) is within
   `[sc_min_r_um, sc_max_r_um]`. Reuses the Strategy C detection bounds;
   no new config fields needed.

3. **Circularity** — sample the eosin channel `E` at 24 evenly spaced
   points on a ring of radius `0.85 × r_px` using `scipy.ndimage.map_coordinates`
   (order-1 sub-pixel interpolation). Compute
   `CV = std(samples) / (mean(samples) + 1e-8)`.
   Discard blobs with `CV > gl_circ_cv_max`.
   True glomerular tufts have a fairly uniform eosin ring (low CV);
   tubular clusters and vessels are asymmetric (high CV).

Surviving blobs gain `circularity_cv` and `kept=True` keys.

### Automated Radial Glomerular Count (`compute_argc`)

A direct computational equivalent of the Hinchliffe manual RGC method:

1. Extract the convex-hull contour of the lobe mask (identical to
   `fusion._spatial_profile` — same contour, same inward-normal logic).
2. Sample 200 evenly spaced points along the contour.
3. For each capsule point, cast an inward-normal ray and count all
   filtered glomeruli whose centres lie:
   - on the inward side of the capsule (dot product with inward > 0), and
   - within `gl_corridor_um / 2` of the ray in the perpendicular direction
     (cross-product magnitude ≤ half-corridor width in pixels).
4. Report `argc_mean` and `argc_sd` over all 200 rays, plus the full
   per-ray `argc_profile` array for the spatial bar chart.

A glomerulus near the boundary of adjacent corridors may be counted by
more than one ray — this intentionally matches the manual method.

### Absolute density (`compute_density`)

`density_per_mm2 = n_filtered / cortical_area_mm2`

where `cortical_area_mm2 = max(lobe_area_mm2 − strip_area_mm2, 1e-6)`.
The strip area is already computed by `compute_metrics`; `analyse_glomeruli`
receives it as a parameter so no repeated calculation is needed.

### Depth profile and generation count (`compute_depth_profile`)

1. Histogram of `depth_um` values with bin width `gl_gen_bin_um`.
2. Gaussian smooth (σ = 1.5 bins).
3. `scipy.signal.find_peaks` with `height ≥ gl_gen_min_count` and
   `distance ≥ 2 bins`.
4. `generation_count` = number of peaks found.

Each peak corresponds to a cohort of glomeruli at a characteristic
cortical depth — a proxy for distinct maturational generations laid down
at different times during nephrogenesis.

### Runner integration

Inside the per-lobe loop in `AnalysisRunner.run()`, after `fuse()`:

```
strategies → fusion → (strip_area_mm2 inline) →
analyse_glomeruli → compute_metrics(glom_result=…) → LobeAnalysis
```

The strip area is computed from `fusion.boundary_um` and `dist_from_bg`
directly in the runner (two lines) so that `analyse_glomeruli` can receive
it without requiring a prior `compute_metrics` call.
`compute_metrics` accepts the completed `GlomerularResult` and populates
the seven `gl_*` fields in `LobeMetrics` in one place.

`analyse_glomeruli` returns `None` when the blob list is empty.
Every downstream consumer (runner, result panel, metrics) handles `None`
gracefully, so the rest of the pipeline is unaffected when Strategy C
finds no blobs.

---

## Design decisions

**Separation of concerns.**
`core/` has zero Tkinter imports; `gui/` has zero analysis code. Either layer
can be tested, replaced, or reused independently. The core can be driven from
a notebook, a CLI script, or a different GUI framework without modification.

**Single Config dataclass.**
All parameters live in one place with explicit defaults. The GUI instantiates
one Config per run and passes it down; nothing reads environment variables or
module-level globals.

**Strategy pattern.**
`BaseStrategy` defines a single `run(image_data, lobe_mask, dist_from_bg,
config) → StrategyResult` interface. Each strategy is a self-contained class.
Adding a new strategy requires creating one file and appending one entry to
`AnalysisRunner.STRATEGIES` — no other file changes.

**Thread safety via queue.**
Analysis runs in a daemon thread. The thread never touches Tkinter widgets
directly; it puts typed message objects into a `queue.Queue`. The main thread
polls with `root.after(100, poll_queue)` and dispatches updates. This keeps
the GUI responsive and avoids all race conditions.

**Matplotlib embedded, not floating.**
Figures are created as `matplotlib.figure.Figure` objects (thread-safe, no
Tk calls) and displayed via `FigureCanvasTkAgg` in `ttk.Notebook` tabs. No
`plt.show()` calls. Each tab has a `NavigationToolbar2Tk` for zoom/pan.

**Convex-hull contour for spatial profile.**
`find_contours(lobe_mask)` traces both the outer capsule and the inner sinus
walls as one combined path. The spatial profile instead uses
`find_contours(convex_hull_image(lobe_mask))`, which traces only the smooth
outer envelope and bridges the sinus mouth — giving a clean
capsule-only sampling line for the ray-casting step.

**Distance map in pixels, displayed in µm.**
`dist_from_bg` is stored in pixels throughout the pipeline for numerical
convenience. Conversion to µm (`× pixel_size_um`) is applied only at display
time and in the metrics layer. This prevents double-conversion bugs.

**Lobe labels are not assumed contiguous.**
`skimage.regionprops` label integers are arbitrary (they reflect the original
connected-component numbering, which may have gaps after filtering). The GUI
stores `_selected_label` (the actual integer label) and uses a `next()`
lookup — never `label - 1` arithmetic.

---

## Adding a new strategy (checklist)

1. Create `core/strategies/my_strategy.py` implementing `BaseStrategy`.
2. Append an instance to `AnalysisRunner.STRATEGIES` in `gui/runner.py`.
3. Add a subplot to `result_panel._draw_profiles()` if a custom profile view
   is needed (optional — the default three-subplot layout auto-iterates).
4. Add the new depth field to `core/metrics.LobeMetrics` and
   `compute_metrics()`.

No other files require modification.

---

## ROI mask flow

1. User selects a file → `ParamPanel` fires `on_image_selected(path)` →
   `App._on_image_selected` loads the image synchronously and calls
   `ResultPanel.show_image()`, which arms `ROIPanel`.
2. User draws polygons → `ROIPanel` stores sealed + in-progress vertices.
3. User clicks Run → `App._on_run` calls `result_panel.get_roi_mask()` and
   passes the bool mask to `AnalysisRunner(roi_mask=...)`.
4. `AnalysisRunner.run()` passes `roi_mask` to `segment()`.
5. `segment()` applies it at two points:
   - After Otsu (step 1): `mask = mask & roi_mask` — only ROI pixels can
     become tissue.
   - After `binary_fill_holes` (step 5): `tissue_filled = tissue_filled & roi_mask`
     — the fill runs on the full per-lobe geometry (so the renal sinus is
     correctly handled), then the ROI clips the result before the EDT.
     Pixels at the ROI boundary get `dist_from_bg = 0` (they adjoin the new
     artificial background edge).

If `get_roi_mask()` returns None (no polygons drawn), the full image is used
and the pipeline is unchanged.

---

## Known limitations (v1)

- **Single image only.** Batch processing not implemented.
- **Renal sinus contamination.** Tissue pixels adjacent to the sinus have
  near-zero depth and may appear in shallow shells for Strategies A and B.
  The convex-hull contour mitigates this in the spatial profile but not in
  the shell-mean statistics.
- **Fixed stain vectors.** Ruifrok-Johnston deconvolution is used without
  stain normalisation. Significant batch-to-batch staining variation may
  require Macenko/Vahadane normalisation as a preprocessing step.
- **Strategy C at low magnification.** The LoG detector captures all rounded
  eosin-positive structures, not exclusively glomeruli. Valley detection
  handles the resulting bimodal distribution but may be unreliable if the two
  populations overlap.
- **Glomerular count depends on Strategy C quality.** `filter_glomeruli`
  post-processes whatever blobs Strategy C produced; it cannot recover
  glomeruli that were missed at detection time (e.g. threshold too high,
  magnification too low, or glomeruli outside the radius bounds).
- **aRGC corridor overlap.** A glomerulus near the boundary between two
  adjacent ray corridors is counted by both rays, matching the manual
  convention but inflating the absolute count relative to a non-overlapping
  partition. This is expected behaviour.
- **Circularity filter is eosin-dependent.** The CV test relies on the eosin
  channel being well-separated from hematoxylin. Under-eosin-stained sections
  may require a higher `gl_circ_cv_max` threshold.
- **Generation count is heuristic.** Peak detection in the depth histogram
  depends on `gl_gen_min_count` and `gl_gen_bin_um`. Very dense or sparse
  glomerular distributions may need manual threshold tuning.
