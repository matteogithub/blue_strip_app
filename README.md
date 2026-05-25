# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
# Create and activate the virtual environment (first time only)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Launch
source .venv/bin/activate
python main.py
```

There are no automated tests. Verification is manual: run the app, load an H&E image, and exercise the affected feature.

## Architecture

The codebase enforces a hard two-layer split:

- **`core/`** — pure analysis, zero Tkinter imports. Can be driven from a notebook or CLI.
- **`gui/`** — pure presentation, zero analysis code. Imports from `core/` only.

All parameters live in the single `Config` dataclass (`config.py`). The GUI constructs one `Config` per run and passes it unchanged through the entire pipeline. There are no globals and no environment-variable reads.

### Pipeline flow

```
load_image() → segment() → [StrategyA, StrategyB, StrategyC] → fuse() → analyse_glomeruli() → compute_metrics()
```

Each step produces a typed dataclass result. The runner (`gui/runner.py`) executes the pipeline in a daemon thread and communicates back to the GUI exclusively via a `queue.Queue` of `ProgressMessage / ResultMessage / ErrorMessage` objects. The main thread polls with `root.after(100)` — nothing in `core/` ever touches a Tkinter widget.

### Strategy pattern

`core/strategies/base.py` defines `BaseStrategy` (ABC) with a single `run(image_data, lobe_mask, dist_from_bg, config) → StrategyResult` interface.

**To add a new strategy:**
1. Create `core/strategies/my_strategy.py` implementing `BaseStrategy`.
2. Append an instance to `AnalysisRunner.STRATEGIES` in `gui/runner.py` — that is the only other file that needs to change.
3. Optionally add a subplot to `result_panel._draw_profiles()` and a depth field to `LobeMetrics` / `compute_metrics()`.

### Distance map invariant

`dist_from_bg` is stored and passed in **pixels** throughout the entire core. Conversion to µm (`× config.pixel_size_um`) is applied only at display time and in `compute_metrics()`. Never convert earlier.

### Lobe labels

`skimage.regionprops` label integers are arbitrary and may have gaps. Always look up the actual label value — never use `label - 1` arithmetic.

### Glomerular quantification

`core/glomerular.py` post-processes the blob list from `StrategyResult.extra["blobs"]` (produced by Strategy C). It does not run new image analysis. `analyse_glomeruli()` returns `None` when the blob list is empty; every caller must handle `None` gracefully.

The runner pre-computes `strip_area_mm2` from `fusion.boundary_um` and `dist_from_bg` before calling `analyse_glomeruli()`, so no prior `compute_metrics()` call is needed.

### GUI panels

`gui/panels/result_panel.py` shows only `lobe_results[0]` (the largest lobe) in all analysis tabs. The CSV export includes all lobes. There is intentionally no lobe selector widget yet.

`gui/panels/roi_panel.py` embeds a matplotlib `PolygonSelector`. `get_roi_mask()` returns a `bool (H×W)` union of all drawn polygons, or `None` (full image) if nothing was drawn.

The scale-bar helpers (`_pick_scale_um`, `_draw_scale_bar`) and `_Tooltip` are duplicated between panel files to avoid a circular import (`result_panel` imports `roi_panel`).

### PDF export

`result_panel._build_pdf()` writes one page per `_pdf_*` method. The page order is: channels → segmentation → profiles → blobs → fusion → spatial → glomeruli. Adding a new page requires one new `_pdf_*` method and one call in `_build_pdf`.
