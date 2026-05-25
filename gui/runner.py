"""
gui/runner.py
-------------
Runs the full analysis pipeline in a background thread.
Communicates with the GUI via a thread-safe Queue of ProgressMessage objects.
The GUI polls the queue with root.after() — no direct widget updates from here.
"""
import threading
import traceback
from dataclasses import dataclass
from queue import Queue
from typing import List, Optional

import numpy as np

from config import Config
from core.loader import load_image, ImageData
from core.segmentation import segment, SegmentationResult
from core.strategies.intensity import StrategyIntensity
from core.strategies.density import StrategyDensity
from core.strategies.blobs import StrategyBlobs
from core.strategies.base import StrategyResult
from core.fusion import fuse, FusionResult
from core.glomerular import analyse_glomeruli, GlomerularResult
from core.metrics import compute_metrics, LobeMetrics, export_csv


# ── Message types sent from runner → GUI ─────────────────────────────────────

@dataclass
class ProgressMessage:
    step: str           # short status text shown in the status bar
    fraction: float     # 0.0 – 1.0 for the progress bar


@dataclass
class ResultMessage:
    """Sent once when analysis is fully complete."""
    image_data: ImageData
    seg: SegmentationResult
    lobe_results: list      # list of LobeAnalysis


@dataclass
class ErrorMessage:
    error: str


@dataclass
class LobeAnalysis:
    """All outputs for one lobe."""
    lobe_label: int
    lobe_mask: np.ndarray
    strategy_results: List[StrategyResult]
    fusion: FusionResult
    metrics: LobeMetrics
    glom_result: Optional[GlomerularResult] = None


# ── Runner ────────────────────────────────────────────────────────────────────

class AnalysisRunner(threading.Thread):
    """
    Background thread that runs the full pipeline and puts messages
    into `queue` for the GUI to consume.

    Usage:
        runner = AnalysisRunner(config, queue)
        runner.start()
        # GUI polls queue with root.after()
    """

    # Registered strategies — add Strategy D here without changing anything else
    STRATEGIES = [StrategyIntensity(), StrategyDensity(), StrategyBlobs()]

    def __init__(self, config: Config, queue: Queue,
                 roi_mask: Optional[np.ndarray] = None):
        super().__init__(daemon=True)
        self.config   = config
        self.queue    = queue
        self.roi_mask = roi_mask

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _progress(self, step: str, fraction: float):
        self.queue.put(ProgressMessage(step=step, fraction=fraction))

    def _result(self, image_data, seg, lobe_results):
        self.queue.put(ResultMessage(
            image_data=image_data, seg=seg, lobe_results=lobe_results))

    def _error(self, msg: str):
        self.queue.put(ErrorMessage(error=msg))

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run(self):
        try:
            cfg = self.config
            n_strategies = len(self.STRATEGIES)

            # ── Step 1: load image ────────────────────────────────────────────
            self._progress("Loading image…", 0.05)
            image_data = load_image(cfg.image_path)

            # ── Step 2: segment ───────────────────────────────────────────────
            self._progress("Detecting tissue and lobes…", 0.15)
            seg = segment(image_data, cfg, roi_mask=self.roi_mask)

            if not seg.lobes:
                self._error("No lobes detected. Try lowering Min Lobe Fraction.")
                return

            lobe_results = []
            n_lobes = len(seg.lobes)

            for lobe_idx, lobe_prop in enumerate(seg.lobes):
                lobe_label = lobe_prop.label
                lobe_mask  = seg.lobe_labels == lobe_label
                base_frac  = 0.15 + (lobe_idx / n_lobes) * 0.75

                # ── Strategies ────────────────────────────────────────────────
                s_results = []
                for s_idx, strategy in enumerate(self.STRATEGIES):
                    frac = base_frac + (s_idx / n_strategies) * (0.75 / n_lobes)
                    self._progress(
                        f"Lobe {lobe_idx + 1}/{n_lobes} — "
                        f"Strategy {strategy.label} ({strategy.name})…",
                        frac,
                    )
                    result = strategy.run(image_data, lobe_mask,
                                          seg.dist_from_bg, cfg)
                    s_results.append(result)

                # ── Fusion ────────────────────────────────────────────────────
                self._progress(
                    f"Lobe {lobe_idx + 1}/{n_lobes} — Fusion…",
                    base_frac + 0.75 / n_lobes,
                )
                fusion = fuse(s_results, lobe_mask, seg.dist_from_bg, cfg)

                # ── Glomerular analysis ───────────────────────────────────────
                blobs_raw = next(
                    (r.extra.get("blobs", []) for r in s_results
                     if r.label == "C"), []
                )
                # Pre-compute strip area so analyse_glomeruli can use it
                # without requiring a prior compute_metrics call.
                _px = cfg.pixel_size_um
                _bnd_px = (fusion.boundary_um / _px
                           if not np.isnan(fusion.boundary_um) else 0.0)
                _strip_mm2 = float(
                    (lobe_mask & (seg.dist_from_bg < _bnd_px)).sum()
                    * _px ** 2 / 1e6
                )
                glom_result = None
                if blobs_raw:
                    self._progress(
                        f"Lobe {lobe_idx + 1}/{n_lobes} — Glomerular analysis…",
                        min(base_frac + 0.90 / n_lobes, 0.98),
                    )
                    glom_result = analyse_glomeruli(
                        blobs=blobs_raw,
                        E=image_data.E,
                        lobe_mask=lobe_mask,
                        dist_from_bg=seg.dist_from_bg,
                        boundary_um=fusion.boundary_um,
                        strip_area_mm2=_strip_mm2,
                        config=cfg,
                    )

                # ── Metrics ───────────────────────────────────────────────────
                metrics = compute_metrics(
                    image_path=cfg.image_path,
                    lobe_label=lobe_label,
                    lobe_mask=lobe_mask,
                    dist_from_bg=seg.dist_from_bg,
                    strategy_results=s_results,
                    fusion=fusion,
                    config=cfg,
                    glom_result=glom_result,
                )
                lobe_results.append(LobeAnalysis(
                    lobe_label=lobe_label,
                    lobe_mask=lobe_mask,
                    strategy_results=s_results,
                    fusion=fusion,
                    metrics=metrics,
                    glom_result=glom_result,
                ))

            self._progress("Analysis complete.", 1.0)
            self._result(image_data, seg, lobe_results)

        except Exception:
            self._error(traceback.format_exc())
