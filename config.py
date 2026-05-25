"""
config.py
---------
Single configuration dataclass passed through the entire pipeline.
All default values live here and nowhere else.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Acquisition ───────────────────────────────────────────────────────────
    image_path: str = ""
    pixel_size_um: float = 1.0          # µm per pixel (check scanner spec sheet)

    # ── Tissue / lobe detection ───────────────────────────────────────────────
    min_lobe_frac: float = 0.005        # min lobe area as fraction of image area
    close_um: float = 5.0               # morphological closing radius (µm)

    # ── Depth-shell analysis (all strategies) ─────────────────────────────────
    shell_thickness_um: float = 10.0    # concentric shell thickness (µm)
    max_depth_um: float = 300.0         # maximum search depth from capsule (µm)
    profile_smooth_sigma: float = 2.0   # Gaussian smoothing (shell units)


    # ── Strategy C — LoG blob detection ──────────────────────────────────────
    sc_min_r_um: float = 35.0
    sc_max_r_um: float = 110.0
    sc_n_sigma: int = 12
    sc_blob_threshold: float = 0.003
    sc_max_overlap: float = 0.40
    sc_depth_percentile: int = 15
    sc_min_depth_um: float = 20.0

    # ── Glomerular analysis ───────────────────────────────────────────────────
    gl_corridor_um:   float = 30.0   # aRGC ray corridor full-width (µm)
    gl_circ_cv_max:   float = 0.50   # max eosin-ring CV to accept as glomerulus
    gl_gen_bin_um:    float = 25.0   # histogram bin width for generation detection
    gl_gen_min_count: int   = 3      # minimum glomeruli per bin to call a generation
