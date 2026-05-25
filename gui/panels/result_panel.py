"""
gui/panels/result_panel.py
--------------------------
Right-side panel: tabbed matplotlib figures + metrics text + export button.
Figures are embedded via FigureCanvasTkAgg (no separate windows needed).
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Optional

import math

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle

from gui.runner import LobeAnalysis
from gui.panels.roi_panel import ROIPanel
from core.loader import ImageData
from core.segmentation import SegmentationResult
from core.metrics import export_csv


# ── Colour maps ───────────────────────────────────────────────────────────────
_CMAP_H = LinearSegmentedColormap.from_list("H", ["white", "#3a1a6e"])
_CMAP_E = LinearSegmentedColormap.from_list("E", ["white", "#C2185B"])
_STRAT_COLOURS = {"A": "#1565C0", "B": "#BF360C", "C": "#2E7D32"}


# ── Scale bar helpers ─────────────────────────────────────────────────────────

def _pick_scale_um(image_width_px: int, pixel_size_um: float) -> float:
    """Return a round scale-bar length (µm) that is ~15 % of the image width."""
    target_um = image_width_px * pixel_size_um * 0.15
    magnitude = 10 ** math.floor(math.log10(max(target_um, 1e-9)))
    norm = target_um / magnitude
    nice = 1 if norm < 1.5 else (2 if norm < 3.5 else (5 if norm < 7.5 else 10))
    return nice * magnitude


def _draw_scale_bar(ax, img_h: int, img_w: int, pixel_size_um: float) -> None:
    """Draw a white/black scale bar in the bottom-right corner of *ax*."""
    if pixel_size_um <= 0:
        return
    scale_um = _pick_scale_um(img_w, pixel_size_um)
    scale_px = scale_um / pixel_size_um

    # Positions in image-pixel data coordinates (y increases downward)
    pad_x = img_w * 0.03
    pad_y = img_h * 0.04
    x1 = img_w - pad_x - scale_px
    x2 = img_w - pad_x
    y  = img_h - pad_y

    # Bar: thick black outline, white fill on top
    for lw, col, zo in [(5, "black", 10), (3, "white", 11)]:
        ax.plot([x1, x2], [y, y], color=col, linewidth=lw,
                solid_capstyle="butt", transform=ax.transData, zorder=zo)

    # Label
    if scale_um < 1000:
        label = f"{int(scale_um)} µm"
    else:
        mm = scale_um / 1000
        label = f"{int(mm)} mm" if mm == int(mm) else f"{mm:.1f} mm"

    txt = ax.text(
        (x1 + x2) / 2, y - img_h * 0.012, label,
        color="white", ha="center", va="bottom",
        fontsize=8, fontweight="bold",
        transform=ax.transData, zorder=12,
    )
    txt.set_path_effects(
        [patheffects.withStroke(linewidth=2, foreground="black")]
    )


class ResultPanel(ttk.Frame):
    """
    Tabbed panel showing:
      Overview  — H&E, hematoxylin channel, depth map
      Profiles  — strategy A / B / C depth profiles (one subplot each)
      Fusion    — fused overlay + strategy bar chart
      Spatial   — d(arc) profile along the capsule
    Plus a metrics text box and Export CSV button.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._lobe_results: List[LobeAnalysis] = []
        self._image_data: Optional[ImageData] = None
        self._seg: Optional[SegmentationResult] = None
        self._csv_path: Optional[str] = None
        self._all_metrics = []
        self._roi_panel: Optional[ROIPanel] = None
        self._pixel_size_um: float = 1.0
        self._build()

    # ── Public API ────────────────────────────────────────────────────────────

    def show_image(self, image_data: ImageData, pixel_size_um: float = 1.0):
        """Called after the user selects an image. Arms the ROI selector."""
        self._image_data = image_data
        self._pixel_size_um = pixel_size_um
        self._roi_panel.set_image(image_data, pixel_size_um=pixel_size_um)
        self._draw_overview(image_data, seg=None, lobe_results=None)
        self._nb.select(self._tabs["ROI"])

    def get_roi_mask(self) -> Optional[np.ndarray]:
        """Return the current ROI mask from the selector, or None."""
        return self._roi_panel.get_roi_mask() if self._roi_panel else None

    def show_results(self, image_data: ImageData,
                     seg: SegmentationResult,
                     lobe_results: List[LobeAnalysis]):
        """Populate all tabs after analysis completes."""
        self._image_data = image_data
        self._seg = seg
        self._lobe_results = lobe_results
        self._all_metrics = [la.metrics for la in lobe_results]

        self._refresh()
        self._export_btn.configure(state="normal")
        self._save_figs_btn.configure(state="normal")
        self._nb.select(self._tabs["Overview"])

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        # ── Figure notebook (tabs) ────────────────────────────────────────────
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill="both", expand=True)

        self._tabs = {}

        # ── ROI tab (first — interactive polygon selector) ────────────────────
        roi_frame = ttk.Frame(self._nb)
        self._nb.add(roi_frame, text="ROI")
        self._tabs["ROI"] = roi_frame
        self._roi_panel = ROIPanel(roi_frame)
        self._roi_panel.pack(fill="both", expand=True)

        # ── Analysis result tabs ──────────────────────────────────────────────
        for tab_name in ("Overview", "Profiles", "Fusion", "Spatial", "Glomeruli"):
            frame = ttk.Frame(self._nb)
            self._nb.add(frame, text=tab_name)
            self._tabs[tab_name] = frame

        # Create one Figure per tab (analysis tabs only)
        self._figures = {}
        self._canvases = {}
        for tab_name, frame in {k: v for k, v in self._tabs.items()
                                 if k != "ROI"}.items():
            fig = plt.Figure(figsize=(7, 4), tight_layout=True)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            toolbar_frame = ttk.Frame(frame)
            toolbar_frame.pack(fill="x")
            NavigationToolbar2Tk(canvas, toolbar_frame)
            self._figures[tab_name] = fig
            self._canvases[tab_name] = canvas

        # ── Metrics text ──────────────────────────────────────────────────────
        metrics_frame = ttk.LabelFrame(self, text="Metrics", padding=6)
        metrics_frame.pack(fill="x", padx=4, pady=4)
        self._metrics_text = tk.Text(
            metrics_frame, height=14, state="disabled",
            font=("Courier", 9), relief="flat", bg="#F5F7FA",
        )
        self._metrics_text.pack(fill="x")

        # ── Export button ─────────────────────────────────────────────────────
        btn_frame = ttk.Frame(self, padding=(4, 0, 4, 4))
        btn_frame.pack(fill="x")
        self._export_btn = ttk.Button(
            btn_frame, text="Export CSV…",
            command=self._export_csv, state="disabled",
        )
        self._export_btn.pack(side="right")
        self._save_figs_btn = ttk.Button(
            btn_frame, text="Save figures…",
            command=self._on_save_figures, state="disabled",
        )
        self._save_figs_btn.pack(side="right", padx=(0, 4))

    # ── Refresh ───────────────────────────────────────────────────────────────

    def _refresh(self):
        if not self._lobe_results:
            return
        la = self._lobe_results[0]   # largest lobe (lobes are sorted by area)
        self._draw_overview(self._image_data, self._seg, self._lobe_results)
        self._draw_profiles(la)
        self._draw_fusion(la)
        self._draw_spatial(la)
        self._draw_glomeruli(la)
        self._update_metrics(la)

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_overview(self, image_data, seg, lobe_results):
        fig = self._figures["Overview"]
        fig.clear()
        axes = fig.subplots(1, 3)

        axes[0].imshow(image_data.img, vmin=0, vmax=1)
        axes[0].set_title("H&E")
        axes[0].axis("off")
        px = (lobe_results[0].metrics.pixel_size_um
              if lobe_results else self._pixel_size_um)
        img_h, img_w = image_data.img.shape[:2]
        _draw_scale_bar(axes[0], img_h, img_w, px)

        H_disp = np.clip(image_data.H, *np.percentile(image_data.H, [1, 99]))
        axes[1].imshow(H_disp, cmap=_CMAP_H)
        axes[1].set_title("Hematoxylin")
        axes[1].axis("off")

        if seg is not None:
            # Convert distance map to µm for display
            px = (lobe_results[0].metrics.pixel_size_um
                  if lobe_results else 1.0)
            depth_map_um = seg.dist_from_bg * px
            max_d_um = (lobe_results[0].fusion.boundary_um or 300.0
                        if lobe_results else 300.0)
            im = axes[2].imshow(
                np.where(seg.tissue_mask, depth_map_um, np.nan),
                cmap="magma", vmin=0, vmax=max_d_um,
            )
            fig.colorbar(im, ax=axes[2], label="depth (µm)", fraction=0.04)
        else:
            axes[2].text(0.5, 0.5, "Run analysis\nto see depth map",
                         ha="center", va="center", transform=axes[2].transAxes,
                         fontsize=10, color="grey")
        axes[2].set_title("Depth map")
        axes[2].axis("off")

        self._canvases["Overview"].draw()

    def _draw_profiles(self, la: LobeAnalysis):
        fig = self._figures["Profiles"]
        fig.clear()
        axes = fig.subplots(1, 3)

        for ax, sr in zip(axes, la.strategy_results):
            if len(sr.centres) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"Strategy {sr.label}")
                continue
            colour = _STRAT_COLOURS.get(sr.label, "steelblue")
            ax.plot(sr.centres, sr.profile, color=colour, alpha=0.4, lw=1.2)
            ax.plot(sr.centres, sr.smoothed, color=colour, lw=2)
            if not np.isnan(sr.boundary_um):
                ax.axvline(sr.boundary_um, color="orange", lw=2,
                           label=f"{sr.boundary_um:.0f} µm")
            ax.set_xlabel("Depth (µm)")
            ax.set_title(f"Strategy {sr.label} — {sr.name}")
            ax.legend(fontsize=8)

        self._canvases["Profiles"].draw()

    def _draw_fusion(self, la: LobeAnalysis):
        fig = self._figures["Fusion"]
        fig.clear()
        ax0, ax1 = fig.subplots(1, 2)

        # Left: fused overlay on H&E
        overlay = self._image_data.img.copy()
        if not np.isnan(la.fusion.boundary_um):
            px = la.metrics.pixel_size_um
            bnd_px = la.fusion.boundary_um / px
            strip = la.lobe_mask & (self._seg.dist_from_bg < bnd_px)
            overlay[strip] = overlay[strip] * 0.3 + np.array([1.0, 0.4, 0.1]) * 0.7
        ax0.imshow(overlay, vmin=0, vmax=1)
        ax0.set_title(f"Fused overlay  ({la.fusion.boundary_um:.1f} µm)")
        ax0.axis("off")

        # Right: bar chart of strategy estimates
        valid = [(l, d) for l, d in zip(la.fusion.strategy_labels,
                                         la.fusion.estimates)
                 if not np.isnan(d)]
        if valid:
            labels, depths = zip(*valid)
            colours = [_STRAT_COLOURS.get(l, "grey") for l in labels]
            bars = ax1.bar(labels, depths, color=colours, width=0.5,
                           edgecolor="grey", linewidth=0.5)
            ax1.axhline(la.fusion.boundary_um, color="orange", lw=2,
                        ls="--", label=f"Fused {la.fusion.boundary_um:.0f} µm")
            for bar, d in zip(bars, depths):
                ax1.text(bar.get_x() + bar.get_width()/2, d + 1,
                         f"{d:.0f}", ha="center", fontsize=9)
            ax1.set_ylabel("Depth (µm)")
            ax1.set_title("Strategy comparison")
            ax1.legend(fontsize=9)

        self._canvases["Fusion"].draw()

    def _draw_spatial(self, la: LobeAnalysis):
        fig = self._figures["Spatial"]
        fig.clear()
        ax = fig.add_subplot(111)

        arc = la.fusion.arc_um
        thick = la.fusion.thickness_um

        if len(arc) > 0:
            ax.plot(arc / 1000, thick, color="steelblue", lw=1.5)
            ax.fill_between(arc / 1000, thick, alpha=0.2, color="steelblue")
            ax.axhline(la.fusion.boundary_um, color="orange", lw=2, ls="--",
                       label=f"Fused {la.fusion.boundary_um:.1f} µm")
            if len(thick):
                ax.axhline(np.mean(thick), color="steelblue", lw=1.5, ls=":",
                           label=f"Profile mean {np.mean(thick):.1f} µm")
            ax.set_xlabel("Arc length along capsule (mm)")
            ax.set_ylabel("Blue strip thickness (µm)")
            ax.set_title("Spatial profile  d(arc)")
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, "Spatial profile unavailable",
                    ha="center", va="center", transform=ax.transAxes)

        self._canvases["Spatial"].draw()

    def _draw_glomeruli(self, la: LobeAnalysis):
        fig = self._figures["Glomeruli"]
        fig.clear()

        gr = la.glom_result
        if gr is None:
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5,
                "No glomeruli detected\n"
                "(Strategy C found no blobs — try lowering sc_blob_threshold)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="grey",
            )
            ax.axis("off")
            self._canvases["Glomeruli"].draw()
            return

        axes = fig.subplots(1, 3)
        px   = la.metrics.pixel_size_um

        # ── Panel 1: Filtered overlay on H&E ─────────────────────────────────
        overlay = self._image_data.img.copy()
        if not np.isnan(la.fusion.boundary_um):
            bnd_px = la.fusion.boundary_um / px
            strip  = la.lobe_mask & (self._seg.dist_from_bg < bnd_px)
            overlay[strip] = (overlay[strip] * 0.3
                              + np.array([1.0, 0.4, 0.1]) * 0.7)
        axes[0].imshow(overlay, vmin=0, vmax=1)

        sr_c      = next((sr for sr in la.strategy_results
                          if sr.label == "C"), None)
        all_blobs = sr_c.extra.get("blobs", []) if sr_c is not None else []
        accepted  = {(b["cy"], b["cx"]) for b in gr.glomeruli}

        for b in all_blobs:
            if (b["cy"], b["cx"]) not in accepted:
                axes[0].plot(b["cx"], b["cy"], "rx",
                             markersize=5, markeredgewidth=0.8, alpha=0.7)
        for b in gr.glomeruli:
            circ = Circle((b["cx"], b["cy"]), radius=b["r_px"],
                          fill=False, edgecolor="#2E7D32", linewidth=1.5)
            axes[0].add_patch(circ)

        axes[0].set_title(
            f"Lobe {la.lobe_label} — {gr.n_filtered} glomeruli detected")
        axes[0].axis("off")

        # ── Panel 2: aRGC spatial bar chart ──────────────────────────────────
        arc_mm  = gr.argc["arc_um"] / 1000
        profile = gr.argc["argc_profile"]
        mean_v  = gr.argc["argc_mean"]
        sd_v    = gr.argc["argc_sd"]

        if len(arc_mm) > 1 and arc_mm[-1] > 0:
            bar_w = arc_mm[-1] / len(arc_mm) * 0.9
            axes[1].bar(arc_mm, profile, width=bar_w,
                        color="steelblue", alpha=0.7, align="center")
            axes[1].axhline(mean_v, color="orange", lw=2, ls="--",
                            label=f"Mean {mean_v:.1f} ± {sd_v:.1f}")
            axes[1].set_xlabel("Arc length along capsule (mm)")
            axes[1].set_ylabel("Glomeruli per ray")
            axes[1].set_title("aRGC spatial profile")
            axes[1].legend(fontsize=8)
        else:
            axes[1].text(0.5, 0.5, "aRGC unavailable",
                         ha="center", va="center",
                         transform=axes[1].transAxes)

        # ── Panel 3: Depth histogram with generation peaks ────────────────────
        dp = gr.depth_profile
        if len(dp["bin_centres_um"]) > 0:
            bin_w = (float(dp["bin_centres_um"][1] - dp["bin_centres_um"][0])
                     if len(dp["bin_centres_um"]) > 1 else 25.0)
            axes[2].bar(dp["bin_centres_um"], dp["hist_counts"],
                        width=bin_w * 0.9, color="#2E7D32", alpha=0.6)
            axes[2].plot(dp["bin_centres_um"], dp["hist_smoothed"],
                         color="#2E7D32", lw=2)
            for i, d_um in enumerate(dp["generation_depths_um"]):
                axes[2].axvline(d_um, ls="--", color="orange", lw=1.5,
                                label=f"Gen {i + 1}: {d_um:.0f} µm")
            axes[2].set_xlabel("Depth from capsule (µm)")
            axes[2].set_ylabel("Glomerulus count")
            axes[2].set_title("Depth distribution")
            if dp["generation_depths_um"]:
                axes[2].legend(fontsize=8)
        else:
            axes[2].text(0.5, 0.5, "No glomeruli passed the filter",
                         ha="center", va="center",
                         transform=axes[2].transAxes)
            axes[2].set_title("Depth distribution")

        fig.text(
            0.5, 0.005,
            f"n={gr.n_filtered}  |  "
            f"density={gr.density['density_per_mm2']:.1f}/mm²  |  "
            f"aRGC={mean_v:.1f}±{sd_v:.1f}  |  "
            f"generations={dp['generation_count']}",
            ha="center", fontsize=9,
        )

        self._canvases["Glomeruli"].draw()

    # ── Metrics text ──────────────────────────────────────────────────────────

    def _update_metrics(self, la: LobeAnalysis):
        m = la.metrics
        def _fmt(v):
            if v is None:
                return f"{"N/A":>8}"
            return f"{v:>8.1f}"

        lines = [
            f"  Strategy A depth    {_fmt(m.depth_A_um)}  µm",
            f"  Strategy B depth    {_fmt(m.depth_B_um)}  µm",
            f"  Strategy C depth    {_fmt(m.depth_C_um)}  µm",
            f"  Fused depth (median)  {m.depth_fused_um:>8.1f}  µm",
            f"  Spatial mean          {m.spatial_mean_um:>8.1f}  µm",
            f"  CV                    {m.cv_pct:>8.1f}  %",
            f"  Strip area            {m.strip_area_mm2:>8.3f}  mm²",
        ]
        if la.glom_result is not None:
            gr = la.glom_result
            lines += [
                f"  ── Glomeruli ──────────────────────",
                f"  Blobs (Strategy C)   {gr.n_raw:>6}",
                f"  Detected (filtered)  {gr.n_filtered:>6}",
                f"  Density (N/mm²)     {gr.density['density_per_mm2']:>8.1f}",
                f"  aRGC mean           {gr.argc['argc_mean']:>8.1f}",
                f"  aRGC SD             {gr.argc['argc_sd']:>8.1f}",
                f"  Generations         {gr.depth_profile['generation_count']:>6}",
            ]

        self._metrics_text.configure(state="normal")
        self._metrics_text.delete("1.0", "end")
        self._metrics_text.insert("end", "\n".join(lines))
        self._metrics_text.configure(state="disabled")

    # ── CSV export ────────────────────────────────────────────────────────────

    def _export_csv(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            title="Save metrics CSV",
        )
        if not path:
            return
        try:
            export_csv(self._all_metrics, path)
            messagebox.showinfo("Exported",
                                f"Metrics saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    # ── Save all figures as PDF ───────────────────────────────────────────────

    def _on_save_figures(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf"), ("All files", "*.*")],
            title="Save all figures as PDF",
        )
        if not path:
            return
        try:
            self._build_pdf(path)
            messagebox.showinfo("Saved", f"Figures saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    def _build_pdf(self, path: str):
        la = self._lobe_results[0]
        px = la.metrics.pixel_size_um
        with PdfPages(path) as pdf:
            self._pdf_channels(pdf, px)
            self._pdf_segmentation(pdf, la, px)
            self._pdf_profiles(pdf, la)
            self._pdf_blobs(pdf, la)
            self._pdf_fusion(pdf, la)
            self._pdf_spatial(pdf, la)
            self._pdf_glomeruli(pdf, la)

    def _pdf_channels(self, pdf, px: float):
        img = self._image_data.img
        H_ch = self._image_data.H
        E_ch = self._image_data.E
        img_h, img_w = img.shape[:2]

        fig = plt.Figure(figsize=(13, 4.5), tight_layout=True)
        axes = fig.subplots(1, 3)

        axes[0].imshow(img, vmin=0, vmax=1)
        axes[0].set_title("H&E")
        axes[0].axis("off")
        _draw_scale_bar(axes[0], img_h, img_w, px)

        H_disp = np.clip(H_ch, *np.percentile(H_ch, [1, 99]))
        axes[1].imshow(H_disp, cmap=_CMAP_H)
        axes[1].set_title("Hematoxylin")
        axes[1].axis("off")
        _draw_scale_bar(axes[1], img_h, img_w, px)

        E_disp = np.clip(E_ch, *np.percentile(E_ch, [1, 99]))
        axes[2].imshow(E_disp, cmap=_CMAP_E)
        axes[2].set_title("Eosin")
        axes[2].axis("off")
        _draw_scale_bar(axes[2], img_h, img_w, px)

        pdf.savefig(fig)
        plt.close(fig)

    def _pdf_segmentation(self, pdf, la: LobeAnalysis, px: float):
        img = self._image_data.img
        seg = self._seg
        img_h, img_w = img.shape[:2]

        fig = plt.Figure(figsize=(13, 4.5), tight_layout=True)
        axes = fig.subplots(1, 3)

        axes[0].imshow(seg.tissue_mask, cmap="Greys_r")
        axes[0].set_title("Tissue mask")
        axes[0].axis("off")

        overlay = img.copy()
        lobe_colours = [
            [0.08, 0.47, 0.75],
            [0.75, 0.22, 0.05],
            [0.18, 0.49, 0.20],
            [0.55, 0.27, 0.07],
        ]
        for idx, la_i in enumerate(self._lobe_results):
            col = np.array(lobe_colours[idx % len(lobe_colours)])
            overlay[la_i.lobe_mask] = overlay[la_i.lobe_mask] * 0.4 + col * 0.6
        n = len(self._lobe_results)
        axes[1].imshow(overlay, vmin=0, vmax=1)
        axes[1].set_title(f"Lobes  ({n} detected)")
        axes[1].axis("off")
        _draw_scale_bar(axes[1], img_h, img_w, px)

        depth_map_um = seg.dist_from_bg * px
        max_d_um = (la.fusion.boundary_um * 1.5
                    if not np.isnan(la.fusion.boundary_um) else 300.0)
        im = axes[2].imshow(
            np.where(seg.tissue_mask, depth_map_um, np.nan),
            cmap="magma", vmin=0, vmax=max_d_um,
        )
        fig.colorbar(im, ax=axes[2], label="depth (µm)", fraction=0.04)
        axes[2].set_title("Depth map")
        axes[2].axis("off")

        pdf.savefig(fig)
        plt.close(fig)

    def _pdf_profiles(self, pdf, la: LobeAnalysis):
        fig = plt.Figure(figsize=(13, 4.5), tight_layout=True)
        axes = fig.subplots(1, 3)

        for ax, sr in zip(axes, la.strategy_results):
            if len(sr.centres) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"Strategy {sr.label}")
                continue
            colour = _STRAT_COLOURS.get(sr.label, "steelblue")
            ax.plot(sr.centres, sr.profile, color=colour, alpha=0.4, lw=1.2)
            ax.plot(sr.centres, sr.smoothed, color=colour, lw=2)
            if not np.isnan(sr.boundary_um):
                ax.axvline(sr.boundary_um, color="orange", lw=2,
                           label=f"{sr.boundary_um:.0f} µm")
            ax.set_xlabel("Depth (µm)")
            ax.set_title(f"Strategy {sr.label} — {sr.name}")
            ax.legend(fontsize=8)

        pdf.savefig(fig)
        plt.close(fig)

    def _pdf_blobs(self, pdf, la: LobeAnalysis):
        sr_c = next((sr for sr in la.strategy_results if sr.label == "C"), None)
        if sr_c is None or not sr_c.extra.get("blobs"):
            return

        blobs = sr_c.extra["blobs"]
        img = self._image_data.img

        fig = plt.Figure(figsize=(13, 5), tight_layout=True)
        ax0, ax1 = fig.subplots(1, 2)

        ax0.imshow(img, vmin=0, vmax=1)
        depths_b = [b["depth_um"] for b in blobs]
        max_d = max(depths_b) if depths_b else 1.0
        cmap_p = plt.cm.plasma
        norm_p = Normalize(vmin=0, vmax=max_d)
        for b in blobs:
            circ = Circle(
                (b["cx"], b["cy"]), radius=b["r_px"],
                fill=False, edgecolor=cmap_p(norm_p(b["depth_um"])),
                linewidth=0.8, alpha=0.85,
            )
            ax0.add_patch(circ)
        ax0.set_title(f"Strategy C — {len(blobs)} blobs")
        ax0.axis("off")
        sm = matplotlib.cm.ScalarMappable(cmap=cmap_p, norm=norm_p)
        sm.set_array([])
        fig.colorbar(sm, ax=ax0, label="depth (µm)", fraction=0.04)

        colour = _STRAT_COLOURS.get("C", "steelblue")
        if len(sr_c.centres):
            ax1.plot(sr_c.centres, sr_c.profile, color=colour, alpha=0.4, lw=1.2)
            ax1.plot(sr_c.centres, sr_c.smoothed, color=colour, lw=2)
        if not np.isnan(sr_c.boundary_um):
            ax1.axvline(sr_c.boundary_um, color="orange", lw=2,
                        label=f"Boundary  {sr_c.boundary_um:.0f} µm")
        ax1.set_xlabel("Depth (µm)")
        ax1.set_ylabel("Blob count")
        ax1.set_title("Strategy C — blob depth histogram")
        ax1.legend(fontsize=8)

        pdf.savefig(fig)
        plt.close(fig)

    def _pdf_fusion(self, pdf, la: LobeAnalysis):
        fig = plt.Figure(figsize=(13, 5), tight_layout=True)
        ax0, ax1 = fig.subplots(1, 2)

        img = self._image_data.img
        px = la.metrics.pixel_size_um
        overlay = img.copy()
        if not np.isnan(la.fusion.boundary_um):
            bnd_px = la.fusion.boundary_um / px
            strip = la.lobe_mask & (self._seg.dist_from_bg < bnd_px)
            overlay[strip] = overlay[strip] * 0.3 + np.array([1.0, 0.4, 0.1]) * 0.7
        ax0.imshow(overlay, vmin=0, vmax=1)
        ax0.set_title(f"Fused overlay  ({la.fusion.boundary_um:.1f} µm)")
        ax0.axis("off")

        valid = [(l, d) for l, d in zip(la.fusion.strategy_labels,
                                         la.fusion.estimates)
                 if not np.isnan(d)]
        if valid:
            labels, depths = zip(*valid)
            colours = [_STRAT_COLOURS.get(l, "grey") for l in labels]
            bars = ax1.bar(labels, depths, color=colours, width=0.5,
                           edgecolor="grey", linewidth=0.5)
            ax1.axhline(la.fusion.boundary_um, color="orange", lw=2,
                        ls="--", label=f"Fused {la.fusion.boundary_um:.0f} µm")
            for bar, d in zip(bars, depths):
                ax1.text(bar.get_x() + bar.get_width() / 2, d + 1,
                         f"{d:.0f}", ha="center", fontsize=9)
            ax1.set_ylabel("Depth (µm)")
            ax1.set_title("Strategy comparison")
            ax1.legend(fontsize=9)

        pdf.savefig(fig)
        plt.close(fig)

    def _pdf_spatial(self, pdf, la: LobeAnalysis):
        fig = plt.Figure(figsize=(9, 4.5), tight_layout=True)
        ax = fig.add_subplot(111)

        arc = la.fusion.arc_um
        thick = la.fusion.thickness_um

        if len(arc) > 0:
            ax.plot(arc / 1000, thick, color="steelblue", lw=1.5)
            ax.fill_between(arc / 1000, thick, alpha=0.2, color="steelblue")
            ax.axhline(la.fusion.boundary_um, color="orange", lw=2, ls="--",
                       label=f"Fused {la.fusion.boundary_um:.1f} µm")
            if len(thick):
                ax.axhline(np.mean(thick), color="steelblue", lw=1.5, ls=":",
                           label=f"Profile mean {np.mean(thick):.1f} µm")
            ax.set_xlabel("Arc length along capsule (mm)")
            ax.set_ylabel("Blue strip thickness (µm)")
            ax.set_title("Spatial profile  d(arc)")
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, "Spatial profile unavailable",
                    ha="center", va="center", transform=ax.transAxes)

        pdf.savefig(fig)
        plt.close(fig)

    def _pdf_glomeruli(self, pdf, la: LobeAnalysis):
        gr = la.glom_result
        px = la.metrics.pixel_size_um

        fig = plt.Figure(figsize=(13, 5), tight_layout=True)

        if gr is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "No glomeruli detected\n"
                    "(Strategy C found no blobs)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="grey")
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
            return

        axes = fig.subplots(1, 3)

        # ── Panel 1: filtered overlay on H&E ─────────────────────────────────
        overlay = self._image_data.img.copy()
        if not np.isnan(la.fusion.boundary_um):
            bnd_px = la.fusion.boundary_um / px
            strip  = la.lobe_mask & (self._seg.dist_from_bg < bnd_px)
            overlay[strip] = overlay[strip] * 0.3 + np.array([1.0, 0.4, 0.1]) * 0.7
        axes[0].imshow(overlay, vmin=0, vmax=1)

        sr_c      = next((sr for sr in la.strategy_results if sr.label == "C"), None)
        all_blobs = sr_c.extra.get("blobs", []) if sr_c is not None else []
        accepted  = {(b["cy"], b["cx"]) for b in gr.glomeruli}

        for b in all_blobs:
            if (b["cy"], b["cx"]) not in accepted:
                axes[0].plot(b["cx"], b["cy"], "rx",
                             markersize=5, markeredgewidth=0.8, alpha=0.7)
        for b in gr.glomeruli:
            circ = Circle((b["cx"], b["cy"]), radius=b["r_px"],
                          fill=False, edgecolor="#2E7D32", linewidth=1.5)
            axes[0].add_patch(circ)
        axes[0].set_title(f"Lobe {la.lobe_label} — {gr.n_filtered} glomeruli detected")
        axes[0].axis("off")

        # ── Panel 2: aRGC spatial bar chart ──────────────────────────────────
        arc_mm  = gr.argc["arc_um"] / 1000
        profile = gr.argc["argc_profile"]
        mean_v  = gr.argc["argc_mean"]
        sd_v    = gr.argc["argc_sd"]

        if len(arc_mm) > 1 and arc_mm[-1] > 0:
            bar_w = arc_mm[-1] / len(arc_mm) * 0.9
            axes[1].bar(arc_mm, profile, width=bar_w,
                        color="steelblue", alpha=0.7, align="center")
            axes[1].axhline(mean_v, color="orange", lw=2, ls="--",
                            label=f"Mean {mean_v:.1f} ± {sd_v:.1f}")
            axes[1].set_xlabel("Arc length along capsule (mm)")
            axes[1].set_ylabel("Glomeruli per ray")
            axes[1].set_title("aRGC spatial profile")
            axes[1].legend(fontsize=8)
        else:
            axes[1].text(0.5, 0.5, "aRGC unavailable",
                         ha="center", va="center", transform=axes[1].transAxes)

        # ── Panel 3: depth histogram with generation peaks ────────────────────
        dp = gr.depth_profile
        if len(dp["bin_centres_um"]) > 0:
            bin_w = (float(dp["bin_centres_um"][1] - dp["bin_centres_um"][0])
                     if len(dp["bin_centres_um"]) > 1 else 25.0)
            axes[2].bar(dp["bin_centres_um"], dp["hist_counts"],
                        width=bin_w * 0.9, color="#2E7D32", alpha=0.6)
            axes[2].plot(dp["bin_centres_um"], dp["hist_smoothed"],
                         color="#2E7D32", lw=2)
            for i, d_um in enumerate(dp["generation_depths_um"]):
                axes[2].axvline(d_um, ls="--", color="orange", lw=1.5,
                                label=f"Gen {i + 1}: {d_um:.0f} µm")
            axes[2].set_xlabel("Depth from capsule (µm)")
            axes[2].set_ylabel("Glomerulus count")
            axes[2].set_title("Depth distribution")
            if dp["generation_depths_um"]:
                axes[2].legend(fontsize=8)
        else:
            axes[2].text(0.5, 0.5, "No glomeruli passed the filter",
                         ha="center", va="center", transform=axes[2].transAxes)
            axes[2].set_title("Depth distribution")

        fig.text(
            0.5, 0.005,
            f"n={gr.n_filtered}  |  "
            f"density={gr.density['density_per_mm2']:.1f}/mm²  |  "
            f"aRGC={mean_v:.1f}±{sd_v:.1f}  |  "
            f"generations={dp['generation_count']}",
            ha="center", fontsize=9,
        )

        pdf.savefig(fig)
        plt.close(fig)
