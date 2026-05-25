"""
gui/panels/roi_panel.py
-----------------------
Interactive multi-polygon ROI selector embedded as a ttk.Frame.
Uses matplotlib.widgets.PolygonSelector for vertex-based drawing.
No analysis code — pure GUI.
"""
import tkinter as tk
from tkinter import ttk
from typing import List, Optional

import math

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib import patheffects
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import PolygonSelector
from skimage.draw import polygon as sk_polygon

from core.loader import ImageData


# ── Scale bar helpers (duplicated from result_panel — no shared import to avoid
#    circular dependency since result_panel imports roi_panel) ─────────────────

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

    pad_x = img_w * 0.03
    pad_y = img_h * 0.04
    x1 = img_w - pad_x - scale_px
    x2 = img_w - pad_x
    y  = img_h - pad_y

    for lw, col, zo in [(5, "black", 10), (3, "white", 11)]:
        ax.plot([x1, x2], [y, y], color=col, linewidth=lw,
                solid_capstyle="butt", transform=ax.transData, zorder=zo)

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


class ROIPanel(ttk.Frame):
    """
    Embeds a matplotlib figure with a live PolygonSelector.

    Workflow
    --------
    1. Call set_image() after loading an image.
    2. User clicks vertices on the image, then clicks the first vertex to
       close the polygon.  The filled patch appears.
    3. User may drag vertices to refine the polygon.
    4. "New polygon" button seals the current polygon and starts a fresh one.
    5. Repeat from (2) to add more inclusion regions.
    6. Call get_roi_mask() before running analysis.

    Public API
    ----------
    set_image(image_data)  — display image and arm a fresh selector
    get_roi_mask()         — bool (H×W) union of all polygons, or None
    """

    _SEALED_FACE  = "#1565C0"   # blue fill for sealed polygons
    _CURRENT_FACE = "#1565C0"
    _ALPHA_SEALED  = 0.30
    _ALPHA_CURRENT = 0.18

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._sealed_verts:  List[np.ndarray]    = []
        self._sealed_patches: List[MplPolygon]   = []
        self._current_verts:  Optional[np.ndarray] = None
        self._current_patch:  Optional[MplPolygon] = None
        self._selector:       Optional[PolygonSelector] = None
        self._img_shape:      Optional[tuple]    = None   # (H, W)
        self._pixel_size_um:  float = 1.0
        self._build()

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_image(self, image_data: ImageData,
                  pixel_size_um: float = 1.0) -> None:
        """Display the image, clear all existing ROI state, arm the selector."""
        self._img_shape = image_data.img.shape[:2]
        self._pixel_size_um = pixel_size_um
        self._reset_state()

        img_h, img_w = self._img_shape
        self._ax.clear()
        self._ax.imshow(image_data.img, vmin=0, vmax=1)
        self._ax.set_title(
            "Click to place vertices  ·  close polygon by clicking the first vertex  ·  "
            "drag to adjust",
            fontsize=8,
        )
        self._ax.axis("off")
        _draw_scale_bar(self._ax, img_h, img_w, pixel_size_um)
        self._canvas.draw()

        self._arm_selector()
        self._sync_ui()

    def get_roi_mask(self) -> Optional[np.ndarray]:
        """
        Return a bool (H×W) array that is True inside the union of all
        defined polygons (sealed + any in-progress with ≥ 3 vertices).
        Returns None if nothing has been drawn — callers treat None as
        "use the full image".
        """
        if self._img_shape is None:
            return None
        all_verts = list(self._sealed_verts)
        if self._current_verts is not None and len(self._current_verts) >= 3:
            all_verts.append(self._current_verts)
        if not all_verts:
            return None

        H, W = self._img_shape
        mask = np.zeros((H, W), dtype=bool)
        for verts in all_verts:
            xs = verts[:, 0]   # matplotlib data coords: x = column
            ys = verts[:, 1]   # y = row
            # skimage.draw.polygon(row_coords, col_coords, shape)
            rr, cc = sk_polygon(ys, xs, shape=(H, W))
            mask[rr, cc] = True
        return mask if mask.any() else None

    # ── Build UI ────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        # ── Canvas ────────────────────────────────────────────────────────────
        self._fig = Figure(figsize=(6, 5), tight_layout=True)
        self._ax  = self._fig.add_subplot(111)
        self._ax.set_facecolor("#CCCCCC")
        self._ax.text(
            0.5, 0.5, "Load an image to begin",
            ha="center", va="center", transform=self._ax.transAxes,
            fontsize=11, color="grey",
        )
        self._ax.axis("off")

        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(fill="x")
        NavigationToolbar2Tk(self._canvas, toolbar_frame)

        # ── Button / status bar ───────────────────────────────────────────────
        bar = ttk.Frame(self, padding=(4, 3))
        bar.pack(fill="x")

        self._new_btn = ttk.Button(
            bar, text="New polygon",
            command=self._seal_and_new, state="disabled",
        )
        self._new_btn.pack(side="left", padx=(0, 4))
        _Tooltip(self._new_btn,
                 "Seal the current polygon and start drawing a new one.")

        self._remove_btn = ttk.Button(
            bar, text="Remove last",
            command=self._remove_last, state="disabled",
        )
        self._remove_btn.pack(side="left", padx=(0, 4))

        self._clear_btn = ttk.Button(
            bar, text="Clear all",
            command=self._clear_all, state="disabled",
        )
        self._clear_btn.pack(side="left", padx=(0, 4))

        self._status_var = tk.StringVar(
            value="No ROI drawn — full image will be used.")
        ttk.Label(bar, textvariable=self._status_var,
                  foreground="#444444").pack(side="left", padx=6)

    # ── Selector management ────────────────────────────────────────────────────

    def _arm_selector(self) -> None:
        """Disconnect any existing selector and create a fresh one."""
        if self._selector is not None:
            try:
                self._selector.disconnect_events()
            except Exception:
                pass
            self._selector = None
        self._selector = PolygonSelector(
            self._ax,
            onselect=self._on_polygon_updated,
            props=dict(color="#1565C0", linestyle="-",
                       linewidth=1.5, alpha=0.7),
            handle_props=dict(
                markeredgecolor="#1565C0",
                markerfacecolor="#90CAF9",
                markersize=6,
            ),
        )

    def _on_polygon_updated(self, verts) -> None:
        """
        Called by PolygonSelector when the polygon is closed for the first
        time, and again on every subsequent vertex drag.
        We update _current_verts and redraw the translucent fill patch.
        """
        if len(verts) < 3:
            return
        self._current_verts = np.array(verts)
        self._redraw_current_patch()
        self._sync_ui()

    def _seal_and_new(self) -> None:
        """
        Lock the current polygon permanently and arm a new selector.
        Called by the 'New polygon' button.
        """
        if self._current_verts is not None and len(self._current_verts) >= 3:
            self._sealed_verts.append(self._current_verts)
            # Promote the current patch to a sealed one (darker alpha)
            if self._current_patch is not None:
                self._current_patch.set_alpha(self._ALPHA_SEALED)
                self._sealed_patches.append(self._current_patch)
                self._current_patch = None
        self._current_verts = None
        # Defer selector creation to avoid re-entrancy in the event stack
        self.after(1, self._arm_selector)
        self._sync_ui()

    # ── Patch helpers ──────────────────────────────────────────────────────────

    def _redraw_current_patch(self) -> None:
        """Replace the translucent fill patch for the in-progress polygon."""
        if self._current_patch is not None:
            try:
                self._current_patch.remove()
            except Exception:
                pass
            self._current_patch = None

        if self._current_verts is not None and len(self._current_verts) >= 3:
            patch = MplPolygon(
                self._current_verts,
                closed=True,
                facecolor=self._CURRENT_FACE,
                alpha=self._ALPHA_CURRENT,
                edgecolor="none",
                transform=self._ax.transData,
            )
            self._ax.add_patch(patch)
            self._current_patch = patch

        self._canvas.draw_idle()

    # ── Remove / clear ─────────────────────────────────────────────────────────

    def _remove_last(self) -> None:
        """Remove the in-progress polygon, or the last sealed one."""
        if self._current_verts is not None:
            # Cancel the in-progress polygon and restart the selector
            self._current_verts = None
            if self._current_patch is not None:
                try:
                    self._current_patch.remove()
                except Exception:
                    pass
                self._current_patch = None
            self._canvas.draw_idle()
            self.after(1, self._arm_selector)
        elif self._sealed_verts:
            self._sealed_verts.pop()
            patch = self._sealed_patches.pop()
            try:
                patch.remove()
            except Exception:
                pass
            self._canvas.draw_idle()
        self._sync_ui()

    def _clear_all(self) -> None:
        """Remove all polygons and reset to a fresh selector."""
        self._reset_state()
        self._arm_selector()
        self._canvas.draw_idle()
        self._sync_ui()

    def _reset_state(self) -> None:
        """Disconnect selector and remove all patches — does NOT re-arm."""
        if self._selector is not None:
            try:
                self._selector.disconnect_events()
            except Exception:
                pass
            self._selector = None
        for patch in self._sealed_patches:
            try:
                patch.remove()
            except Exception:
                pass
        if self._current_patch is not None:
            try:
                self._current_patch.remove()
            except Exception:
                pass
        self._sealed_verts.clear()
        self._sealed_patches.clear()
        self._current_verts  = None
        self._current_patch  = None

    # ── UI sync ────────────────────────────────────────────────────────────────

    def _sync_ui(self) -> None:
        """Update button states and status label to reflect current ROI state."""
        has_current = (self._current_verts is not None
                       and len(self._current_verts) >= 3)
        n_sealed    = len(self._sealed_verts)
        has_any     = has_current or n_sealed > 0
        loaded      = self._img_shape is not None

        self._new_btn.configure(
            state="normal" if (has_current and loaded) else "disabled")
        self._remove_btn.configure(
            state="normal" if (has_any and loaded) else "disabled")
        self._clear_btn.configure(
            state="normal" if (has_any and loaded) else "disabled")

        if not has_any:
            self._status_var.set(
                "No ROI drawn — full image will be used.")
        else:
            parts = []
            if n_sealed:
                s = "s" if n_sealed > 1 else ""
                parts.append(f"{n_sealed} sealed polygon{s}")
            if has_current:
                parts.append("1 in progress")
            self._status_var.set(
                "ROI active: " + ", ".join(parts) + ".")


class _Tooltip:
    """Small floating label shown on widget hover."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self._tip: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", lambda e: self._show(e, text))
        widget.bind("<Leave>", lambda _e: self._hide())

    def _show(self, event: tk.Event, text: str) -> None:
        self._hide()
        tip = tk.Toplevel()
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{event.x_root + 12}+{event.y_root + 6}")
        tk.Label(
            tip, text=text, background="#FFFFCC",
            relief="solid", borderwidth=1,
            font=("TkDefaultFont", 9),
            wraplength=240, justify="left",
            padx=4, pady=2,
        ).pack()
        self._tip = tip

    def _hide(self) -> None:
        if self._tip:
            self._tip.destroy()
            self._tip = None
