"""
gui/panels/param_panel.py
-------------------------
Left-side scrollable panel containing all parameter inputs.
Reads from and writes to a Config instance.
"""
import tkinter as tk
from tkinter import ttk, filedialog
from config import Config



class _Tooltip:
    """Show a small floating label when hovering over a widget."""

    def __init__(self, widget: "tk.Widget", text: str) -> None:
        self._tip: "tk.Toplevel | None" = None
        widget.bind("<Enter>", lambda e: self._show(e, text))
        widget.bind("<Leave>", lambda _e: self._hide())

    def _show(self, event: "tk.Event", text: str) -> None:
        self._hide()
        tip = tk.Toplevel()
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{event.x_root + 12}+{event.y_root + 6}")
        tk.Label(
            tip, text=text, background="#FFFFCC",
            relief="solid", borderwidth=1,
            font=("TkDefaultFont", 9),
            wraplength=220, justify="left",
            padx=4, pady=2,
        ).pack()
        self._tip = tip

    def _hide(self) -> None:
        if self._tip:
            self._tip.destroy()
            self._tip = None


class ParamPanel(ttk.Frame):
    """
    Scrollable frame with grouped parameter inputs.
    Call get_config() to retrieve a Config with current values.
    Call set_enabled(bool) to lock/unlock during analysis.
    """

    def __init__(self, parent, on_run_callback, on_image_selected=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._on_run = on_run_callback
        self._on_image_selected = on_image_selected
        self._vars = {}     # name → tk.Variable
        self._entries = {}  # name → widget (for enable/disable)
        self._build()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_config(self) -> Config:
        """Build a Config from current widget values."""
        def f(name):
            try:
                return float(self._vars[name].get())
            except (ValueError, KeyError):
                return float(Config.__dataclass_fields__[name].default)

        def i(name):
            try:
                return int(self._vars[name].get())
            except (ValueError, KeyError):
                return int(Config.__dataclass_fields__[name].default)

        return Config(
            image_path=self._vars["image_path"].get(),
            pixel_size_um=f("pixel_size_um"),
            min_lobe_frac=f("min_lobe_frac"),
            close_um=f("close_um"),
            shell_thickness_um=f("shell_thickness_um"),
            max_depth_um=f("max_depth_um"),
            profile_smooth_sigma=f("profile_smooth_sigma"),
            sc_min_r_um=f("sc_min_r_um"),
            sc_max_r_um=f("sc_max_r_um"),
            sc_n_sigma=i("sc_n_sigma"),
            sc_blob_threshold=f("sc_blob_threshold"),
            sc_max_overlap=f("sc_max_overlap"),
            sc_depth_percentile=i("sc_depth_percentile"),
            sc_min_depth_um=f("sc_min_depth_um"),
            gl_corridor_um=f("gl_corridor_um"),
            gl_circ_cv_max=f("gl_circ_cv_max"),
            gl_gen_bin_um=f("gl_gen_bin_um"),
            gl_gen_min_count=i("gl_gen_min_count"),
        )

    def set_enabled(self, enabled: bool):
        """Lock all inputs and the run button during analysis."""
        state = "normal" if enabled else "disabled"
        for w in self._entries.values():
            try:
                w.configure(state=state)
            except tk.TclError:
                pass
        self._run_btn.configure(state=state)

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build(self):
        # Outer canvas + scrollbar for vertical scrolling
        canvas = tk.Canvas(self, width=220, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical",
                                  command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas, padding=8)
        inner_id = canvas.create_window((0, 0), window=inner,
                                         anchor="nw")

        def _on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(inner_id, width=canvas.winfo_width())

        inner.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", _on_configure)

        # Mouse-wheel scrolling — cross-platform
        # macOS:   event.delta is ±1
        # Windows: event.delta is ±120
        # Linux:   uses Button-4 / Button-5 events
        def _on_wheel(event):
            if event.delta:   # macOS / Windows
                direction = -1 if event.delta > 0 else 1
            else:             # Linux
                direction = -1 if event.num == 4 else 1
            canvas.yview_scroll(direction, "units")

        canvas.bind_all("<MouseWheel>", _on_wheel)   # macOS / Windows
        canvas.bind_all("<Button-4>", _on_wheel)     # Linux scroll up
        canvas.bind_all("<Button-5>", _on_wheel)     # Linux scroll down

        # ── File picker ───────────────────────────────────────────────────────
        self._add_section(inner, "Image")
        self._add_file_row(inner, "image_path", "Image file", "")

        # ── Acquisition ───────────────────────────────────────────────────────
        self._add_section(inner, "Acquisition")
        self._add_row(inner, "pixel_size_um", "Pixel size (µm/px)", 1.0,
                      tip="Check your scanner spec: 0.25(40×) 0.5(20×) 1.0(10×)")

        # ── Tissue detection ──────────────────────────────────────────────────
        self._add_section(inner, "Tissue Detection")
        self._add_row(inner, "min_lobe_frac", "Min lobe fraction", 0.005,
                      tip="Lower if lobes are missed (e.g. 0.001)")
        self._add_row(inner, "close_um", "Closing radius (µm)", 5.0,
                      tip="Lower if separate lobes are merged")

        # ── Analysis ──────────────────────────────────────────────────────────
        self._add_section(inner, "Analysis")
        self._add_row(inner, "shell_thickness_um", "Shell thickness (µm)", 10.0)
        self._add_row(inner, "max_depth_um", "Max depth (µm)", 300.0)
        self._add_row(inner, "profile_smooth_sigma", "Smooth sigma", 2.0,
                      tip="Gaussian smoothing of depth profiles (shell units)")

        # ── Strategy C ────────────────────────────────────────────────────────
        self._add_section(inner, "Strategy C — Blob Detection")
        self._add_row(inner, "sc_min_r_um", "Min radius (µm)", 35.0)
        self._add_row(inner, "sc_max_r_um", "Max radius (µm)", 110.0)
        self._add_row(inner, "sc_blob_threshold", "Blob threshold", 0.003,
                      tip="Lower for more blobs; raise to reduce noise")
        self._add_row(inner, "sc_min_depth_um", "Min blob depth (µm)", 20.0,
                      tip="Discard blobs closer than this to any boundary")

        # ── Glomerular analysis ───────────────────────────────────────────────
        self._add_section(inner, "Glomerular Analysis")
        self._add_row(
            inner, "gl_corridor_um", "Ray corridor (µm)", 30.0,
            tip=(
                "Full width of the inward-normal ray corridor used for the "
                "automated radial glomerular count (aRGC), in µm.\n\n"
                "Glomeruli within corridor/2 of the ray are counted. "
                "30 µm matches the typical manual method. "
                "Increase if counts seem too low at this magnification."
            ),
        )
        self._add_row(
            inner, "gl_circ_cv_max", "Circularity CV max", 0.50,
            tip=(
                "Maximum coefficient of variation (CV) of the eosin "
                "intensity sampled around the rim of each blob.\n\n"
                "True glomerular tufts have a fairly uniform eosin ring "
                "→ low CV. Tubules and vessels are asymmetric → high CV "
                "and are rejected.\n\n"
                "Lower = stricter (fewer, cleaner detections).\n"
                "Raise to 0.7 if too few glomeruli are found.\n"
                "Default 0.5 works well for standard H&E."
            ),
        )
        self._add_row(
            inner, "gl_gen_bin_um", "Generation bin (µm)", 25.0,
            tip=(
                "Bin width of the glomerular depth histogram used to "
                "detect distinct maturational generations, in µm.\n\n"
                "Each peak in the smoothed histogram = one generation "
                "of glomeruli at a characteristic cortical depth. "
                "25 µm is appropriate for 10× images."
            ),
        )
        self._add_row(
            inner, "gl_gen_min_count", "Min per generation", 3,
            tip=(
                "Minimum number of glomeruli per histogram bin to be "
                "recognised as a distinct generation peak.\n\n"
                "Prevents single outlier glomeruli from being counted "
                "as a generation. Raise if spurious extra generations "
                "appear; lower if real generations are being missed."
            ),
        )

        # ── Run button ────────────────────────────────────────────────────────
        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=8)
        self._run_btn = ttk.Button(
            inner, text="▶  Run Analysis",
            command=self._on_run, style="Accent.TButton",
        )
        self._run_btn.pack(fill="x", pady=(0, 4))
        self._run_btn.configure(state="disabled")  # enabled after image loaded

    # ── Row builders ──────────────────────────────────────────────────────────

    def _add_section(self, parent, title: str):
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=(8, 2))
        ttk.Label(parent, text=title,
                  font=("TkDefaultFont", 9, "bold")).pack(anchor="w")

    def _add_row(self, parent, name: str, label: str, default,
                 tip: str = ""):
        var = tk.StringVar(value=str(default))
        self._vars[name] = var
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=1)
        lbl = ttk.Label(frame, text=label, width=20, wraplength=130,
                        justify="left")
        lbl.pack(side="left")
        entry = ttk.Entry(frame, textvariable=var, width=9)
        entry.pack(side="right")
        self._entries[name] = entry
        if tip:
            _Tooltip(entry, tip)

    def _add_file_row(self, parent, name: str, label: str, default: str):
        var = tk.StringVar(value=default)
        self._vars[name] = var
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        ttk.Label(frame, text=label).pack(anchor="w")
        inner = ttk.Frame(frame)
        inner.pack(fill="x")
        entry = ttk.Entry(inner, textvariable=var)
        entry.pack(side="left", fill="x", expand=True)
        self._entries[name] = entry
        btn = ttk.Button(inner, text="…", width=3,
                         command=lambda: self._browse(var))
        btn.pack(side="right")

    def _browse(self, var: tk.StringVar):
        path = filedialog.askopenfilename(
            title="Select H&E image",
            filetypes=[
                ("Images", "*.png *.tif *.tiff *.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )
        if path:
            var.set(path)
            self._run_btn.configure(state="normal")
            if self._on_image_selected is not None:
                self._on_image_selected(path)

