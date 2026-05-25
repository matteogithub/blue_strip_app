"""
gui/app.py
----------
Main application window. Assembles ParamPanel and ResultPanel,
wires the Run button to AnalysisRunner, and polls the queue.
"""
import queue
import tkinter as tk
from tkinter import ttk, messagebox

from config import Config
from gui.runner import (AnalysisRunner, ProgressMessage,
                        ResultMessage, ErrorMessage)
from gui.panels.param_panel import ParamPanel
from gui.panels.result_panel import ResultPanel
from core.loader import load_image


class App(tk.Tk):
    """
    Root window. Layout:
      ┌──────────────┬───────────────────────────────┐
      │  ParamPanel  │       ResultPanel              │
      ├──────────────┴───────────────────────────────┤
      │  progress bar  │  status label               │
      └─────────────────────────────────────────────┘
    """

    POLL_MS = 100   # queue poll interval (ms)

    def __init__(self):
        super().__init__()
        self.title("Blue Strip Depth Analysis")
        self.geometry("1100x680")
        self.minsize(900, 600)
        self._queue: queue.Queue = queue.Queue()
        self._runner: AnalysisRunner | None = None
        self._build()
        self._apply_theme()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        # ── Main pane ─────────────────────────────────────────────────────────
        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True)

        # Left: parameter panel
        self._param_panel = ParamPanel(
            pane,
            on_run_callback=self._on_run,
            on_image_selected=self._on_image_selected,
            width=240,
        )
        pane.add(self._param_panel, weight=0)

        # Right: result panel
        self._result_panel = ResultPanel(pane)
        pane.add(self._result_panel, weight=1)

        # ── Status bar ────────────────────────────────────────────────────────
        status_bar = ttk.Frame(self, relief="sunken", padding=(4, 2))
        status_bar.pack(fill="x", side="bottom")

        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(
            status_bar, variable=self._progress_var,
            maximum=1.0, length=200, mode="determinate",
        )
        self._progress_bar.pack(side="left", padx=(0, 8))

        self._status_var = tk.StringVar(value="Open an image to begin.")
        ttk.Label(status_bar, textvariable=self._status_var,
                  anchor="w").pack(side="left", fill="x", expand=True)

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _apply_theme(self):
        """Apply a clean, modern look using ttk styles."""
        style = ttk.Style(self)
        # Use the most modern available theme as base
        available = style.theme_names()
        for preferred in ("clam", "alt", "default"):
            if preferred in available:
                style.theme_use(preferred)
                break

        style.configure(".", font=("TkDefaultFont", 10))
        style.configure("TLabelframe.Label", font=("TkDefaultFont", 9, "bold"))

        # Accent button style for Run
        style.configure("Accent.TButton",
                         font=("TkDefaultFont", 10, "bold"),
                         foreground="white", background="#0D7377")
        style.map("Accent.TButton",
                  background=[("active", "#0a5c60"), ("disabled", "#aaaaaa")])

    # ── Image selection ───────────────────────────────────────────────────────

    def _on_image_selected(self, path: str):
        """
        Called from the file picker as soon as an image is chosen.
        Loads the image synchronously (< 2 s for typical histology files)
        and hands it to the ROI panel so the user can draw before running.
        """
        self._status("Loading image…", 0.0)
        self.update_idletasks()
        try:
            image_data = load_image(path)
        except Exception as exc:
            self._status("Error loading image.", 0.0)
            messagebox.showerror("Image load error", str(exc))
            return
        pixel_size_um = self._param_panel.get_config().pixel_size_um
        self._result_panel.show_image(image_data, pixel_size_um=pixel_size_um)
        self._status("Image loaded — draw ROI (optional), then click ▶ Run Analysis.", 0.0)

    # ── Run analysis ──────────────────────────────────────────────────────────

    def _on_run(self):
        config = self._param_panel.get_config()

        if not config.image_path:
            messagebox.showwarning("No image", "Please select an image first.")
            return

        roi_mask = self._result_panel.get_roi_mask()

        # Lock UI
        self._param_panel.set_enabled(False)
        self._status("Loading…", 0.0)

        # Start background runner
        self._runner = AnalysisRunner(config, self._queue, roi_mask=roi_mask)
        self._runner.start()

        # Start polling
        self.after(self.POLL_MS, self._poll_queue)

    # ── Queue polling ─────────────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                msg = self._queue.get_nowait()
                if isinstance(msg, ProgressMessage):
                    self._status(msg.step, msg.fraction)

                elif isinstance(msg, ResultMessage):
                    self._on_analysis_done(msg)
                    return   # stop polling

                elif isinstance(msg, ErrorMessage):
                    self._on_analysis_error(msg.error)
                    return   # stop polling

        except queue.Empty:
            pass

        # Keep polling if runner is still alive
        if self._runner and self._runner.is_alive():
            self.after(self.POLL_MS, self._poll_queue)

    # ── Completion handlers ───────────────────────────────────────────────────

    def _on_analysis_done(self, msg: ResultMessage):
        self._status("Analysis complete.", 1.0)
        self._param_panel.set_enabled(True)
        self._result_panel.show_results(
            msg.image_data, msg.seg, msg.lobe_results)

    def _on_analysis_error(self, error: str):
        self._status("Error — see dialog.", 0.0)
        self._param_panel.set_enabled(True)
        messagebox.showerror(
            "Analysis error",
            f"An error occurred during analysis:\n\n{error[:800]}",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _status(self, text: str, fraction: float):
        self._status_var.set(text)
        self._progress_var.set(fraction)
        self.update_idletasks()
