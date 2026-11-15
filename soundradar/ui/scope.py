"""Spectrogram with a residual view.

Raw audio hides exactly what this tool is for: a footstep 13 dB under a loud
bed is invisible on a dBFS spectrogram but obvious once the noise floor is
subtracted. The two views are the same data, and the toggle exists so the user
can see for themselves that nothing was invented -- only the background
removed.
"""
from __future__ import annotations

import math
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

BG = "#0b0f14"
TEXT = "#7d94a4"
GRID = "#16222c"
F_MIN, F_MAX = 40.0, 16000.0
#: dBFS window for the raw view.
RAW_FLOOR, RAW_CEIL = -96.0, -10.0
#: dB above the noise floor for the residual view. Faint events live at the top
#: of this range, so it is deliberately tight.
RES_FLOOR, RES_CEIL = 3.0, 34.0


def _ramp() -> np.ndarray:
    """Dark navy through teal to amber: one table lookup per pixel."""
    stops_x = np.array([0.00, 0.25, 0.50, 0.72, 0.88, 1.00])
    stops_rgb = np.array([
        [11, 15, 20], [18, 58, 78], [26, 140, 130],
        [110, 200, 110], [240, 200, 80], [255, 245, 220],
    ], dtype=np.float64)
    out = np.empty((256, 3), dtype=np.uint8)
    grid = np.linspace(0.0, 1.0, 256)
    for channel in range(3):
        out[:, channel] = np.interp(grid, stops_x, stops_rgb[:, channel])
    return out


LUT = _ramp()


class Waterfall(tk.Frame):
    def __init__(self, master: tk.Misc, height: int = 190, **kw) -> None:
        # The height has to go on the *frame*: pack_propagate(False) stops the
        # frame shrinking to its children, so a height set only on the inner
        # canvas leaves the whole widget collapsed to nothing.
        super().__init__(master, bg=BG, height=height, **kw)
        self.pack_propagate(False)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._rows = 0
        self._cols = 0
        self._hist: np.ndarray | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._index: np.ndarray | None = None
        self._rate = 48000
        self._n_bins = 1025
        self.residual = False
        self.canvas.bind("<Configure>", lambda _e: self._resize())

    def configure_source(self, rate: int, n_bins: int) -> None:
        """Tell the widget the shape of the spectra it will be fed."""
        self._rate = rate
        self._n_bins = n_bins
        self._build_index()
        self._resize()

    def set_residual(self, on: bool) -> None:
        self.residual = on

    def _build_index(self) -> None:
        """Which FFT bins collapse into which display row, on a log axis."""
        if self._rows < 2:
            return
        top = min(F_MAX, self._rate / 2.0)
        edges = np.geomspace(F_MIN, top, self._rows + 1)
        freqs = np.linspace(0.0, self._rate / 2.0, self._n_bins)
        starts = np.searchsorted(freqs, edges[:-1])
        stops = np.minimum(np.maximum(starts + 1, np.searchsorted(freqs, edges[1:])),
                           self._n_bins)
        self._index = np.stack([starts, stops], axis=1)

    def _draw_ticks(self) -> None:
        self.canvas.delete("ticks")
        h = self.canvas.winfo_height()
        top = min(F_MAX, self._rate / 2.0)
        for hz in (100, 1000, 10000):
            if hz > top:
                continue
            frac = math.log(hz / F_MIN) / math.log(top / F_MIN)
            y = h - frac * h
            self.canvas.create_line(0, y, self.canvas.winfo_width(), y, fill=GRID,
                                    tags="ticks")
            label = f"{hz // 1000}k" if hz >= 1000 else str(hz)
            self.canvas.create_text(4, y - 7, text=label, fill=TEXT, anchor="w",
                                    font=("Segoe UI", 7), tags="ticks")

    def _resize(self) -> None:
        w = max(64, self.canvas.winfo_width())
        h = max(32, self.canvas.winfo_height())
        if w == self._cols and h == self._rows:
            return
        self._cols, self._rows = w, h
        self._build_index()
        self._hist = np.zeros((self._rows, w, 3), dtype=np.uint8)
        self._draw_ticks()

    def feed(self, spectrum: np.ndarray) -> None:
        # Re-check every frame: a Configure event can land before the canvas
        # reaches its final size, which would leave the image stale.
        self._resize()
        if self._index is None or self._hist is None:
            return
        # A row covering several bins shows their strongest, which is what the
        # eye expects from a spectrogram at low zoom.
        peak = np.empty(self._rows, dtype=np.float32)
        for i, (a, b) in enumerate(self._index):
            peak[i] = spectrum[a:b].max()
        if self.residual:
            norm = (peak - RES_FLOOR) / (RES_CEIL - RES_FLOOR)
        else:
            norm = (peak - RAW_FLOOR) / (RAW_CEIL - RAW_FLOOR)
        norm = np.clip(norm, 0.0, 1.0)
        self._hist[:, :-1] = self._hist[:, 1:]
        self._hist[:, -1] = LUT[(norm * 255).astype(np.uint8)]
        self._blit()

    def _blit(self) -> None:
        if self._hist is None:
            return
        self._photo = ImageTk.PhotoImage(Image.fromarray(self._hist, "RGB"))
        self.canvas.delete("img")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw", tags="img")
