"""Event ribbon: a scrolling timeline of what was heard, and when.

The log table answers "what was that". The ribbon answers "when, and how
often", which is the question that actually matters in play -- a run of
footsteps reads as a rhythm at a glance, and a burst of gunfire as a solid bar.
"""
from __future__ import annotations

import tkinter as tk

BG = "#0b0f14"
GRID = "#182530"
TEXT = "#7d94a4"
FAINT = "#2b3a46"

#: Seconds of history shown across the full width.
WINDOW_S = 20.0


class Ribbon(tk.Canvas):
    def __init__(self, master: tk.Misc, height: int = 54, **kw) -> None:
        super().__init__(master, bg=BG, height=height, highlightthickness=0, **kw)
        self._events: list[tuple[float, str, float]] = []
        self._now = 0.0
        self._rows: list[str] = []
        self._row_index: dict[str, int] = {}
        self.bind("<Configure>", lambda _e: self.draw())

    def clear(self) -> None:
        self._events.clear()
        self._rows.clear()
        self._row_index.clear()

    def add(self, t: float, kind: str, level_db: float) -> None:
        self._events.append((t, kind, level_db))
        if len(self._events) > 400:
            del self._events[:-400]
        if kind not in self._row_index:
            self._row_index[kind] = len(self._rows)
            self._rows.append(kind)

    def draw(self, now: float | None = None) -> None:
        if now is not None:
            self._now = now
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 40 or not self._rows:
            if w >= 40:
                self.create_text(8, h / 2, anchor="w", fill=FAINT,
                                 text="events appear here", font=("Segoe UI", 9))
            return

        left = 74
        span = max(1, w - left - 6)
        row_h = min(11.0, (h - 4) / max(1, len(self._rows)))

        for i, kind in enumerate(self._rows):
            y = 2 + row_h * (i + 0.5)
            self.create_text(6, y, anchor="w", text=kind, fill=TEXT,
                             font=("Segoe UI", 7))
            self.create_line(left, y, w - 6, y, fill=GRID)

        # A tick every second keeps the eye oriented without clutter.
        first = int(self._now - WINDOW_S)
        for sec in range(first, int(self._now) + 1):
            x = left + (sec - (self._now - WINDOW_S)) / WINDOW_S * span
            if left <= x <= w - 6:
                self.create_line(x, 2, x, h - 2, fill=FAINT)
                if sec % 5 == 0:
                    self.create_text(x, h - 6, anchor="n", text=f"-{self._now - sec:.0f}s",
                                     fill=FAINT, font=("Segoe UI", 7))

        for t, kind, level_db in self._events:
            age = self._now - t
            if age < 0 or age > WINDOW_S:
                continue
            row = self._row_index.get(kind)
            if row is None:
                continue
            x = left + (1.0 - age / WINDOW_S) * span
            y = 2 + row_h * (row + 0.5)
            height = 2.0 + 6.0 * max(0.0, min(1.0, (level_db + 90.0) / 90.0))
            fade = 1.0 - age / WINDOW_S
            colour = _shade(kind, fade)
            self.create_line(x, y - height / 2, x, y + height / 2,
                             fill=colour, width=2)

    @property
    def window_s(self) -> float:
        return WINDOW_S


_BASE = {
    "GUNSHOT": (255, 77, 77),
    "EXPLOSION": (255, 154, 60),
    "FOOTSTEP": (77, 210, 255),
    "CLICK": (201, 160, 255),
    "VOICE": (143, 227, 136),
}


def _shade(kind: str, fade: float) -> str:
    rgb = _BASE.get(kind, (125, 148, 164))
    t = max(0.15, min(1.0, fade))
    mixed = tuple(int(c * t) for c in rgb)
    return "#%02x%02x%02x" % mixed
