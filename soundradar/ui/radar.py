"""The polar plot.

What can honestly be drawn depends entirely on the hardware:

* two non-parallel baselines  -> a true azimuth, plotted as a blip
* a single collinear baseline  -> a cone angle only, so the contact is drawn as
  a sector, because it genuinely could sit anywhere inside that cone
* one channel                 -> nothing at all, and the widget says so
"""
from __future__ import annotations

import math
import tkinter as tk

BG = "#0b0f14"
GRID = "#1d2b36"
RING = "#263a49"
TEXT = "#7d94a4"
HINT = "#3d5261"
HOT = "#3ee0a4"
WARM = "#ffc95c"

#: Contacts persist this long, fading as they age, like a real radar scope.
PERSIST = 4.0


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> str:
    t = max(0.0, min(1.0, t))
    return "#%02x%02x%02x" % tuple(int(x + (y - x) * t) for x, y in zip(a, b))


FADE_BG = (0x0b, 0x0f, 0x14)


class RadarCanvas(tk.Canvas):
    def __init__(self, master: tk.Misc, **kw) -> None:
        super().__init__(master, bg=BG, highlightthickness=0, **kw)
        self.bind("<Configure>", lambda _e: self.redraw())
        self._items: list[int] = []
        self._items.clear()
        self._contacts: list[tuple[float, str, float | None, float, float]] = []
        self._now = 0.0
        self._mode = "none"
        self._level = 0.0

    def clear(self) -> None:
        self._contacts.clear()

    def add(self, t: float, kind: str, angle: float | None, quality: float, db: float) -> None:
        self._contacts.append((t, kind, angle, quality, db))
        if len(self._contacts) > 200:
            del self._contacts[:-200]

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.redraw()

    def redraw(self, now: float | None = None, level: float = 0.0) -> None:
        if now is not None:
            self._now = now
        self._level = level
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20 or h < 20:
            return
        cx, cy = w / 2, h / 2
        radius = min(cx, cy) - 34
        if radius <= 10:
            return

        self._draw_grid(cx, cy, radius)
        if self._mode == "none":
            self.create_text(cx, cy, text="SINGLE CHANNEL\nno bearing available\n\nuse a stereo or multi-mic\ninput for direction",
                             fill=HINT, justify="center", font=("Segoe UI", 10))
            return
        self._draw_sweep(cx, cy, radius)
        self._draw_contacts(cx, cy, radius)

    def _draw_grid(self, cx: float, cy: float, r: float) -> None:
        for i in range(1, 4):
            self.create_oval(cx - r * i / 3, cy - r * i / 3, cx + r * i / 3, cy + r * i / 3,
                             outline=GRID)
        for deg in range(0, 360, 30):
            a = math.radians(deg - 90)
            self.create_line(cx, cy, cx + r * math.cos(a), cy + r * math.sin(a), fill=GRID)
        for label, deg in (("0", 0), ("90", 90), ("180", 180), ("270", 270)):
            a = math.radians(deg - 90)
            self.create_text(cx + (r + 15) * math.cos(a), cy + (r + 15) * math.sin(a),
                             text=label, fill=TEXT, font=("Segoe UI", 8))
        self.create_text(cx, cy - r - 28, text="front", fill=TEXT, font=("Segoe UI", 8))
        caption = {"cone": "cone angle - front/back ambiguous",
                   "image": "stereo image position"}.get(self._mode, "bearing")
        self.create_text(cx, cy + r + 26, text=caption, fill=HINT, font=("Segoe UI", 8))

    def _draw_sweep(self, cx: float, cy: float, r: float) -> None:
        # One revolution every four seconds, matched to the fade window so a
        # contact is brightest at the moment the sweep first paints it.
        a = math.radians((self._now * 90.0) % 360.0 - 90.0)
        self.create_line(cx, cy, cx + r * math.cos(a), cy + r * math.sin(a),
                         fill="#16323f", width=2)

    def _draw_contacts(self, cx: float, cy: float, r: float) -> None:
        for t, kind, angle, quality, db in self._contacts:
            age = self._now - t
            if age < 0 or age > PERSIST:
                continue
            fade = 1.0 - age / PERSIST
            if angle is None or self._mode == "none":
                continue
            strength = max(0.15, min(1.0, (db + 70.0) / 70.0))
            colour = _mix(FADE_BG, tuple(int(WARM[i:i + 2], 16) for i in (1, 3, 5)) if kind == "CLICK" else
                          tuple(int(HOT[i:i + 2], 16) for i in (1, 3, 5)), fade * strength)
            if self._mode == "cone":
                self._draw_cone(cx, cy, r, angle, fade, colour)
            else:
                a = math.radians(90.0 - angle - 90.0)
                dist = r * (0.45 + 0.5 * strength)
                x, y = cx + dist * math.cos(a), cy + dist * math.sin(a)
                size = 3 + 4 * strength
                self.create_oval(x - size, y - size, x + size, y + size,
                                 outline=colour, width=2)

    def _draw_cone(self, cx: float, cy: float, r: float, angle: float,
                   fade: float, colour: str) -> None:
        """A collinear array resolves a cone, so draw the cone -- both mirror
        images at once, because the hardware cannot tell them apart."""
        for sign in (1, -1):
            a = math.radians(90.0 - sign * angle - 90.0)
            dist = r * 0.6
            x, y = cx + dist * math.cos(a), cy + dist * math.sin(a)
            self.create_oval(x - 4, y - 4, x + 4, y + 4, outline=colour, width=2)
        half = math.radians(max(3.0, abs(angle) * 0.35))
        a = math.radians(-90.0)
        self.create_arc(cx - r * 0.75, cy - r * 0.75, cx + r * 0.75, cy + r * 0.75,
                        start=-90 - math.degrees(half), extent=2 * math.degrees(half),
                        style=tk.ARC, outline=colour, width=2)
