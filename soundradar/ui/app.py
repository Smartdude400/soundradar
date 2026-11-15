"""The application window.

One Tk main thread drives a ~60 Hz redraw off a timer. It never reads audio and
never blocks: each tick takes whatever the newest analysed frame happens to be
and draws it, so a slow frame drops old ones instead of queueing latency.

The layout answers the three questions that matter while playing, in order:
what just happened (ribbon), what it was (log), and where from (locator).
"""
from __future__ import annotations

import ctypes
import sys
import time
import tkinter as tk

if sys.platform == "win32":
    # Must happen before Tk initialises, or Windows scales the window after the
    # fact and everything is drawn at the wrong size on a high-DPI display.
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:  # pragma: no cover - older Windows
        pass
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


from .. import offline
from ..detect import Event
from ..devices import Device, inputs
from ..doa import Geometry, linear_array, square_array, tetra_array
from ..engine import Engine
from .radar import RadarCanvas
from .ribbon import Ribbon
from .scope import Waterfall

BG = "#0b0f14"
PANEL = "#121a21"
FG = "#c7d6e0"
DIM = "#7d94a4"
WARM = "#ffc95c"
ACCENT = "#3ee0a4"

ARRAYS = {
    "1 channel": None,
    "2-mic linear": (2, "linear"),
    "3-mic linear": (3, "linear"),
    "4-mic linear": (4, "linear"),
    "4-mic square": (4, "square"),
    "4-mic tetra (3D)": (4, "tetra"),
}

EVENT_COLUMNS = ("time", "type", "freq", "level", "above bg", "dur", "bearing")


def _hz(value: float) -> str:
    return f"{value / 1000:.1f}k" if value >= 1000 else f"{value:.0f}"


class Table(tk.Frame):
    """An events table that is both a widget and a row API.

    Call sites need to fill rows *and* place the widget with whichever geometry
    manager their own layout uses -- Tk refuses to mix pack and grid inside one
    parent -- so the frame and the tree's row operations are combined here.
    """

    WIDTHS = {"time": 70, "type": 98, "freq": 70, "level": 68, "above bg": 82,
              "dur": 72, "bearing": 78}

    def __init__(self, master: tk.Misc, columns: tuple[str, ...]) -> None:
        super().__init__(master, bg=BG)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        style = ttk.Style()
        style.configure("Dark.Treeview", background=PANEL, fieldbackground=PANEL,
                        foreground=FG, rowheight=20, borderwidth=0)
        style.configure("Dark.Treeview.Heading", background="#1b2731",
                        foreground=DIM, relief="flat")
        style.map("Dark.Treeview", background=[("selected", "#1f3b46")])
        self.tree = ttk.Treeview(self, columns=columns, show="headings",
                                 style="Dark.Treeview")
        for name in columns:
            self.tree.heading(name, text=name)
            # stretch=False keeps the declared widths: Treeview otherwise
            # expands every column to fill the pane and pushes the last ones
            # off the right edge.
            self.tree.column(name, width=self.WIDTHS.get(name, 80), anchor="w",
                             stretch=False)
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

    def insert(self, *args, **kw):
        return self.tree.insert(*args, **kw)

    def delete(self, *args, **kw):
        return self.tree.delete(*args, **kw)

    def get_children(self, *args, **kw):
        return self.tree.get_children(*args, **kw)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SoundRadar")
        self.geometry("1280x800")
        self.minsize(1000, 640)
        self.configure(bg=BG)

        self.engine: Engine | None = None
        self._last_frame_t = -1.0
        self._fps_times: list[float] = []
        self._sensitivity = 0.35
        self._residual = False
        self._report: offline.FileReport | None = None
        self.after(16, self._tick)

        self._build_header()
        self._build_tabs()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------------------------------------------------------- header
    def _build_header(self) -> None:
        bar = tk.Frame(self, bg=PANEL, padx=10, pady=8)
        bar.pack(fill="x")

        tk.Label(bar, text="SOURCE", bg=PANEL, fg=DIM, font=("Segoe UI", 8)).pack(side="left")
        try:
            self._devices = inputs()
        except Exception as exc:
            self._devices = []
            messagebox.showerror("SoundRadar",
                                 f"Could not enumerate audio devices:\n{exc}")
        # `Combobox.insert` is a silent no-op on Tk 9, so the whole list goes in
        # through `values` at construction time.
        labels = [f"{d.name}  ·  {d.kind}  ·  {d.rate} Hz" for d in self._devices]
        self.source = ttk.Combobox(bar, state="readonly", width=44, values=labels)
        if labels:
            self.source.current(0)
        self.source.pack(side="left", padx=(6, 14))
        self.source.bind("<<ComboboxSelected>>", lambda _e: self._reset())

        tk.Label(bar, text="SENSITIVITY", bg=PANEL, fg=DIM,
                 font=("Segoe UI", 8)).pack(side="left")
        # The readout is built first: ttk.Scale fires its command as soon as it
        # is given an initial value, and that callback writes to this label.
        self.sens_value = tk.Label(bar, text="", bg=PANEL, fg=FG, width=10,
                                   anchor="w", font=("Segoe UI", 9))
        self.sens_value.pack(side="left")
        self.sens = ttk.Scale(bar, from_=0, to=100, length=170,
                              command=self._on_sensitivity)
        self.sens.pack(side="left", padx=(6, 4))
        self.sens.set(self._sensitivity * 100)

        tk.Label(bar, text="ARRAY", bg=PANEL, fg=DIM, font=("Segoe UI", 8)).pack(
            side="left", padx=(14, 0))
        self.array = ttk.Combobox(bar, state="readonly", width=16, values=list(ARRAYS))
        self.array.current(0)
        self.array.pack(side="left", padx=(6, 10))
        self.array.bind("<<ComboboxSelected>>", lambda _e: self._reset())
        tk.Label(bar, text="cm", bg=PANEL, fg=DIM, font=("Segoe UI", 8)).pack(side="left")
        self.spacing = ttk.Entry(bar, width=5)
        self.spacing.insert(0, "20")
        self.spacing.pack(side="left", padx=(4, 10))
        self.spacing.bind("<Return>", lambda _e: self._reset())

        self.toggle = tk.Button(bar, text="Start", command=self._toggle, bg=ACCENT,
                                fg="#06210f", activebackground="#2fc78d",
                                relief="flat", padx=20, font=("Segoe UI", 9, "bold"))
        self.toggle.pack(side="right")

        self.status = tk.Label(self, text="idle", bg=BG, fg=DIM, anchor="w",
                               font=("Consolas", 9))
        self.status.pack(fill="x", padx=10, pady=(4, 0))

    def _on_sensitivity(self, raw: str) -> None:
        self._sensitivity = float(raw) / 100.0
        self.sens_value.configure(text=_SENS_WORDS(self._sensitivity))
        if self.engine:
            self.engine.set_sensitivity(self._sensitivity)

    # ------------------------------------------------------------------ tabs
    def _build_tabs(self) -> None:
        book = ttk.Notebook(self)
        book.pack(fill="both", expand=True, padx=10, pady=8)
        self._book = book
        self._build_detect_tab(book)
        self._build_file_tab(book)

    def _build_detect_tab(self, book: ttk.Notebook) -> None:
        tab = ttk.Frame(book, padding=8)
        book.add(tab, text="  Detect  ")

        # pack with equal `expand` gives a dependable 50/50 split; grid weights
        # inside a notebook child left the slack undistributed.
        left = tk.Frame(tab, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        tk.Label(left, text="BEARING", bg=BG, fg=DIM, font=("Segoe UI", 8)).pack(anchor="w")
        self.radar = RadarCanvas(left, width=360, height=360)
        self.radar.pack(fill="both", expand=True)
        self.gait = tk.Label(left, text="", bg=BG, fg=ACCENT, anchor="w",
                             font=("Segoe UI", 10, "bold"))
        self.gait.pack(fill="x", pady=(6, 0))

        right = tk.Frame(tab, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        head = tk.Frame(right, bg=BG)
        head.pack(fill="x")
        tk.Label(head, text="SPECTRUM", bg=BG, fg=DIM,
                 font=("Segoe UI", 8)).pack(side="left")
        self.view_btn = tk.Button(head, text="show residual", command=self._toggle_view,
                                  bg=PANEL, fg=FG, activebackground="#1b2731",
                                  relief="flat", padx=10, font=("Segoe UI", 8))
        self.view_btn.pack(side="right")
        self.waterfall = Waterfall(right, height=200)
        self.waterfall.pack(fill="x")

        self.level = tk.Canvas(right, bg=PANEL, height=24, highlightthickness=0)
        self.level.pack(fill="x", pady=(6, 4))
        self._bg_rect = self.level.create_rectangle(0, 0, 0, 20, fill="#2b3a46", width=0)
        self._lvl_rect = self.level.create_rectangle(0, 0, 0, 20, fill=ACCENT, width=0)
        self.level.create_text(6, 12, text="BACKGROUND", fill=DIM, anchor="w",
                               font=("Segoe UI", 7))
        self.level_text = self.level.create_text(0, 12, text="", fill=FG, anchor="e",
                                                 font=("Consolas", 9))

        self.ribbon = Ribbon(right, height=64)
        self.ribbon.pack(fill="x", pady=(4, 6))

        self.events = self._table(right, EVENT_COLUMNS)
        self.events.pack(fill="both", expand=True)

    def _build_file_tab(self, book: ttk.Notebook) -> None:
        tab = ttk.Frame(book, padding=12)
        book.add(tab, text="  File  ")

        bar = tk.Frame(tab, bg=BG)
        bar.pack(fill="x")
        tk.Button(bar, text="Open audio file…", command=self._open_file, bg=PANEL, fg=FG,
                  activebackground="#1b2731", relief="flat", padx=16).pack(side="left")
        tk.Button(bar, text="Export CSV", command=lambda: self._export(".csv"),
                  bg=PANEL, fg=FG, activebackground="#1b2731", relief="flat",
                  padx=16).pack(side="left", padx=8)
        tk.Button(bar, text="Export JSON", command=lambda: self._export(".json"),
                  bg=PANEL, fg=FG, activebackground="#1b2731", relief="flat",
                  padx=16).pack(side="left")

        self.file_info = tk.Label(tab, text="No file analysed.", bg=BG, fg=FG,
                                  anchor="w", justify="left", font=("Segoe UI", 10))
        self.file_info.pack(fill="x", pady=(16, 10))
        self.file_table = self._table(tab, EVENT_COLUMNS)
        self.file_table.pack(fill="both", expand=True)

    def _table(self, master: tk.Misc, columns: tuple[str, ...]) -> Table:
        return Table(master, columns)

    # ------------------------------------------------------------- lifecycle
    def _current_device(self) -> Device | None:
        index = self.source.current()
        return self._devices[index] if 0 <= index < len(self._devices) else None

    def _current_geometry(self, device: Device) -> Geometry | None:
        spec = ARRAYS.get(self.array.get())
        if not spec or device.channels < spec[0]:
            return None
        try:
            spacing = max(0.01, float(self.spacing.get())) / 100.0
        except ValueError:
            spacing = 0.20
        if spec[1] == "linear":
            return linear_array(spec[0], spacing)
        if spec[1] == "square":
            return square_array(spacing)
        return tetra_array(spacing, spacing)

    def _reset(self) -> None:
        if self.engine is not None:
            self._stop_engine()
            self.toggle.configure(text="Start")
        self.ribbon.clear()
        for tree in (self.events,):
            tree.delete(*tree.get_children())
        self.radar.clear()
        self.radar.redraw(0.0)

    def _toggle(self) -> None:
        if self.engine is not None:
            self._stop_engine()
            self.toggle.configure(text="Start")
            return
        device = self._current_device()
        if device is None:
            messagebox.showerror("SoundRadar", "No input device selected.")
            return
        geometry = self._current_geometry(device)
        self.engine = Engine(device, geometry, sensitivity=self._sensitivity)
        self.engine.start()
        if self.engine.error:
            messagebox.showerror("SoundRadar", self.engine.error)
            self._stop_engine()
            return
        self.ribbon.clear()
        self.events.delete(*self.events.get_children())
        self.waterfall.configure_source(device.rate, self.engine.n_bins)
        if device.loopback:
            self.radar.set_mode("image" if device.channels > 1 else "none")
        elif geometry is None:
            self.radar.set_mode("none")
        elif geometry.is_linear:
            self.radar.set_mode("cone")
        else:
            self.radar.set_mode("bearing")
        self.toggle.configure(text="Stop")

    def _stop_engine(self) -> None:
        if self.engine is not None:
            self.engine.stop()
            self.engine = None

    def _on_close(self) -> None:
        self._stop_engine()
        self.destroy()

    def _toggle_view(self) -> None:
        self._residual = not self._residual
        self.waterfall.set_residual(self._residual)
        self.view_btn.configure(text="show level" if self._residual
                                else "show residual")

    # ------------------------------------------------------------------ loop
    def _tick(self) -> None:
        """Redraw timer.

        The reschedule happens in a ``finally`` so one bad frame cannot silently
        kill the display for the rest of the session, and any failure is shown
        in the status bar rather than swallowed.
        """
        try:
            if not self.winfo_exists():
                return
            engine = self.engine
            if engine is not None:
                frame = engine.frame()
                if frame is not None and frame.t != self._last_frame_t:
                    self._last_frame_t = frame.t
                    self._draw(frame)
        except Exception as exc:
            self.status.configure(text=f"render error: {exc}")
        finally:
            if self.winfo_exists():
                self.after(16, self._tick)

    def _draw(self, frame) -> None:
        self.waterfall.feed(frame.residual_db if self._residual
                            else self._raw_spectrum(frame))
        self.radar.redraw(frame.t, frame.level_db)
        self.ribbon.draw(frame.t)

        self._draw_level(frame)
        for event in frame.events:
            self._add_event(event, frame.angle, frame.quality)

        now = time.perf_counter()
        self._fps_times.append(now)
        while self._fps_times and now - self._fps_times[0] > 1.0:
            self._fps_times.pop(0)
        gait = frame.gait
        self.gait.configure(
            text=(f"{gait.steps_per_second:.1f} steps/sec  ·  {gait.label}"
                  if gait.label else ""), fg=ACCENT)
        if self.engine:
            device = self.engine.device
            self.status.configure(
                text=(f"{len(self._fps_times):3d} fps  ·  {device.rate} Hz  ·  "
                      f"background {frame.background_db:6.1f} dBFS  ·  "
                      f"detection {frame.flux_db:5.1f} / {frame.threshold_db:5.1f} dB  ·  "
                      f"{self.array.get()}"))
        else:
            self.status.configure(text="idle")

    def _raw_spectrum(self, frame):
        """Reconstruct a dBFS spectrum for the raw view from the residual.

        The detector only keeps the floor-relative copy, so the background is
        added back from the level it reports.
        """
        return frame.residual_db + frame.background_db

    def _draw_level(self, frame) -> None:
        width = max(1, self.level.winfo_width() - 12)
        span = width - 92
        lo, hi = -96.0, 0.0
        x0 = 92 + (frame.background_db - lo) / (hi - lo) * span
        x1 = 92 + (frame.level_db - lo) / (hi - lo) * span
        self.level.coords(self._bg_rect, x0, 2, x1, 22)
        self.level.coords(self._lvl_rect, x1 - 1.5, 2, x1 + 1.5, 22)
        self.level.itemconfig(self._lvl_rect, fill=ACCENT)
        self.level.coords(self.level_text, width - 4, 12)
        self.level.itemconfig(
            self.level_text,
            text=f"bg {frame.background_db:6.1f}   now {frame.level_db:6.1f} dBFS")

    def _add_event(self, event: Event, angle, quality) -> None:
        located = replace(event, axis_deg=angle, confidence=quality) if angle is not None \
            else event
        self.ribbon.add(located.t, located.kind, located.db)
        if self.radar._mode != "none":
            self.radar.add(located.t, located.kind, located.axis_deg,
                           located.confidence, located.db)
        self.events.insert("", 0, values=self._row(located))
        children = self.events.get_children()
        if len(children) > 200:
            self.events.delete(*children[200:])

    @staticmethod
    def _row(event: Event) -> tuple:
        bearing = "—" if event.axis_deg is None else f"{event.axis_deg:+.0f}°"
        return (f"{event.t:7.2f}s", event.kind, f"{_hz(event.hz)} Hz",
                f"{event.db:6.1f}", f"{event.snr_db:5.1f}",
                f"{event.duration * 1000:5.0f}ms", bearing)

    # ------------------------------------------------------------------ file
    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an audio file",
            filetypes=[("Audio", "*.wav *.flac *.ogg"), ("All files", "*.*")])
        if not path:
            return
        self.status.configure(text="analysing…")
        self.update_idletasks()
        try:
            report = offline.analyse_file(path, sensitivity=self._sensitivity)
        except Exception as exc:
            messagebox.showerror("SoundRadar", f"Could not read that file:\n{exc}")
            self.status.configure(text="idle")
            return
        self._report = report
        summary = (f"{Path(path).name}   {report.duration:.1f} s   {report.rate} Hz   "
                   f"{report.channels} ch   background {report.background_db:.1f} dBFS   "
                   f"{len(report.events)} events")
        if report.channels > 1:
            summary += "\nposition is the stereo image, a property of the mix -- not a direction"
        self.file_info.configure(text=summary, fg=FG)
        self.file_table.delete(*self.file_table.get_children())
        for event in report.events:
            self.file_table.insert("", "end", values=self._row(event))
        self.status.configure(text=f"analysed {len(report.events)} events")

    def _export(self, suffix: str) -> None:
        if self._report is None:
            messagebox.showinfo("SoundRadar", "Analyse a file first.")
            return
        target = Path(self._report.path).with_suffix(suffix)
        if suffix == ".csv":
            self._report.write_csv(target)
        else:
            self._report.write_json(target)
        self.status.configure(text=f"wrote {target}")


def _SENS_WORDS(value: float) -> str:
    if value < 0.15:
        return "maximum"
    if value < 0.4:
        return "sensitive"
    if value < 0.65:
        return "balanced"
    if value < 0.85:
        return "strict"
    return "gunshots only"

def main() -> None:
    """Open the window and run until it is closed."""
    App().mainloop()
