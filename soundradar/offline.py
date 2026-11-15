"""Offline analysis of an audio file.

Runs the same detector the live engine uses, frame by frame, so a recording can
be mined for exactly the events the radar would have shown in real time. Two
things differ on purpose:

* The analysis is not time-constrained, so the block size is larger and the
  whole file is processed as fast as it can be.
* Events come back in one batch, with the bearing attached afterwards, since
  there is no "now" to lag.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .detect import Detector, Event

BLOCK = 4096


@dataclass
class FileReport:
    path: str
    rate: int
    duration: float
    channels: int
    background_db: float
    events: list[Event]
    pan_events: int = 0
    truncated: bool = False

    def to_rows(self) -> list[dict]:
        return [
            {
                "time_s": round(e.t, 3),
                "type": e.kind,
                "freq_hz": round(e.hz, 1),
                "level_dbfs": round(e.db, 1),
                "snr_db": round(e.snr_db, 1),
                "duration_ms": round(e.duration * 1000, 1),
                "position_deg": "" if e.axis_deg is None else round(e.axis_deg, 1),
            }
            for e in self.events
        ]

    def write_csv(self, target: Path) -> None:
        rows = self.to_rows()
        fields = ["time_s", "type", "freq_hz", "level_dbfs", "snr_db",
                  "duration_ms", "position_deg"]
        with target.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def write_json(self, target: Path) -> None:
        payload = asdict(self)
        payload["events"] = [asdict(e) | {"colour": e.colour} for e in self.events]
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load(path: str | Path) -> tuple[np.ndarray, int]:
    """Read any format SciPy handles (WAV, FLAC, OGG) as float samples."""
    rate, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    if data.ndim == 1:
        data = data[:, None]
    return np.ascontiguousarray(data), int(rate)


def analyse_file(path: str | Path, limit_s: float | None = None,
                 sensitivity: float = 0.35) -> FileReport:
    """Collect every event a recording contains."""
    data, rate = load(path)
    channels = data.shape[1]
    if limit_s:
        data = data[: int(limit_s * rate)]

    detector = Detector(rate, sensitivity=sensitivity)
    mono_all = data[:, 0]
    events: list[Event] = []
    for start in range(0, len(data) - BLOCK + 1, BLOCK):
        events.extend(detector.push(mono_all[start:start + BLOCK]))
    if len(data) % BLOCK:
        events.extend(detector.push(mono_all[-BLOCK:]))

    positioned = 0
    if channels > 1:
        left = np.sqrt(np.mean(np.square(data[:, 0].astype(np.float64))))
        right = np.sqrt(np.mean(np.square(data[:, 1].astype(np.float64))))
        total = left + right
        if total > 1e-9:
            # A whole-file pan is a property of the *mix*, not a physical
            # direction, which is why the UI labels it an image position.
            pan = float((left - right) / total)
            events = [replace(e, axis_deg=pan * 90.0,
                                         confidence=abs(pan)) for e in events]
            positioned = len(events)

    return FileReport(
        path=str(path), rate=rate, duration=len(data) / rate, channels=channels,
        background_db=detector.background_db, events=events,
        pan_events=positioned,
    )


