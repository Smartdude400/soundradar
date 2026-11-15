"""The real-time capture and analysis engine.

Threading, which is what makes this feel instant:

* PortAudio runs its own output-callback thread, which does nothing but copy
  bytes from a preallocated buffer. It is deliberately trivial -- anything
  expensive there causes dropouts.
* One reader thread does the blocking ``read`` (which self-paces at real time)
  and feeds the detector inline. Folding the analysis into the reader avoids a
  queue hop and the latency that comes with it.
* The Tk main thread never touches audio. It swaps in the newest published
  :class:`Frame` roughly every 16 ms and draws whatever that frame says, so a
  slow frame drops an old one instead of building a backlog.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pyaudiowpatch as pa

from .detect import Detector, Event, Gait
from .devices import Device, render_index_for
from .doa import Geometry, gcc_phat

BLOCK = 1024


@dataclass(frozen=True)
class Frame:
    """One instant of audio, already reduced to what a frame of UI needs."""
    t: float
    level_db: float
    background_db: float
    #: Spectrum with the noise floor removed, for the residual view.
    residual_db: np.ndarray
    #: Peak frequency of the current frame, in Hz.
    peak_hz: float
    #: Live detection strength and the bar it must beat.
    flux_db: float
    threshold_db: float
    angle: float | None
    elevation: float | None
    quality: float
    events: tuple[Event, ...] = ()
    gait: Gait = field(default_factory=Gait)


class OutputSink:
    """A permanently running output stream.

    Its real job is not playback but keeping the WASAPI render endpoint alive.
    A loopback capture stream blocks *forever* if nothing is rendering to its
    speaker, so silence is pumped continuously whenever loopback capture is in
    use -- on the speaker that tap actually mirrors, since the default output is
    often a different endpoint entirely.
    """

    def __init__(self, device_index: int, rate: int, channels: int = 2,
                 pyaudio: pa.PyAudio | None = None) -> None:
        self._lock = threading.Lock()
        self._buffer = np.zeros((max(rate // 10, BLOCK * 4), channels), dtype=np.float32)
        self._pos = self._buffer.shape[0]
        self.channels = channels
        self._stream = (pyaudio or pa.PyAudio()).open(
            format=pa.paFloat32, channels=channels, rate=rate, output=True,
            frames_per_buffer=BLOCK, output_device_index=device_index,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):  # noqa: ARG002
        """Called ~47 times a second from PortAudio's thread, so it stays
        allocation-free and never blocks on anything."""
        with self._lock:
            pos = self._pos
            if pos + frame_count > self._buffer.shape[0]:
                self._buffer[:] = 0.0
                pos = 0
            out = self._buffer[pos:pos + frame_count]
            self._pos = pos + frame_count
        return out.tobytes(), pa.paContinue

    def close(self) -> None:
        try:
            self._stream.stop_stream()
            self._stream.close()
        except Exception:
            pass


class Engine:
    """Captures one device and publishes analysed frames."""

    def __init__(self, device: Device, geometry: Geometry | None = None,
                 sensitivity: float = 0.35) -> None:
        self.device = device
        self.geometry = geometry
        self.sensitivity = sensitivity
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pyaudio: pa.PyAudio | None = None
        self._sink: OutputSink | None = None
        self._stream = None
        self._error: str | None = None
        self._detector: Detector | None = None
        self.history: list[tuple[float, Event, float | None, float]] = []

    @property
    def n_bins(self) -> int:
        """Spectrum length, so the display cannot index past the data."""
        return self._detector.n_bins if self._detector else 1025

    @property
    def error(self) -> str | None:
        return self._error

    def set_sensitivity(self, value: float) -> None:
        self.sensitivity = value

    def start(self) -> None:
        dev = self.device
        self._pyaudio = pa.PyAudio()
        if dev.loopback:
            out_index = render_index_for(dev, self._pyaudio)
            if out_index is not None:
                try:
                    self._sink = OutputSink(out_index, dev.rate, pyaudio=self._pyaudio)
                except Exception as exc:
                    self._error = f"output sink unavailable: {exc}"
        try:
            self._stream = self._pyaudio.open(
                format=pa.paFloat32, channels=dev.channels, rate=dev.rate,
                input=True, frames_per_buffer=BLOCK, input_device_index=dev.index,
            )
        except Exception as exc:
            self.stop()
            self._error = f"could not open {dev.name}: {exc}"
            return
        self._stream.start_stream()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        rate = self.device.rate
        detector = Detector(rate, sensitivity=self.sensitivity)
        self._detector = detector
        geometry = self.geometry
        pair = geometry.pairs[0] if geometry and geometry.pairs else None
        max_lag = geometry.max_lag(rate) if geometry else 0
        history = np.zeros((2, 2048), dtype=np.float32)
        started = time.perf_counter()
        angle = elevation = quality = None
        counter = 0

        while not self._stop.is_set():
            raw = self._stream.read(BLOCK)
            block = np.frombuffer(raw, dtype=np.float32).reshape(-1, self.device.channels)
            mono = block[:, 0]
            now = time.perf_counter() - started

            events = detector.push(mono)
            if events:
                # Bearing lags the sound, so every event published in this block
                # shares the direction measured for it.
                for event in events:
                    self.history.append((event.t, event, angle, quality or 0.0))
                del self.history[:-200]

            if pair is not None and self.device.channels > 1:
                history[:, 1:] = history[:, :-1]
                history[0, 0] = block[0, pair.a]
                history[1, 0] = block[0, pair.b]
                counter += 1
                if counter % 2 == 0 and history.shape[1] == 2048:
                    delay, conf = gcc_phat(history[0], history[1], max_lag)
                    if conf > 0.05:
                        az, el, q = geometry.solve(np.array([delay]),
                                                   np.array([conf]), rate)
                        if q > 0.05:
                            angle, elevation, quality = az, el, q

            with self._lock:
                self._frame = Frame(
                    t=now,
                    level_db=detector.level_db,
                    background_db=detector.background_db,
                    residual_db=detector.residual_db,
                    peak_hz=float(np.argmax(detector.residual_db) * rate
                                  / detector.n_fft),
                    flux_db=detector.flux_db,
                    threshold_db=detector.threshold(),
                    angle=angle, elevation=elevation, quality=quality or 0.0,
                    events=tuple(events),
                    gait=detector.gait,
                )

    def frame(self) -> Frame | None:
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        for closer in (self._stream, self._sink):
            if closer is not None:
                try:
                    closer.close()
                except Exception:
                    pass
        self._stream = self._sink = None
        if self._pyaudio is not None:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None
