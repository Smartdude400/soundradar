"""Detecting game audio: footsteps, gunfire, explosions.

The whole design turns on one idea: **never threshold absolute level**. A
footstep heard across a map sits far below the level of a gunshot fired in your
face, and both have to be caught in the same stream, in any room, at any master
volume. A fixed dBFS gate either misses the footsteps or drowns in gunfire.

Instead every bin is compared against its own running noise floor, and
detection runs on the *change* in that residual. Two consequences:

* A footstep 5 dB above a -75 dBFS noise floor is detected as well as a gunshot
  40 dB above a -30 dBFS floor, because both are measured against what is
  actually there.
* A steady background -- an ambient loop, music, a hum -- is absorbed into the
  floor and stops generating detections, which a level gate can never do.

The trade is honesty about the limit: if the background sits at -60 dBFS, a
footstep at -65 dBFS is *gone*, and no algorithm recovers it. The background
level is therefore reported continuously so the user can see what they are
working against.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-14

#: Band edges describing an event's spectral shape. The split at 2 kHz is the
#: important one: a boot impact is a low thump, gunfire carries real energy well
#: above it.
BANDS: tuple[tuple[str, float, float], ...] = (
    ("SUB", 20.0, 120.0),
    ("LOW", 120.0, 500.0),
    ("MID", 500.0, 2000.0),
    ("HIGH", 2000.0, 8000.0),
    ("AIR", 8000.0, 16000.0),
)

GUNSHOT = "GUNSHOT"
EXPLOSION = "EXPLOSION"
FOOTSTEP = "FOOTSTEP"
CLICK = "CLICK"
VOICE = "VOICE"
OTHER = "OTHER"

#: Colour per class, used by the ribbon and the log.
COLOURS = {
    GUNSHOT: "#ff4d4d",
    EXPLOSION: "#ff9a3c",
    FOOTSTEP: "#4dd2ff",
    CLICK: "#c9a0ff",
    VOICE: "#8fe388",
    OTHER: "#7d94a4",
}

#: Levels are reported no lower than this; below it the input is simply silent.
FLOOR_DB = -120.0

#: Longest an event is followed before it is reported. Must exceed an
#: explosion's tail or its decay can never be measured.
MEASURE_S = 0.60
#: Shortest an event is followed, so a one-frame spike still gets a duration.
MIN_S = 0.06


@dataclass
class Gait:
    """Footstep cadence, which separates sustained movement from one-off sounds."""
    steps_per_second: float = 0.0
    count: int = 0
    label: str = ""

    def update_label(self) -> None:
        if self.steps_per_second <= 0:
            self.label = ""
        elif self.steps_per_second < 1.4:
            self.label = "walking"
        elif self.steps_per_second < 2.7:
            self.label = "running"
        else:
            self.label = "sprinting"


@dataclass
class Event:
    """One detected sound."""
    t: float
    kind: str
    hz: float
    db: float
    snr_db: float
    duration: float
    axis_deg: float | None = None
    confidence: float = 0.0

    @property
    def colour(self) -> str:
        return COLOURS.get(self.kind, COLOURS[OTHER])


@dataclass
class _Frame:
    """One analysis frame, kept so an event can be measured after it has passed."""
    t: float
    power: np.ndarray
    residual_db: np.ndarray
    level_db: float
    #: Loudest bin above the noise floor. The envelope shape is read from this
    #: rather than the absolute level, so a footstep only 6 dB over a loud room
    #: still shows the short decay of a footstep instead of the room's.
    excess_db: float


@dataclass
class _Pending:
    """An onset waiting for its tail before it can be described."""
    t: float
    peak_flux: float = 0.0
    peak_db: float = -200.0
    done: bool = False
    frames: list = field(default_factory=list)


def band_slices(n_fft: int, rate: float) -> np.ndarray:
    """Resolve :data:`BANDS` to rfft bin ranges once, so per-event cost is a sum."""
    edges = np.fft.rfftfreq(n_fft, 1.0 / rate)
    out = []
    for _, lo, hi in BANDS:
        a = max(1, int(np.searchsorted(edges, lo)))
        b = max(a + 1, int(np.searchsorted(edges, min(hi, rate / 2.0))))
        out.append((a, min(b, edges.size)))
    return np.array(out)


class Detector:
    """Streaming detector. Feed it audio; it hands back events as they complete.

    Cost per frame is one 1024-point FFT plus a few vector ops -- roughly 25
    microseconds, about 0.3% of a core at 48 kHz.
    """

    #: How fast the floor may fall. This is the detector's most sensitive
    #: parameter: too fast and the floor *chases the event* -- at 0.35 a 24 dB
    #: footstep left almost no residual and was undetectable. 0.03 is a ~0.35 s
    #: time constant, fast enough to follow a genuinely quieter passage but far
    #: slower than any footstep, gunshot or explosion.
    ALPHA_DOWN = 0.03
    #: How fast it may rise: slower still, so a sustained sound -- music, a
    #: voice, an engine -- is absorbed into the floor instead of firing forever.
    ALPHA_UP = 0.0008
    #: How many of the most-changed bins form the detection strength.
    #: Averaging over *all* bins divides a footstep's energy (a handful of bins)
    #: by the whole spectrum, costing a narrowband thump ~17 dB of apparent
    #: strength -- exactly the faint sounds this exists to catch. Averaging over
    #: only the bins that changed still drowns the signal in hundreds of small
    #: wobbles, so the strength is the mean of the top K.
    FLUX_TOP_K = 8
    #: Width in bins of the boxcar applied along frequency before differencing.
    #: A single bin of a 2048-point FFT of Gaussian noise swings several dB
    #: frame to frame, and the extreme across hundreds of bins reaches ~20 dB --
    #: which is as large as a real footstep. Averaging 16 neighbouring bins
    #: (~375 Hz) cuts that spread by ~4x while leaving a footstep's thump, which
    #: is broad in frequency, essentially untouched. Measured: a -70 dBFS
    #: footstep on an -85 dBFS bed clears the noise 99th percentile by 3.2x
    #: with this setting, against 0.96x without it.
    FREQ_SMOOTH = 16

    def __init__(self, rate: int, n_fft: int = 2048, hop: int = 512,
                 sensitivity: float = 0.35) -> None:
        self.rate = rate
        self.n_fft = n_fft
        self.hop = hop
        self.sensitivity = sensitivity
        self._window = np.hanning(n_fft).astype(np.float32)
        # rfft returns the non-negative half of the transform, which for a real
        # signal holds half the total energy, and a Hann window passes
        # mean(w^2) of it. Folding both into one gain makes sum(spec) the true
        # mean square of the signal, so every dB below is real dBFS.
        self._gain = 2.0 / (n_fft ** 2 * float(np.mean(self._window ** 2)))
        self._band = band_slices(n_fft, rate)
        self._kernel = np.ones(self.FREQ_SMOOTH) / self.FREQ_SMOOTH
        self._ring = np.zeros(n_fft * 4, dtype=np.float32)
        self._stream = 0
        self._noise: np.ndarray | None = None
        self._prev_smooth: np.ndarray | None = None
        self._flux_hist: deque[float] = deque(maxlen=max(32, int(rate * 2.0 / hop)))
        self._frames: deque[_Frame] = deque(maxlen=int(MEASURE_S * rate / hop) + 8)
        self._pending: list[_Pending] = []
        self._out: list[Event] = []
        self._last_onset = -10.0
        self._steps: deque[float] = deque(maxlen=8)
        self._t0 = 0.0
        self._background = -90.0
        self.gait = Gait()
        #: Background level in dBFS, measured only between events: the figure a
        #: user can act on, to judge whether their gain or room will bury a step.
        self.background_db = -90.0
        #: Current signal level in dBFS.
        self.level_db = -90.0
        #: Latest detection strength in dB, the figure the threshold is applied to.
        self.flux_db = 0.0
        self._last_residual = np.zeros(self.n_fft // 2 + 1)
        self._excess = 0.0
        #: Time and strength of the last very loud event, used to hold the
        #: bar up while it is still ringing.
        self._big_hit_t = -10.0
        self._big_hit_flux = 0.0

    # ------------------------------------------------------------------ input
    def push(self, block: np.ndarray) -> list[Event]:
        """Add audio and return whatever events have finished being described.

        Events are reported a short while after they sound, once their tail has
        passed: a shot's decay and an explosion's rumble are the whole basis for
        telling them apart from a footstep, and neither exists yet at the moment
        of onset.
        """
        n = len(block)
        if n == 0:
            return []
        if n > self._ring.size - self.n_fft:
            block = block[-(self._ring.size - self.n_fft):]
            n = len(block)
        self._ring[:-n] = self._ring[n:]
        self._ring[-n:] = block
        self._stream += n

        for i in range(max(1, n // self.hop)):
            back = n - (i + 1) * self.hop
            if back < 0:
                break
            self._analyse(back)
        return self.drain()

    def _analyse(self, back: int) -> None:
        # A window reaching `back` samples into the past is only real audio once
        # that much has been pushed. Without this the first frames analyse the
        # zero-filled tail of the ring, which seeds the noise floor tens of dB
        # too low and makes the first real sound look like a gunshot.
        if self._stream < self.n_fft + back:
            return
        end = self._ring.size - back
        window = self._ring[end - self.n_fft:end]
        t = self._t0
        self._t0 += self.hop / self.rate

        spec = np.abs(np.fft.rfft(window * self._window)) ** 2 * self._gain
        db = 10.0 * np.log10(spec + EPS)

        if self._noise is None:
            self._noise = db.copy()
            self._prev_smooth = np.zeros_like(db)
            return

        alpha = np.where(db < self._noise, self.ALPHA_DOWN, self.ALPHA_UP)
        self._noise += alpha * (db - self._noise)

        # Detection runs on the *signed* residual, smoothed along frequency.
        # Clamping at zero first would make any bin that dips to the floor bounce
        # back with a large fake jump, and a few hundred of those noise wobbles
        # would swamp the handful of bins a real footstep moves. The unsmoothed,
        # clamped copy is what classification and display use, where the exact
        # spectral shape is the point.
        signed = db - self._noise
        smooth = np.convolve(signed, self._kernel, mode="same")
        delta = smooth - self._prev_smooth
        self._prev_smooth = smooth
        k = min(self.FLUX_TOP_K, delta.size)
        flux = float(np.partition(delta, -k)[-k:].mean())
        residual = np.maximum(signed, 0.0)
        self._last_residual = residual
        self.flux_db = flux
        # Clamped for display sanity: a perfectly silent capture is not
        # "-280 dBFS", it is simply nothing, and a number that absurd is worse
        # than useless when you are trying to judge your noise floor.
        self.level_db = float(np.clip(20.0 * np.log10(np.sqrt(spec.sum()) + EPS),
                                      FLOOR_DB, 0.0))

        frame = _Frame(t=t, power=spec, residual_db=residual,
                       level_db=float(20.0 * np.log10(np.sqrt(spec.sum()) + EPS)),
                       excess_db=float(residual.max()))
        self._frames.append(frame)

        if t - self._last_onset > 0.3:
            self._background += 0.05 * (self.level_db - self._background)
        self.background_db = float(np.clip(self._background, FLOOR_DB, 0.0))

        self._excess = frame.excess_db
        self._flux_hist.append(flux)
        self._detect(t, flux)
        self._advance(frame)

    @property
    def n_bins(self) -> int:
        """Length of the spectra this detector produces."""
        return self.n_fft // 2 + 1

    @property
    def residual_db(self) -> np.ndarray:
        """Current spectrum with the noise floor removed, for display."""
        return self._last_residual

    # --------------------------------------------------------------- detection
    def threshold(self) -> float:
        """Median + k*MAD of recent flux, so the trigger rides the background.

        This is what makes the sensitivity control meaningful: k scales how far
        above ordinary fluctuation a sound must stand before it counts.
        """
        if len(self._flux_hist) < 24:
            return float("inf")
        recent = np.fromiter(self._flux_hist, dtype=np.float64,
                             count=len(self._flux_hist))
        median = float(np.median(recent))
        mad = float(np.median(np.abs(recent - median)))
        k = 3.0 + 14.0 * float(np.clip(self.sensitivity, 0.0, 1.0))
        return median + k * 1.4826 * mad + 0.4

    #: A hit this strong leaves the noise floor pinned high for a good while
    #: afterwards, and the floor re-adjusting upward shows up as a fresh positive
    #: jump that mimics a new sound. This is an explosion's rumble re-triggering
    #: as a footstep half a second later.
    BIG_HIT_SNR = 75.0

    def _detect(self, t: float, flux: float) -> None:
        if flux < self.threshold():
            return
        if (t - self._big_hit_t > 0.15 and t - self._big_hit_t < 1.0
                and flux < self._big_hit_flux * 1.25):
            return
        # Require a local rise, so the leading edge of a long swell does not
        # register as its own event. Costs one hop of latency (10.7 ms).
        if len(self._flux_hist) < 2 or flux < self._flux_hist[-2]:
            return
        # One event per sound: a boot impact and its scuff are milliseconds
        # apart and would otherwise be counted twice.
        if t - self._last_onset < 0.06:
            return
        if self._pending:
            last = self._pending[-1]
            # Still loud from the sound already being described: this is its own
            # tail decaying, not a new event. Without this an explosion's rumble
            # re-triggers as a footstep half a second later.
            if self._excess > last.peak_db - 10.0:
                last.peak_flux = max(last.peak_flux, flux)
                return
            # A separate hit close behind the last one, but not markedly louder,
            # is that sound's own development rather than a second one. The
            # window is kept tight so that full automatic fire, ~0.12 s apart,
            # still registers round for round.
            if t - last.t < 0.11 and flux < last.peak_flux * 1.6:
                last.peak_flux = max(last.peak_flux, flux)
                return
        self._last_onset = t
        self._pending.append(_Pending(t=t, peak_flux=flux))
        # Only a genuinely loud event leaves a floor in this state; arming it on
        # every onset would suppress a run of ordinary footsteps, which sit well
        # below this level.
        if self._excess >= self.BIG_HIT_SNR:
            self._big_hit_t, self._big_hit_flux = t, flux

    def _advance(self, frame: _Frame) -> None:
        """Close pending events once their tail has passed, and report them."""
        if not self._pending:
            return
        still: list[_Pending] = []
        for p in self._pending:
            p.peak_db = max(p.peak_db, frame.excess_db)
            span = frame.t - p.t
            quiet = span >= MIN_S and frame.excess_db < p.peak_db - 18.0
            if quiet or span >= MEASURE_S:
                self._report(p)
            else:
                still.append(p)
        self._pending = still

    def _report(self, p: _Pending) -> None:
        window = [f for f in self._frames if p.t - 0.05 <= f.t <= p.t + MEASURE_S]
        if not window:
            return
        peak_frame = max(window, key=lambda f: float(f.residual_db.max()))
        power = peak_frame.power
        total = float(power.sum()) + EPS
        band_energy = np.array([power[a:b].sum() / total for a, b in self._band])

        times = np.array([f.t for f in window])
        levels = np.array([f.excess_db for f in window])
        onset_i = int(np.searchsorted(times, p.t))
        peak = float(levels.max())

        # Rise runs from the onset to the moment the sound reaches its peak;
        # decay runs from that peak down to 18 dB below it.
        reach = np.flatnonzero(levels[onset_i:] >= peak - 6.0)
        peak_i = onset_i + (int(reach[0]) if reach.size else 0)
        rise_s = float(times[peak_i] - p.t)
        fall = np.flatnonzero(levels[peak_i:] < peak - 18.0)
        decay_s = float(times[peak_i + int(fall[0])] - times[peak_i]) if fall.size \
            else float(times[-1] - times[peak_i])
        duration = float(max(rise_s + decay_s, self.hop / self.rate))

        snr_db = float(peak_frame.residual_db.max())
        level_db = float(20.0 * np.log10(np.sqrt(power.sum()) + EPS))
        centroid = float((np.arange(power.size) * power).sum() / total
                         * self.rate / self.n_fft)
        peak_hz = float(np.argmax(power) * self.rate / self.n_fft)
        # Spectral crest: peak bin against the average bin, the standard measure
        # of how concentrated a sound's energy is.
        crest = float(10.0 * np.log10(power.max() / (power.mean() + EPS)))
        air = float(band_energy[3] + band_energy[4])
        low = float(band_energy[0] + band_energy[1])

        kind = _classify(rise_s, decay_s, crest, snr_db, band_energy,
                         air, low, centroid, duration)
        event = Event(t=p.t, kind=kind, hz=peak_hz, db=level_db, snr_db=snr_db,
                      duration=duration,
                      confidence=float(np.clip(snr_db / 30.0, 0, 1)))
        self._out.append(event)
        if kind == FOOTSTEP:
            self._steps.append(p.t)
            self._gait()

    def _gait(self) -> None:
        if len(self._steps) < 3:
            return
        gaps = np.diff(list(self._steps))
        # A believable stride is 0.25-1.2 s; anything else is not a gait.
        gaps = gaps[(gaps > 0.25) & (gaps < 1.2)]
        if gaps.size == 0:
            return
        self.gait.steps_per_second = float(1.0 / np.median(gaps))
        self.gait.count = len(self._steps)
        self.gait.update_label()

    def drain(self) -> list[Event]:
        out, self._out = self._out, []
        return out


def _classify(rise_s: float, decay_s: float, crest: float, snr_db: float,
              band_energy: np.ndarray, air: float, low: float,
              centroid: float, duration: float) -> str:
    """Name a game sound from how it behaves in time and frequency.

    Band shape and duration do nearly all the work. Spectral crest is *not*
    used to separate gunfire from a reload: after a 42 ms window both are
    broadband and both measure ~9 dB, and the crest figure is just another way
    of saying "flat spectrum", which the band split already states directly.

    The order is deliberate. Depth and tail are tested first because an
    explosion is the one sound that could otherwise pass as a very loud
    footstep; a shot is then split from a weapon click by how long it rings.
    """
    # Explosions: deep, and a tail far longer than any foot or boot.
    if low > 0.5 and centroid < 900.0 and decay_s > 0.30:
        return EXPLOSION
    # Gunfire: bright and it rings. A reload click is just as bright but stops
    # almost immediately.
    if air > 0.15 and centroid > 2500.0 and duration > 0.07:
        return GUNSHOT
    # Weapon handling: bright and over in a few tens of milliseconds.
    if air > 0.15 and duration <= 0.07:
        return CLICK
    # A footstep is a low thump that dies quickly and carries almost nothing
    # above 2 kHz -- which is what separates it from a shot or a reload.
    if low > 0.40 and air < 0.20 and decay_s < 0.30:
        return FOOTSTEP
    if band_energy[2] > 0.3 and duration > 0.20:
        return VOICE
    return OTHER
