"""Direction finding from a microphone array.

Two channels are the minimum that carry any positional information at all --
a single channel cannot distinguish "in front" from "behind", and nothing in
this module pretends otherwise. With a pair of microphones the time difference
of arrival gives a real angle; the maths is GCC-PHAT followed by a least
squares solve against the known array geometry.

Sign convention
---------------
Right-handed frame: ``x`` right, ``y`` forward, ``z`` up. A source direction
``u`` is a unit vector pointing from the array towards the source. For mics at
``p_a`` and ``p_b``, far-field geometry gives

    delay_b_relative_to_a = t_b - t_a = -((p_b - p_a) . u) / c

so a source on the ``+x`` side reaches the right-hand mic *first*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_AIR = 343.0
EPS = 1e-12


@dataclass(frozen=True)
class Pair:
    """A baseline between two channels: mic ``b`` sits ``(dx, dy, dz)`` from ``a``."""

    a: int
    b: int
    dx: float
    dy: float = 0.0
    dz: float = 0.0


def gcc_phat(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[float, float]:
    """GCC-PHAT delay of *b* relative to *a*, in samples, plus a 0..1 quality.

    Returns ``(delay, confidence)`` where ``delay > 0`` means ``b`` hears the
    sound *later* than ``a``. The phase transform whitens the spectrum, which
    turns a broad correlation mound in a reverberant room into a single sharp
    peak and makes the peak a far better estimate than plain cross-correlation.
    """
    n = min(len(a), len(b))
    a = np.asarray(a[:n], dtype=np.float64)
    b = np.asarray(b[:n], dtype=np.float64)
    if n < 8 or not (np.any(a) or np.any(b)):
        return 0.0, 0.0

    nfft = 1 << (2 * n - 1).bit_length()
    fa = np.fft.rfft(a, nfft)
    fb = np.fft.rfft(b, nfft)
    cross = fa * np.conj(fb)
    cross /= np.abs(cross) + EPS
    cc = np.fft.irfft(cross, nfft)

    # Zero padding makes the circular result linear over the lags we use:
    # negative lags live at the tail, positive at the head.
    lin = np.concatenate([cc[-(n - 1) :], cc[:n]])
    lags = np.arange(-(n - 1), n)  # signed lead of a over b, negated below
    keep = np.abs(lags) <= max_lag
    win = np.where(keep, lin, -np.inf)
    k = int(np.argmax(win))

    # `lin` is indexed by the lead of a over b, so negate to report the delay
    # of b relative to a, as documented.
    delay = -float(lags[k])
    if 0 < k < win.size - 1 and np.isfinite(win[k - 1]) and np.isfinite(win[k + 1]):
        span, rise = float(win[k - 1]), float(win[k + 1])
        denom = span - 2.0 * float(win[k]) + rise
        if abs(denom) > EPS:  # sub-sample refinement across the peak
            offset = 0.5 * (span - rise) / denom
            if abs(offset) <= 1.0:
                delay -= offset

    quality = float(win[k]) / (float(np.mean(np.abs(lin[keep]))) + EPS)
    return delay, float(np.clip(quality - 1.0, 0.0, 1.0))


@dataclass(frozen=True)
class Geometry:
    """Channel pairs and the physical spacing between them."""

    pairs: tuple[Pair, ...]

    @property
    def channels(self) -> int:
        return max(max(p.a, p.b) for p in self.pairs) + 1

    @property
    def is_linear(self) -> bool:
        """True when every baseline is parallel -- elevation is then unknowable."""
        first = np.array([self.pairs[0].dx, self.pairs[0].dy, self.pairs[0].dz])
        n = float(np.linalg.norm(first))
        if n < EPS:
            return True
        for p in self.pairs[1:]:
            d = np.array([p.dx, p.dy, p.dz])
            if float(np.linalg.norm(np.cross(first / n, d))) > 1e-6:
                return False
        return True

    def max_lag(self, rate: float) -> int:
        longest = max(float(np.linalg.norm([p.dx, p.dy, p.dz])) for p in self.pairs)
        return max(1, int(np.ceil(longest / C_AIR * rate)))

    def solve(
        self, delays: np.ndarray, confidences: np.ndarray, rate: float
    ) -> tuple[float, float | None, float]:
        """Delays over pairs to ``(angle_deg, elevation_deg or None, confidence)``.

        Azimuth runs CCW from ``+x`` (right). A linear array only ever resolves
        the component of ``u`` along its baseline, so it returns a *cone angle*
        measured from that baseline and ``None`` for elevation rather than
        inventing a second axis it cannot see. Such a contact could genuinely
        sit anywhere in the mirrored cone, and the UI draws it that way.
        """
        weights = np.asarray(confidences, dtype=np.float64)
        delays = np.asarray(delays, dtype=np.float64)
        if weights.sum() < 0.15:
            return 0.0, None, 0.0
        quality = float(np.clip(weights.mean(), 0.0, 1.0))
        rows = np.array([[p.dx, p.dy, p.dz] for p in self.pairs], dtype=np.float64)
        lengths = np.linalg.norm(rows, axis=1)
        # Delay is in samples; convert to seconds before applying the speed of
        # sound, otherwise every angle saturates at the array's endfire.
        seconds = delays / rate

        if self.is_linear:
            # Every baseline shares one axis, so the only observable is the
            # unit vector's component along it. Solving this directly also
            # keeps a legitimately zero TDOA (source on the axis) meaningful,
            # where normalising a least-squares fit would divide by nothing.
            axis = rows[0] / max(float(lengths[0]), EPS)
            comp = float(np.sum((-C_AIR * seconds / lengths) * weights) / weights.sum())
            return (
                float(np.degrees(np.arcsin(np.clip(axis[0] * comp, -1.0, 1.0)))),
                None,
                quality,
            )

        # Weighted normal equations keep a single clipped or garbage pair from
        # dragging the whole solution.
        rw = rows * np.sqrt(weights)[:, None]
        bw = -C_AIR * seconds * np.sqrt(weights)
        u, *_ = np.linalg.lstsq(rw, bw, rcond=None)
        norm = float(np.linalg.norm(u))
        if norm < EPS:
            return 0.0, None, 0.0
        u = u / norm
        azimuth = float(np.degrees(np.arctan2(u[1], u[0])))
        elevation = (
            float(np.degrees(np.arcsin(np.clip(u[2], -1.0, 1.0))))
            if self.observes_elevation
            else None
        )
        return azimuth, elevation, quality

    @property
    def observes_elevation(self) -> bool:
        """Elevation needs a baseline with height. Mics laid out flat -- even a
        square -- leave the vertical axis in the null space of the delays, so
        any elevation fitted from them is an artefact of the regularisation."""
        return any(abs(p.dz) > 1e-6 for p in self.pairs)


def linear_array(channels: int, spacing_m: float) -> Geometry:
    """Evenly spaced collinear mics along ``x`` -- stereo mics, a two-capsule
    array, or a multi-capsule USB interface. Resolves a cone angle only."""
    return Geometry(
        tuple(Pair(i, i + 1, spacing_m) for i in range(max(1, channels) - 1))
    )


def square_array(spacing_m: float) -> Geometry:
    """Four mics in a flat square. Two non-parallel axes give a true azimuth;
    elevation stays ``None`` because no baseline has any height."""
    s = spacing_m
    return Geometry(
        (
            Pair(0, 1, s),
            Pair(2, 3, s),
            Pair(0, 2, 0.0, s),
            Pair(1, 3, 0.0, s),
        )
    )


def tetra_array(spacing_m: float, height_m: float) -> Geometry:
    """Three mics on a horizontal triangle plus a fourth raised above them.

    This is the smallest layout that actually resolves elevation: the lifted
    mic supplies the vertical baseline the flat layouts cannot.
    """
    s, h = spacing_m, height_m
    root3 = np.sqrt(3.0)
    return Geometry(
        (
            Pair(0, 1, s, 0.0, 0.0),
            Pair(1, 2, -s / 2.0, s * root3 / 2.0, 0.0),
            Pair(0, 2, s / 2.0, s * root3 / 2.0, 0.0),
            Pair(0, 3, s / 2.0, s / (2.0 * root3), h),
            Pair(1, 3, -s / 2.0, s / (2.0 * root3), h),
            Pair(2, 3, 0.0, -s / root3, h),
        )
    )
