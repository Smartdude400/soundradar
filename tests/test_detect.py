"""Game-audio detection: the faint-footstep requirement, under test.

Signals here are synthesised to match the shapes the detector claims to
separate, and every one carries a known time, level and spectral tilt.
"""
import numpy as np
import pytest

from soundradar import detect
from soundradar.detect import Detector

RATE = 48000
BLOCK = 1024


# ----------------------------------------------------------------- synthesis
def footstep(rate=RATE, decay=0.055, thump_hz=150.0, brightness=0.25):
    """A boot impact: a low resonant thump with a short scuff on top."""
    n = int(rate * 0.18)
    t = np.arange(n) / rate
    env = np.exp(-t / decay)
    thump = np.sin(2 * np.pi * thump_hz * t) * env
    noise = rng().standard_normal(n) * env * brightness
    # Weight the noise down so the thump dominates, as a real footstep does.
    noise = lowpass(noise, 1800.0, rate)
    out = thump + noise
    return (out / np.abs(out).max()).astype(np.float32)


def gunshot(rate=RATE, decay=0.11):
    """A shot: an instantaneous crack across the whole spectrum plus a body."""
    n = int(rate * 0.30)
    t = np.arange(n) / rate
    env = np.exp(-t / decay)
    crack = rng().standard_normal(n) * env
    crack[: int(rate * 0.001)] *= 12.0            # the muzzle blast itself
    body = np.sin(2 * np.pi * 85.0 * t) * env * 0.6
    out = crack + body
    return (out / np.abs(out).max()).astype(np.float32)


def explosion(rate=RATE, decay=0.30):
    n = int(rate * 0.8)
    t = np.arange(n) / rate
    env = np.exp(-t / decay)
    out = lowpass(rng().standard_normal(n), 350.0, rate) * env
    out += np.sin(2 * np.pi * 60.0 * t) * env * 0.8
    return (out / np.abs(out).max()).astype(np.float32)


def click(rate=RATE):
    n = int(rate * 0.05)
    out = np.zeros(n, dtype=np.float64)
    out[: int(rate * 0.002)] = rng().standard_normal(int(rate * 0.002)) * 8
    out = highpass(out, 1200.0, rate)
    return (out / np.abs(out).max()).astype(np.float32)


def lowpass(x, hz, rate):
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / rate)
    return np.fft.irfft(spec * (freqs < hz), len(x))


def highpass(x, hz, rate):
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / rate)
    return np.fft.irfft(spec * (freqs > hz), len(x))


def rng():
    return np.random.default_rng(4242)


def scene(events, seconds=6.0, noise_db=-85.0, rate=RATE, seed=7):
    """Place timed events over a noise bed.

    ``level_db`` is the event's **RMS** level in dBFS, which is the figure the
    UI shows and the one a user can reason about. Scaling by peak amplitude
    instead would understate a footstep by its crest factor -- about 10 dB --
    and quietly turn a demanding test into an impossible one.
    """
    r = np.random.default_rng(seed)
    n = int(seconds * rate)
    bed = r.standard_normal(n) * (10 ** (noise_db / 20.0))
    placed = []
    for t, kind, level_db in events:
        sig = {"step": footstep, "gun": gunshot,
               "boom": explosion, "click": click}[kind](rate)
        rms = float(np.sqrt(np.mean(sig.astype(np.float64) ** 2))) + 1e-12
        sig = sig * (10 ** (level_db / 20.0) / rms)
        start = int(t * rate)
        span = min(len(sig), n - start)
        if span <= 0:
            continue
        bed[start:start + span] += sig[:span]
        placed.append((t, kind))
    return bed.astype(np.float32), placed


def run(audio, rate=RATE, sensitivity=0.35):
    det = Detector(rate, sensitivity=sensitivity)
    found = []
    for i in range(0, len(audio) - BLOCK, BLOCK):
        found.extend(det.push(audio[i:i + BLOCK]))
    return det, found


# --------------------------------------------------------------------- tests
def test_detects_a_footstep_far_below_any_usable_level_gate():
    """The headline requirement: a -70 dBFS footstep on a -78 dBFS floor."""
    audio, placed = scene([(2.0, "step", -70.0)], seconds=4.0)
    _, found = run(audio)
    steps = [e for e in found if e.kind == detect.FOOTSTEP]
    assert steps, f"faint footstep missed; got {[(e.kind, round(e.snr_db,1)) for e in found]}"
    assert abs(steps[0].t - 2.0) < 0.08
    assert steps[0].snr_db > 5.0, "detected, but not meaningfully above the floor"


def test_still_detects_faint_steps_when_gunfire_is_in_the_same_stream():
    """The whole point of residual detection: a loud event must not raise the
    bar for everything after it."""
    audio, _ = scene([(0.7, "gun", -8.0), (2.0, "step", -70.0),
                      (3.0, "gun", -10.0), (4.2, "step", -68.0)], seconds=6.0)
    _, found = run(audio)
    assert any(e.kind == detect.GUNSHOT for e in found)
    steps = [e for e in found if e.kind == detect.FOOTSTEP]
    assert len(steps) >= 2, f"gunfire masked the footsteps: {[(e.kind, round(e.t,2)) for e in found]}"


def test_quiet_noise_alone_does_not_trigger():
    """The cost of a low threshold is false positives; this pins that cost."""
    r = np.random.default_rng(99)
    audio = (r.standard_normal(int(5.0 * RATE)) * 10 ** (-80.0 / 20.0)).astype(np.float32)
    _, found = run(audio, sensitivity=0.35)
    assert len(found) == 0, f"{len(found)} false detections on pure noise"


def test_a_steady_tone_is_absorbed_into_the_floor_not_repeatedly_detected():
    """Music or a hum must not fire forever."""
    t = np.arange(int(5.0 * RATE)) / RATE
    audio = (0.02 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    _, found = run(audio, sensitivity=0.35)
    assert len(found) <= 1, f"sustained tone produced {len(found)} detections"


def test_classifies_gunfire_and_footsteps_apart():
    audio, _ = scene([(1.0, "gun", -12.0), (2.5, "step", -45.0)], seconds=4.0)
    _, found = run(audio)
    kinds = [e.kind for e in found]
    assert detect.GUNSHOT in kinds, f"gunshot not recognised: {kinds}"
    assert detect.FOOTSTEP in kinds, f"footstep not recognised: {kinds}"


def test_classifies_an_explosion():
    audio, _ = scene([(1.5, "boom", -14.0)], seconds=4.0)
    _, found = run(audio)
    assert any(e.kind == detect.EXPLOSION for e in found), \
        f"explosion not recognised: {[(e.kind, round(e.snr_db,1)) for e in found]}"


def test_sensitivity_trade_is_real():
    """Permissive settings find more, stricter settings find fewer."""
    events = [(1.0 + i * 0.9, "step", -74.0) for i in range(5)]
    audio, _ = scene(events, seconds=6.0, noise_db=-76.0)
    _, strict = run(audio, sensitivity=0.9)
    _, loose = run(audio, sensitivity=0.05)
    assert len(loose) >= len(strict)


def test_footstep_cadence_becomes_a_gait():
    """Four steps at 0.5 s spacing is a 2 steps/s run."""
    audio, _ = scene([(1.0 + i * 0.5, "step", -50.0) for i in range(5)], seconds=5.0)
    det, found = run(audio)
    assert len([e for e in found if e.kind == detect.FOOTSTEP]) >= 3
    assert det.gait.steps_per_second == pytest.approx(2.0, rel=0.25)
    assert det.gait.label == "running"


def test_reported_background_level_tracks_the_bed():
    audio, _ = scene([], seconds=4.0, noise_db=-85.0)
    det, _ = run(audio)
    assert det.background_db == pytest.approx(-85.0, abs=4.0)


def test_push_handles_blocks_of_any_size():
    r = np.random.default_rng(3)
    audio = (r.standard_normal(RATE) * 0.01).astype(np.float32)
    for block in (256, 512, 1024, 4096):
        det = Detector(RATE)
        for i in range(0, len(audio) - block, block):
            det.push(audio[i:i + block])
        assert det.background_db < 0.0    # a background level was established
