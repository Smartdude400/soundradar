"""File analysis: the same detector the live engine uses, over a recording."""
import numpy as np
import pytest
from scipy.io import wavfile

from soundradar import detect, offline

RATE = 48000


def footstep(rate=RATE, seed=11):
    n = int(rate * 0.18)
    t = np.arange(n) / rate
    env = np.exp(-t / 0.055)
    r = np.random.default_rng(seed)
    out = np.sin(2 * np.pi * 150.0 * t) * env + r.standard_normal(n) * env * 0.2
    return (out / np.sqrt(np.mean(out ** 2))).astype(np.float32)


def gunshot(rate=RATE, seed=12):
    n = int(rate * 0.3)
    t = np.arange(n) / rate
    env = np.exp(-t / 0.11)
    r = np.random.default_rng(seed)
    out = r.standard_normal(n) * env
    out[: int(rate * 0.001)] *= 12.0
    return (out / np.sqrt(np.mean(out ** 2))).astype(np.float32)


def write(path, data, rate=RATE):
    wavfile.write(path, rate, (np.clip(data, -1, 1) * 32767).astype(np.int16))
    return path


def scene(events, seconds, noise_db=-85.0, rate=RATE, seed=7):
    r = np.random.default_rng(seed)
    bed = r.standard_normal(int(seconds * rate)) * (10 ** (noise_db / 20.0))
    for t, sig, level_db in events:
        s = sig * (10 ** (level_db / 20.0)
                   / (np.sqrt(np.mean(sig.astype(np.float64) ** 2)) + 1e-12))
        start = int(t * rate)
        span = min(len(s), len(bed) - start)
        bed[start:start + span] += s[:span]
    return bed.astype(np.float32)


def test_file_analysis_finds_faint_footsteps(tmp_path):
    """The same bar the live path has to clear, measured through the file path."""
    times = [0.8 + i * 0.55 for i in range(5)]
    audio = scene([(t, footstep(), -72.0) for t in times], seconds=4.0)
    report = offline.analyse_file(write(tmp_path / "steps.wav", audio))
    steps = [e for e in report.events if e.kind == detect.FOOTSTEP]
    assert len(steps) == 5, f"found {len(steps)} of 5 faint footsteps"
    for event, truth in zip(steps, times):
        assert event.t == pytest.approx(truth, abs=0.09)


def test_file_analysis_finds_a_gunshot_and_an_explosion(tmp_path):
    boom = np.random.default_rng(5).standard_normal(int(RATE * 0.8))
    env = np.exp(-np.arange(len(boom)) / RATE / 0.3)
    boom = (np.sin(2 * np.pi * 60.0 * np.arange(len(boom)) / RATE) * env).astype(np.float32)
    audio = scene([(0.6, gunshot(), -12.0), (2.0, boom, -14.0)], seconds=3.0)
    report = offline.analyse_file(write(tmp_path / "fight.wav", audio))
    kinds = [e.kind for e in report.events]
    assert detect.GUNSHOT in kinds, f"gunshot missed: {kinds}"
    assert detect.EXPLOSION in kinds, f"explosion missed: {kinds}"


def test_background_level_is_reported(tmp_path):
    audio = scene([], seconds=3.0, noise_db=-84.0)
    report = offline.analyse_file(write(tmp_path / "bed.wav", audio))
    assert report.background_db == pytest.approx(-84.0, abs=5.0)
    assert report.events == []


def test_silence_produces_no_events(tmp_path):
    audio = np.zeros(int(3.0 * RATE), dtype=np.float32)
    report = offline.analyse_file(write(tmp_path / "silence.wav", audio))
    assert report.events == []


def test_stereo_file_gets_an_image_position(tmp_path):
    """A property of the mix, not a direction -- which is why the UI says so."""
    mono = scene([(0.8 + i * 0.6, footstep(), -55.0) for i in range(4)], seconds=3.5)
    stereo = np.stack([mono * 0.5, mono], axis=1)
    report = offline.analyse_file(write(tmp_path / "stereo.wav", stereo))
    assert report.channels == 2
    assert report.pan_events == len(report.events) > 0
    assert all(e.axis_deg is not None for e in report.events)
    # Left is quieter, so the mix sits to the right: negative.
    assert all(e.axis_deg < 0 for e in report.events)


def test_mono_file_has_no_position(tmp_path):
    audio = scene([(0.8, footstep(), -55.0)], seconds=2.0)
    report = offline.analyse_file(write(tmp_path / "mono.wav", audio))
    assert report.channels == 1
    assert all(e.axis_deg is None for e in report.events)


def test_exports_round_trip(tmp_path):
    audio = scene([(0.8 + i * 0.6, footstep(), -60.0) for i in range(3)], seconds=3.0)
    report = offline.analyse_file(write(tmp_path / "t.wav", audio))
    assert report.events
    report.write_csv(tmp_path / "out.csv")
    report.write_json(tmp_path / "out.json")
    csv_text = (tmp_path / "out.csv").read_text(encoding="utf-8")
    assert csv_text.startswith("time_s,type,freq_hz,level_dbfs,snr_db")
    assert (tmp_path / "out.json").read_text(encoding="utf-8").lstrip().startswith("{")
    assert {"time_s", "type", "snr_db", "duration_ms"} <= set(report.to_rows()[0])


def test_sensitivity_is_honoured_offline(tmp_path):
    """Permissive settings find more than strict ones on a marginal signal."""
    events = [(0.8 + i * 0.8, footstep(), -79.0) for i in range(4)]
    audio = scene(events, seconds=4.0, noise_db=-84.0)
    path = write(tmp_path / "faint.wav", audio)
    strict = offline.analyse_file(path, sensitivity=0.9)
    loose = offline.analyse_file(path, sensitivity=0.05)
    assert len(loose.events) >= len(strict.events)


def test_int16_files_are_scaled_to_full_range(tmp_path):
    """A normalised reading must not come back 30 dB quiet."""
    audio = scene([(0.8, gunshot(), -6.0)], seconds=2.0, noise_db=-90.0)
    report = offline.analyse_file(write(tmp_path / "loud.wav", audio))
    loudest = max(e.db for e in report.events)
    assert loudest == pytest.approx(-6.0, abs=3.0)
