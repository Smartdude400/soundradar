"""Direction finding: does it recover angles it was never told?"""
import numpy as np
import pytest

from soundradar.doa import (C_AIR, gcc_phat, linear_array, square_array,
                            tetra_array)

RATE = 48000
N = 4096
NOISE = np.random.default_rng(20240513).standard_normal(N)


def delayed(source: np.ndarray, samples: float) -> np.ndarray:
    """A copy of *source* shifted later by a fractional number of samples."""
    return np.interp(np.arange(len(source)) - samples, np.arange(len(source)), source)


def cone_angle(geo, delay, rate):
    return geo.solve(np.array([delay]), np.array([1.0]), rate)[0]


@pytest.mark.parametrize("delay", [-27.0, -9.5, -1.25, 0.0, 3.75, 18.0, 26.5])
def test_gcc_phat_recovers_known_delay(delay):
    found, quality = gcc_phat(NOISE, delayed(NOISE, delay), max_lag=40)
    # GCC-PHAT resolves a whole sample exactly and fractional delays to about a
    # quarter of one. On a 0.20 m baseline that is 0.04 degrees of bearing, far
    # below anything the geometry could ever use.
    assert found == pytest.approx(delay, abs=0.25)
    assert 0.9 <= quality <= 1.0


def test_gcc_phat_sign_puts_later_signal_on_positive_delay():
    """The whole bearing solution depends on this sign; pin it explicitly."""
    found, _ = gcc_phat(NOISE, delayed(NOISE, 12.0), max_lag=40)
    assert found > 0, "a signal arriving later must report a positive delay"
    found, _ = gcc_phat(delayed(NOISE, 12.0), NOISE, max_lag=40)
    assert found < 0


def test_gcc_phat_survives_silence():
    quiet = np.zeros(N)
    assert gcc_phat(quiet, quiet, max_lag=40) == (0.0, 0.0)


def test_max_lag_tracks_baseline_length():
    assert linear_array(2, 0.20).max_lag(RATE) == pytest.approx(28, abs=1)
    assert linear_array(2, 0.40).max_lag(RATE) == pytest.approx(56, abs=1)


@pytest.mark.parametrize("cone", [-70, -40, -15, 0, 15, 40, 70])
def test_linear_array_recovers_cone_angle(cone):
    spacing = 0.20
    geo = linear_array(2, spacing)
    true_delay = -(spacing * np.sin(np.radians(cone))) / C_AIR * RATE
    found, quality = gcc_phat(NOISE, delayed(NOISE, true_delay), geo.max_lag(RATE))
    assert cone_angle(geo, found, RATE) == pytest.approx(cone, abs=2.0)
    assert quality > 0.5


def test_linear_array_refuses_to_invent_elevation():
    geo = linear_array(2, 0.20)
    assert geo.is_linear and not geo.observes_elevation
    _, elevation, _ = geo.solve(np.array([5.0]), np.array([1.0]), RATE)
    assert elevation is None


def test_flat_square_resolves_azimuth_but_not_elevation():
    """A horizontal array leaves the vertical axis unobservable; reporting a
    number there would be fitting noise."""
    geo = square_array(0.20)
    assert not geo.is_linear
    assert not geo.observes_elevation
    positions = [(-0.1, -0.1, 0.0), (0.1, -0.1, 0.0), (-0.1, 0.1, 0.0), (0.1, 0.1, 0.0)]
    for azimuth in (-150, -60, 30, 90, 180):
        u = np.array([np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth)), 0.0])
        channels = [delayed(NOISE, -float(np.dot(p, u)) / C_AIR * RATE) for p in positions]
        delays, confidences = [], []
        for pair in geo.pairs:
            d, c = gcc_phat(channels[pair.a], channels[pair.b], geo.max_lag(RATE))
            delays.append(d)
            confidences.append(c)
        solved, elevation, _ = geo.solve(np.array(delays), np.array(confidences), RATE)
        assert elevation is None
        assert solved == pytest.approx(azimuth, abs=2.0)


def test_lifted_array_resolves_azimuth_and_elevation():
    spacing, height = 0.20, 0.25
    geo = tetra_array(spacing, height)
    assert geo.observes_elevation
    root3 = np.sqrt(3.0)
    positions = [(-spacing / 2, -spacing / (2 * root3), 0.0),
                 (spacing / 2, -spacing / (2 * root3), 0.0),
                 (0.0, spacing / root3, 0.0),
                 (0.0, 0.0, height)]
    for azimuth, elevation in ((30, 10), (90, 0), (-60, 25), (-135, 40)):
        el, az = np.radians(elevation), np.radians(azimuth)
        u = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        channels = [delayed(NOISE, -float(np.dot(p, u)) / C_AIR * RATE) for p in positions]
        delays, confidences = [], []
        for pair in geo.pairs:
            d, c = gcc_phat(channels[pair.a], channels[pair.b], geo.max_lag(RATE))
            delays.append(d)
            confidences.append(c)
        got_az, got_el, _ = geo.solve(np.array(delays), np.array(confidences), RATE)
        assert got_az == pytest.approx(azimuth, abs=3.0)
        assert got_el == pytest.approx(elevation, abs=3.0)


def test_low_confidence_gives_no_bearing():
    """Below the noise gate the tool must stay silent rather than point somewhere."""
    geo = square_array(0.20)
    azimuth, elevation, quality = geo.solve(np.array([1.0, 1.0, 1.0, 1.0]),
                                           np.array([0.01, 0.01, 0.01, 0.01]), RATE)
    assert quality == 0.0
    assert elevation is None
