"""Enumerating the audio endpoints this machine actually has."""

from __future__ import annotations

from dataclasses import dataclass

import pyaudiowpatch as pa


@dataclass(frozen=True)
class Device:
    """An endpoint that can be captured from."""

    index: int
    name: str
    channels: int
    rate: int
    loopback: bool = False

    @property
    def kind(self) -> str:
        return "system audio" if self.loopback else "microphone"


def _to_device(info: dict, loopback: bool) -> Device:
    return Device(
        index=info["index"],
        name=info["name"],
        channels=info["maxInputChannels"],
        rate=int(info["defaultSampleRate"]),
        loopback=loopback,
    )


def inputs(pyaudio: pa.PyAudio | None = None) -> list[Device]:
    """Every capture-capable endpoint, mics first, then loopback taps.

    WASAPI loopback is how any audio the machine is *playing* is fed back in as
    ordinary samples, so the tool can watch music or a game exactly as the
    speakers receive it.
    """
    own = pyaudio is None
    pa_ = pyaudio or pa.PyAudio()
    try:
        found: list[Device] = []
        for info in pa_.get_device_info_generator():
            if info["maxInputChannels"] > 0 and not info.get("isLoopbackDevice"):
                found.append(_to_device(info, False))
        for info in pa_.get_loopback_device_info_generator():
            found.append(_to_device(info, True))
        return found
    finally:
        if own:
            pa_.terminate()


def default_input(pyaudio: pa.PyAudio | None = None) -> Device | None:
    own = pyaudio is None
    pa_ = pyaudio or pa.PyAudio()
    try:
        info = pa_.get_default_input_device_info()
        return _to_device(info, bool(info.get("isLoopbackDevice")))
    except Exception:
        return None
    finally:
        if own:
            pa_.terminate()


def default_output_index(pyaudio: pa.PyAudio | None = None) -> int | None:
    own = pyaudio is None
    pa_ = pyaudio or pa.PyAudio()
    try:
        return pa_.get_default_output_device_info()["index"]
    except Exception:
        return None
    finally:
        if own:
            pa_.terminate()


def render_index_for(loopback: Device, pyaudio: pa.PyAudio | None = None) -> int | None:
    """The speaker a loopback tap is actually listening to.

    The default output is frequently a *different* endpoint from the one a given
    loopback device mirrors -- a dongle and a wireless receiver, say. Opening
    the keep-alive sink on the wrong one leaves the tap receiving pure silence
    with nothing to show for it, so match on the name instead.

    Windows exposes the same speaker more than once, at each rate it supports,
    under one identical name. The tap captures at its own rate, so a 48 kHz
    loopback fed by a 44.1 kHz render endpoint captures silence; the rate has to
    be part of the match, not just the name.
    """
    own = pyaudio is None
    pa_ = pyaudio or pa.PyAudio()
    try:
        want = loopback.name.replace("[Loopback]", "").strip().lower()
        fallback = None
        for info in pa_.get_device_info_generator():
            if info["maxOutputChannels"] <= 0 or info["name"].strip().lower() != want:
                continue
            if int(info["defaultSampleRate"]) == loopback.rate:
                return info["index"]
            if fallback is None:
                fallback = info["index"]
        return fallback if fallback is not None else default_output_index(pa_)
    finally:
        if own:
            pa_.terminate()





def describe(device: Device) -> str:
    return f"{device.name}  ({device.kind}, {device.channels}ch, {device.rate} Hz)"
