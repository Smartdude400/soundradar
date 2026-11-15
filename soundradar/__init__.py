"""SoundRadar -- hear what a game is telling you, even when it is quiet.

Built around one idea: never threshold absolute level. Every frequency bin is
compared against its own running noise floor, so a footstep 6 dB over a busy
room is found as reliably as a gunshot in a silent one, and a steady background
stops generating detections entirely.
"""
from . import detect, devices, doa, offline  # noqa: F401

__version__ = "2.0.0"
