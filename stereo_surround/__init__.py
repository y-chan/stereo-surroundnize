"""Stateful NumPy/SciPy stereo upmix. Channel order: FL FR FC LFE SL SR."""

from .decoder import CHANNELS, SurroundDecoder

__all__ = ["CHANNELS", "SurroundDecoder"]
