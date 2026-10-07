"""Reproducible DSP timing. python -m stereo_surround.benchmark --seconds 60."""

import argparse
import json
import platform
import time

import numpy as np
import scipy

from .decoder import SurroundDecoder


def benchmark(seconds=60, block=800, rate=48000, profile="balanced", mode="ambience", onset_adaptation=False,
              *, listening_mode="standard", center_width=0.0):
    decoder = SurroundDecoder(rate, profile=profile, mode=mode, onset_adaptation=onset_adaptation,
                              listening_mode=listening_mode, center_width=center_width)
    x = np.random.default_rng(42).normal(0, .1, (block, 2)).astype(np.float32)
    for _ in range(100):
        decoder.process(x)
    count = max(100, int(seconds*rate/block))
    times = np.empty(count)
    fft_calls = np.zeros(count, dtype=bool)
    for i in range(count):
        before = decoder._fill
        start = time.perf_counter_ns()
        decoder.process(x)
        times[i] = (time.perf_counter_ns()-start)*1e-6
        fft_calls[i] = before+block >= decoder.hop_size
    duration = block/rate*1000
    return {"rate": rate, "block": block, "profile": profile, "mode": mode,
            "listening_mode": listening_mode, "center_width": center_width,
            "onset_adaptation": onset_adaptation,
            "audio_seconds": count*block/rate, "rtf": float(times.sum()/(count*duration)),
            "p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)), "max_ms": float(times.max()),
            "deadline_misses": int(np.count_nonzero(times > duration)),
            "fft_calls": int(np.count_nonzero(fft_calls)),
            "fft_call_p99_ms": float(np.percentile(times[fft_calls], 99)) if fft_calls.any() else None,
            "latency_samples": decoder.latency_samples,
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "os": platform.system(),
                            "fft_workers": 1}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--block", type=int, default=800)
    parser.add_argument("--rate", type=int, default=48000)
    parser.add_argument("--profile", choices=("low_latency", "balanced", "quality"), default="balanced")
    parser.add_argument("--mode", choices=("ambience", "matrix"), default="ambience")
    parser.add_argument("--onset-adaptation", action="store_true")
    parser.add_argument("--listening-mode", choices=("standard", "music"), default="standard")
    parser.add_argument("--center-width", type=float, default=0.0)
    args = parser.parse_args()
    if args.seconds <= 0 or args.block <= 0:
        parser.error("seconds and block must be positive")
    print(json.dumps(benchmark(args.seconds, args.block, args.rate, args.profile, args.mode,
                               args.onset_adaptation, listening_mode=args.listening_mode,
                               center_width=args.center_width), indent=2))


if __name__ == "__main__":
    main()
