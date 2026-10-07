"""Accelerated continuous-audio and Python allocation check.

python -m stereo_surround.longrun --seconds 1800
The elapsed audio time differs from wall-clock time; not a hardware soak test.
"""

import argparse
import json
import time
import tracemalloc

import numpy as np

from .decoder import SurroundDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=1800)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("seconds must be positive and finite")
    rng = np.random.default_rng(123)
    decoder = SurroundDecoder()
    x = rng.normal(0, .1, (800, 2)).astype(np.float32)
    for _ in range(100):
        decoder.process(x)
    tracemalloc.start()
    start = time.perf_counter()
    maximum_queue = 0
    count = int(np.ceil(args.seconds*60))
    for i in range(count):
        x[:] = rng.normal(0, .1, x.shape)
        y = decoder.process(x)
        if not np.isfinite(y).all():
            raise RuntimeError("Non-finite output")
        maximum_queue = max(maximum_queue,
                            sum(len(z) for z in decoder._queue)-decoder._queue_offset)
        if (i+1) % 18000 == 0 or i+1 == count:
            current, peak = tracemalloc.get_traced_memory()
            print(json.dumps({"audio_seconds": (i+1)/60, "current_traced_bytes": current,
                              "peak_traced_bytes": peak, "max_queue_samples": maximum_queue,
                              "wall_seconds": round(time.perf_counter()-start, 2)}), flush=True)


if __name__ == "__main__":
    main()
