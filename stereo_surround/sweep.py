"""Sequential repeated benchmark matrix; python -m stereo_surround.sweep --output results.json."""

import argparse
import hashlib
import json
from pathlib import Path

from .benchmark import benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0 or args.repeats <= 0:
        parser.error("seconds and repeats must be positive")
    cases = []
    for rate in (44100, 48000, 96000):
        for profile in ("low_latency", "balanced", "quality"):
            for mode in ("ambience", "matrix"):
                cases.append((rate, round(rate/60), profile, mode, False))
    for block in (128, 256, 512, 1024):
        for mode in ("ambience", "matrix"):
            cases.append((48000, block, "balanced", mode, False))
    cases.append((48000, 800, "balanced", "ambience", True))
    path = Path(__file__).with_name("decoder.py")
    report = {"decoder_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "seconds_per_run": args.seconds, "repeats": args.repeats, "results": []}
    # Refuse to overwrite a prior report. Save incrementally for interrupted runs.
    with args.output.open("x", encoding="utf-8") as output:
        for rate, block, profile, mode, adaptive in cases:
            rows = []
            for repeat in range(args.repeats):
                row = benchmark(args.seconds, block, rate, profile, mode, adaptive)
                report["environment"] = row.pop("environment")
                row["repeat"] = repeat+1
                report["results"].append(row)
                rows.append(row)
            print(json.dumps({"rate": rate, "block": block, "profile": profile, "mode": mode,
                              "adaptive": adaptive, "rtf_max": max(r["rtf"] for r in rows),
                              "p99_ms_max": max(r["p99_ms"] for r in rows),
                              "deadline_misses": sum(r["deadline_misses"] for r in rows)}), flush=True)
            output.seek(0)
            json.dump(report, output, indent=2)
            output.truncate()
            output.flush()


if __name__ == "__main__":
    main()
