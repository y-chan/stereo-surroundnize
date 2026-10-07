import argparse
import json

from .files import convert_file


def main():
    parser = argparse.ArgumentParser(description="Convert stereo audio to speaker-labelled 5.1 WAV")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--mode", choices=("ambience", "matrix"), default="ambience")
    parser.add_argument("--profile", choices=("low_latency", "balanced", "quality"), default="balanced")
    parser.add_argument("--subtype", choices=("float32", "pcm16", "pcm24"), default="float32")
    parser.add_argument("--layout", choices=("side", "back"), default="side")
    parser.add_argument("--gain", type=float, default=.5)
    parser.add_argument("--tail-samples", type=int, default=0)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--onset-adaptation", action="store_true")
    parser.add_argument("--listening-mode", choices=("standard", "music"), default="standard")
    parser.add_argument("--center-width", type=float, default=0.0, help="0: normal center, 1: phantom center")
    parser.add_argument("--rear-gain", type=float, default=0.7, help="Front/rear balance through surround gain")
    parser.add_argument("--rear-delay-ms", type=float, help="Override listening-mode surround delay")
    parser.add_argument("--rear-shelf-db", type=float, help="Override high shelf (-12 to 0 dB)")
    parser.add_argument("--rear-shelf-hz", type=float, default=3000.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = convert_file(args.input, args.output, subtype=args.subtype, layout=args.layout,
                              overwrite=args.overwrite, normalize=args.normalize,
                              tail_samples=args.tail_samples, mode=args.mode,
                              onset_adaptation=args.onset_adaptation,
                              profile=args.profile, output_gain=args.gain,
                              listening_mode=args.listening_mode, center_width=args.center_width,
                              rear_gain=args.rear_gain, rear_delay_ms=args.rear_delay_ms,
                              rear_shelf_db=args.rear_shelf_db, rear_shelf_hz=args.rear_shelf_hz)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
