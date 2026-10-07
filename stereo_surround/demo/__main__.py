"""python -m stereo_surround.demo [file.wav] [--device DEVICE]."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Realtime 5.1 playback with PySide6/PyQtGraph")
    parser.add_argument("file", nargs="?")
    parser.add_argument("--device", help="Device ID or a substring of its name")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    try:
        if args.list_devices:
            from .audio import output_devices
            print(json.dumps(output_devices(), ensure_ascii=True, indent=2))
            return
        from PySide6 import QtWidgets
        from .window import DemoWindow
        app = QtWidgets.QApplication(sys.argv[:1])
        window = DemoWindow(args.file, args.device)
        window.show()
        sys.exit(app.exec())
    except ImportError as exc:
        parser.exit(1, f"Demo dependencies are missing: {exc}\nInstall: python -m pip install -e \".[demo]\"\n")
    except ValueError as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
