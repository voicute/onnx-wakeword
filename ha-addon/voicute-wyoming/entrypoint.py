#!/usr/bin/env python3
"""Home Assistant add-on entrypoint for the Voicute Wyoming service.

Home Assistant Supervisor writes the add-on options to ``/data/options.json``
and runs this script. We read those options (with defaults that mirror
``config.yaml``), then exec the Wyoming service. No bash/bashio/jq required —
the base image is python:3.10-slim, which ships none of those.
"""
import json
import os
import sys

OPTIONS_PATH = "/data/options.json"
SERVICE = "/app/wyoming/wyoming_voicute.py"
URI = "tcp://0.0.0.0:10400"

# Defaults must stay in sync with config.yaml's `options` block.
DEFAULTS = {
    "model_info": "/app/models/model_info.json",   # bundled demo; point to /data/... for custom
    "mel": "/app/models/melspectrogram.onnx",      # bundled, universal — no upload needed
    "threshold": 0.4,
    "cooldown": 1500,
    "L1": True,
    "L3": True,
    "L5": False,
    "debug": False,
}


def load_options():
    opts = dict(DEFAULTS)
    if os.path.exists(OPTIONS_PATH):
        try:
            with open(OPTIONS_PATH, encoding="utf-8") as f:
                opts.update(json.load(f))
        except Exception as exc:  # never crash the add-on on a malformed file
            print(f"WARN: could not read {OPTIONS_PATH}: {exc}", file=sys.stderr)
    return opts


def main():
    opts = load_options()
    args = [
        "--uri", URI,
        "--model-info", str(opts["model_info"]),
        "--mel", str(opts["mel"]),
        "--threshold", str(opts["threshold"]),
        "--cooldown", str(int(opts["cooldown"])),
        "--L1", "1" if opts["L1"] else "0",
        "--L3", "1" if opts["L3"] else "0",
        "--L5", "1" if opts["L5"] else "0",
    ]
    if opts.get("debug"):
        args.append("--debug")

    print(f"Starting Voicute Wyoming service: {SERVICE} " + " ".join(args))
    os.execvp("python", ["python", SERVICE] + args)


if __name__ == "__main__":
    main()
