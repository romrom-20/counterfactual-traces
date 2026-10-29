#!/usr/bin/env python3
"""Launch a model job with one shared Hugging Face cache and a disk reserve."""

import argparse
import math
import os
import shutil
from pathlib import Path


def existing_parent(path):
    while not path.exists():
        path = path.parent
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".cache/huggingface",
        help="Shared HF_HOME root; defaults to the user's existing Hugging Face cache.",
    )
    parser.add_argument(
        "--minimum-free-gib",
        type=float,
        default=8,
        help="Required free space on both the working and cache volumes (default:8GiB).",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Supply a model command after --.")
    if not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib < 0:
        parser.error("The disk reserve must be a finite, nonnegative number.")
    cache = args.cache_root.expanduser().resolve()
    if "runs" in cache.parts:
        parser.error("The shared model cache must be outside experiment runs directories.")
    minimum = int(args.minimum_free_gib * 1024**3)
    for path in {Path.cwd(), existing_parent(cache)}:
        available = shutil.disk_usage(path).free
        if available < minimum:
            parser.error(
                f"Insufficient free disk at {path}: {available:,} bytes available; "
                f"{minimum:,} bytes reserved. No command was started."
            )
    cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        HF_HOME=str(cache),
        HF_HUB_CACHE=str(cache / "hub"),
        HUGGINGFACE_HUB_CACHE=str(cache / "hub"),
    )
    # Old Transformers overrides can otherwise silently create a second per-run cache.
    env.pop("TRANSFORMERS_CACHE", None)
    print(f"Shared model cache: {cache}", flush=True)
    try:
        os.execvpe(command[0], command, env)
    except OSError as error:
        parser.exit(127, f"Cannot start model command: {error.strerror}\n")


if __name__ == "__main__":
    main()
