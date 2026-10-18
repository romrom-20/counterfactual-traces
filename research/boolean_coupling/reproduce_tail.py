"""Verify the 226b/227b release or reproduce its finite-intervention pipeline.

The verification and plan paths use only the standard library. The sealed model
runners are called unchanged in separate processes; this wrapper is orchestration.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "results/tail-interactions-226-227"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(history=False):
    manifest = json.loads((RELEASE / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert sha(ROOT / name) == expected, name
    original = json.loads((RELEASE / "original-local-manifest.json").read_text())
    revision = json.loads((RELEASE / "publication-revision.json").read_text())
    for name, expected in original["files"].items():
        if name in revision["presentation_changes"]:
            item = revision["presentation_changes"][name]
            assert item["original_sha256"] == expected, name
            expected = item["published_sha256"]
        assert sha(ROOT / name) == expected, name
    for folder in ("finite-tail-interaction-226b", "mlp24-mixed-primitives-227b"):
        design = json.loads((ROOT / "results" / folder / "design.json").read_text())
        for name, expected in design["sources"].items():
            assert sha(ROOT / name) == expected, name
    command = [sys.executable, str(ROOT / "research/boolean_coupling/reproduce.py"),
               "--verify-only"]
    if history:
        command.append("--history")
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Verified {len(manifest['files'])} continuation release files and original "
          "scientific source seals; zero model calls. Raw arrays are excluded.", flush=True)


def plan(out, base=None, offline=False, checkpoint=None):
    reuse_parent = base is not None
    base = base or out / "225b"
    tail = out / "226b"
    primitive = out / "227b"
    steps = []
    if not reuse_parent:
        command = [sys.executable, str(ROOT / "research/boolean_coupling/reproduce.py"),
                   "--run", "--out", str(base)]
        if offline:
            command.append("--offline")
        steps.append(dict(name="225b-parent", command=command))
    for name, module, arguments in (
        ("226b-model", "tail226b", ["--input", str(base), "--out", str(tail)]),
        ("226b-audit", "audit226b", ["--out", str(tail)]),
        ("227b-model", "tail227b", ["--input", str(base), "--tail", str(tail),
                                  "--out", str(primitive)]),
        ("227b-audit", "audit227b", ["--out", str(primitive)]),
        ("227b-finite-validation", "validate227b", ["--out", str(primitive),
            "--checkpoint", str(checkpoint) if checkpoint else "{cached_checkpoint}"]),
        ("227b-posthoc-algebra", "describe227b", ["--out", str(primitive)]),
    ):
        steps.append(dict(name=name, command=[sys.executable, "-m",
                     "boolean_coupling." + module, *arguments]))
    return steps


def cached_checkpoint():
    # Imported only during an actual run, after the parent has cached its weights.
    from huggingface_hub import hf_hub_download

    specification = json.loads((ROOT / "research/boolean_coupling/settings.json")
                               .read_text())["model"]
    return hf_hub_download(specification["model"], filename="model.safetensors",
                          revision=specification["revision"], local_files_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--offline", action="store_true",
                        help="Require the parent checkpoint to be cached too.")
    parser.add_argument("--base", type=Path, help="Reuse a completed 225b run.")
    parser.add_argument("--checkpoint", type=Path,
                        help="Pinned snapshot model.safetensors; otherwise find it in the cache.")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/tail-reproduction")
    args = parser.parse_args()
    out = args.out.resolve()
    base = args.base.resolve() if args.base else None
    checkpoint = args.checkpoint.absolute() if args.checkpoint else None
    steps = plan(out, base, args.offline, checkpoint)
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(steps=steps, retrospective=True,
            note="No model calls for --plan. Existing parent arrays are required only with --base. "
                 "Successful verification is not model replication."), indent=2))
    if not args.run:
        if not (args.verify_only or args.history or args.plan):
            parser.print_help()
        return
    if out.exists():
        parser.error("--out must be a new directory; existing evidence is never overwritten.")
    if base:
        for name in ("qualification-seal.json", "audit.json", "status.json"):
            if not (base / name).is_file():
                parser.error(f"--base is missing {name}; use a completed 225b output directory.")
    verify()
    out.mkdir(parents=True)
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "research"),
                       OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    if args.offline:
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    record = dict(wrapper_sha256=sha(__file__), steps=steps, completed=[], status="running")
    status = out / "pipeline.json"
    for step in steps:
        command = step["command"]
        try:
            if "{cached_checkpoint}" in command:
                command[command.index("{cached_checkpoint}")] = cached_checkpoint()
            status.write_text(json.dumps(record, indent=2) + "\n")
            log = out / (step["name"] + ".log")
            print(f"Running {step['name']}; log: {log}", flush=True)
            with log.open("w") as handle:
                subprocess.run(command, cwd=ROOT, env=environment, check=True,
                               stdout=handle, stderr=subprocess.STDOUT)
        except Exception as exc:
            record.update(status="failed", failed_step=step["name"], error=str(exc))
            status.write_text(json.dumps(record, indent=2) + "\n")
            raise
        record["completed"].append(step["name"])
    record["status"] = "complete"
    status.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Finished. New raw arrays and audits are under {out}.", flush=True)


if __name__ == "__main__":
    main()
