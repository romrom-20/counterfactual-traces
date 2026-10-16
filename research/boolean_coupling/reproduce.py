"""Verify published evidence without a model, or execute the standalone225b experiment."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(history=False):
    manifest = json.loads((ROOT/"results/baseline-interaction-225b/manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert sha(ROOT/name) == expected, name
    print(f"Verified {len(manifest['files'])} published files; zero model calls.", flush=True)
    if history:
        hashes, old = {}, None
        with tarfile.open(ROOT/"evidence/through-224b.tar.gz", mode="r|gz") as archive:
            for member in archive:
                assert member.isfile()
                name = member.name.removeprefix("through-224b/")
                data = archive.extractfile(member).read()
                hashes[name] = hashlib.sha256(data).hexdigest()
                if name == "results/mlp-primitive-factorial-224b/manifest.json":
                    old = json.loads(data)
        assert old and len(old["files"]) == 2868
        assert all(hashes[name] == expected for name, expected in old["files"].items())
        print("Verified all2,868 historical preservation files inside the archive; no activation arrays or model replication.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT/"runs/225b-reproduction")
    args = parser.parse_args()
    commands = [
        ["boolean_coupling.run225b", "--phase", "qualify"],
        ["boolean_coupling.audit225b", "--phase", "qualify"],
        ["boolean_coupling.run225b", "--phase", "intervene"],
        ["boolean_coupling.audit225b", "--phase", "interventions"],
        ["boolean_coupling.present"],
    ]
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(commands=commands, ordinary_models=64, suffix_continuations=704,
                             remaining_blocks=3872, finite_heads=704,
                             qualification_primitive_comparisons=512,
                             note="Retrospective native-input study. Verification is not model replication."), indent=2))
    if not args.run:
        if not (args.verify_only or args.history or args.plan):
            parser.print_help()
        return
    assert not args.out.exists(), "Use a fresh output directory; preserve previous results."
    environment = dict(os.environ, PYTHONPATH=str(ROOT/"research"),
                       OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    if args.offline:
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for module, *arguments in commands:
        command = [sys.executable, "-m", module, *arguments, "--out", str(args.out.resolve())]
        if args.offline and module.endswith("run225b"):
            command.append("--offline")
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
