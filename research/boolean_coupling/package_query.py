"""Publish complete253/254 metadata while retaining arrays and drafts locally."""

import argparse
import json
import shutil
from pathlib import Path

from .audit229 import sha
from .core225b import write_json
from .package_attention import copy_evidence

ROOT = Path(__file__).resolve().parents[2]


def main(run, out, logs):
    assert not (out/"manifest.json").exists() and json.loads((run/"audit.json").read_text())["status"] == "passed"
    out.mkdir(parents=True, exist_ok=True)
    copy_evidence(run, out)
    for p in logs:
        dest = out/"logs"/p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    design = json.loads((run/"design.json").read_text())
    num = design["experiment"]
    report = "253-query-generation-result.md" if num == 253 else "254-predicted-query-curvature-result.md"
    extras = [ROOT/p for p in (
        "research/boolean_coupling/reproduce_query.py", "research/boolean_coupling/describe_query.py",
        "research/boolean_coupling/package_query.py", "research/boolean_coupling/package_attention.py",
        "docs/analysis/"+report, f"docs/figures/{num}/{num}-query-generation.png", f"docs/figures/{num}/{num}-query-generation.svg")]
    extras.extend(ROOT/p for p in design["source_hashes"])
    if num == 254:
        extras.extend(ROOT/p for p in (
            "research/boolean_coupling/plot_query_release.py",
            "docs/figures/254/254-query-curvature.png", "docs/figures/254/254-query-curvature.svg",
            "docs/figures/254/presentation-provenance.json"))
    published = [p for p in out.rglob("*") if p.is_file()]
    write_json(out/"manifest.json", dict(release=out.name, fresh_literals=True, source_freeze_commit=design["source_freeze_commit"],
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(published+extras))}))
    print("Packaged complete query-generation evidence; activation arrays, weights and drafts stay local.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, action="append", default=[])
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve(), [p.resolve() for p in args.log])
