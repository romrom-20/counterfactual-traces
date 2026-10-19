"""Complete retrospective residual-write error budget for the failed 228 F forecast."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .audit229 import checked, sha

ROOT = Path(__file__).resolve().parents[2]
NODES = tuple((i, kind) for i in range(24) for kind in ("attention", "mlp") if not (i == 23 and kind == "mlp"))
SOURCES = [ROOT/p for p in ("research/boolean_coupling/ledger235.py", "research/boolean_coupling/finite235.py",
    "research/boolean_coupling/audit235.py", "research/boolean_coupling/core225b.py",
    "research/boolean_coupling/alignment228.py", "research/boolean_coupling/tail226b.py",
    "research/boolean_coupling/audit229.py", "research/boolean_coupling/smooth228.py", "research/boolean_coupling/settings.json",
    "research/boolean_coupling/cases228.json", "docs/experiments/235-global-forecast-error-budget.md")]


def states(f, o, i, kind):
    if kind == "attention":
        native_in = f["false_boundaries"][0].astype(float) if i == 0 else o["true_states"][i-1].astype(float)
        pred_in = f["false_boundaries"][i].astype(float)+f["F_delta_in"][i]
        return native_in, o["true_mids"][i].astype(float), pred_in, f["F_pred_mid"][i]
    native_in, native_out = o["true_mids"][i].astype(float), o["true_states"][i].astype(float)
    pred_in = f["F_pred_mid"][i]
    pred_out = f["false_boundaries"][i+1].astype(float)+f["F_delta_out"][i]
    return native_in, native_out, pred_in, pred_out


def write_errors(f, o):
    values = []
    for i, kind in NODES:
        ni, no, pi, po = states(f, o, i, kind)
        values.append((no-ni)-(po-pi))
    values.append(f["F_pred_mid"][-1]-f["F_target_mid"].astype(float))
    return np.stack(values)


def main(previous, out):
    assert not out.exists(), "Never overwrite evidence."
    out.mkdir(parents=True)
    for name in ("sources", "ledger", "cases"):
        (out/name).mkdir()
    sources = {str(p.relative_to(ROOT)):sha(p) for p in SOURCES}
    for p in SOURCES:
        (out/"sources"/p.name).write_bytes(p.read_bytes())
    cases = json.loads((ROOT/"research/boolean_coupling/cases228.json").read_text())
    paths = [previous/n for n in ("design.json", "runtime.json", "audit.json", "forecast-manifest.json", "forecast-qualification-seal.json", "weights.npz")]
    paths += [previous/phase/(c["key"]+suffix) for c in cases for phase in ("forecasts", "observations") for suffix in (".json", ".npz")]
    inputs = {str(p.relative_to(previous)):sha(p) for p in paths}
    (out/"design.json").write_text(json.dumps(dict(experiment=235, retrospective=True, input_root=str(previous),
        source_hashes=sources, inputs=inputs, node_order=[dict(layer=i+1, kind=k) for i, k in NODES],
        bounds=dict(identity=1e-9, primal=1e-4, head=.005)), indent=2)+"\n")
    rows, entries, checks = [], [], []
    for c in cases:
        key = c["key"]
        paths = [previous/phase/(key+".npz") for phase in ("forecasts", "observations")]
        for p in paths:
            assert sha(p) == json.loads(p.with_suffix(".json").read_text())["arrays_sha256"]
        f, o = (dict(np.load(p)) for p in paths)
        terms = write_errors(f, o)
        target = o["true_mids"][-1].astype(float)-f["F_target_mid"].astype(float)
        checked(checks, "complete_ledger_closure", terms.sum(0), target, 1e-9)
        checked(checks, "initial_receiving_error", f["F_delta_in"][0], np.zeros_like(target), 0.)
        for index, term in enumerate(terms):
            for scope in ("all", "verdict"):
                value, error = (term, target) if scope == "all" else (term[-1], target[-1])
                rows.append(dict(key=key, **c["group"], node=index, layer=NODES[index][0]+1 if index < 47 else 24,
                    kind=NODES[index][1] if index < 47 else "cast", scope=scope,
                    norm=float(np.linalg.norm(value)), terminal_error_norm=float(np.linalg.norm(error)),
                    signed_share=float(np.vdot(value, error)/np.vdot(error, error))))
        path = out/"ledger"/(key+".npz")
        np.savez_compressed(path, terms=terms, target_error=target)
        meta = dict(key=key, group=c["group"], arrays_sha256=sha(path))
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
        entries.append(dict(key=key, arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json"))))
    ranking = sorted(range(47), key=lambda n:(-float(np.median([r["signed_share"] for r in rows if r["node"] == n and r["scope"] == "all"])), n))
    selection = dict(retrospective=True, rule="Median signed whole-parent projection; chronological tie-break; before finite outcomes.",
        ranking=[dict(node=n, layer=NODES[n][0]+1, kind=NODES[n][1],
            median_share=float(np.median([r["signed_share"] for r in rows if r["node"] == n and r["scope"] == "all"]))) for n in ranking],
        selected=ranking[:4])
    with (out/"ledger.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out/"selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    (out/"ledger-status.json").write_text(json.dumps(dict(status="complete", cases=entries, checks=checks,
        ordinary_models=0, selection_sha256=sha(out/"selection.json")), indent=2)+"\n")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-228", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    target = args.out.resolve()
    try:
        main(args.input_228.resolve(), target)
    except Exception as exc:
        if target.exists():
            (target/"failure-ledger.json").write_text(json.dumps(dict(error=repr(exc)), indent=2)+"\n")
        raise
