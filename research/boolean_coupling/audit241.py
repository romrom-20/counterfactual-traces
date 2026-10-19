"""Independent analytic/finite-difference vector transport and fixed241 scoring."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .audit229 import checked, sha
from .checkpoint238 import locate, weights
from .maps241 import maps
from .vector241 import NODES, inputs, write


def geometry(pred, target):
    den, mag = np.linalg.norm(target), np.linalg.norm(pred)
    return float(np.linalg.norm(pred - target) / den) if den else 0., float(np.vdot(pred, target) / (mag * den)) if mag * den else 0.


def main(run):
    assert not (run / "audit.json").exists()
    root = Path(__file__).resolve().parents[2]
    design, previous, old, last = inputs(run)
    for p, h in design["source_hashes"].items():
        assert sha(root / p) == sha(run / "sources" / Path(p).name) == h
    assert sha(previous / "audit.json") == design["input_audits"]["238"]
    assert sha(last / "audit.json") == design["input_audits"]["240"]
    status = json.loads((run / "status.json").read_text())
    assert status["status"] == "complete" and len(status["cases"]) == 32
    folder, checkpoint = locate(root)
    assert checkpoint == design["checkpoint"]
    config = json.loads((folder / "config.json").read_text())
    checks, transported_rows, cases, count = [], [], [], Counter()
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:
        for ent in status["cases"]:
            key = ent["key"]
            path = run / "cases" / (key + ".npz")
            meta = json.loads(path.with_suffix(".json").read_text())
            assert sha(path) == ent["arrays_sha256"] == meta["arrays_sha256"]
            assert sha(path.with_suffix(".json")) == ent["record_sha256"]
            assert all(c["error"] <= c["bound"] for c in meta["checks"])
            paths = dict(ledger=previous / "ledger" / path.name, forecasts=old / "forecasts" / path.name,
                observations=old / "observations" / path.name, mask=last / "masks" / path.name)
            for label, p in paths.items():
                assert sha(p) == meta["inputs"][label]
            ledger, f, o, installed = (dict(np.load(paths[k])) for k in ("ledger", "forecasts", "observations", "mask"))
            d = dict(np.load(path))
            context = {k: ledger[k].astype(float) for k in ("mask", "rope_cos", "rope_sin")}
            start = meta["start"]
            terms = np.empty((0, *d["target"].shape))
            for n in NODES:
                i, kind = n // 2, "attention" if n % 2 == 0 else "mlp"
                w = weights(handle, config, i, kind)
                x = ledger[f"a{i}_false"].astype(float) if kind == "attention" else f["false_mids"][i].astype(float)
                _, derivative, finite = maps(w, context, x, start, kind)
                anchor = x[start:] if kind == "attention" else x
                directions = np.concatenate((terms, ledger[f"n{n}_direction"][None]))
                tangent = derivative(directions)
                plus, minus = (finite(anchor + sign * 1e-5 * directions) for sign in (1, -1))
                checked(checks, "independent_analytic_all_JVP", tangent, d[f"n{n}_tangent"], 1e-9)
                checked(checks, "independent_finite_endpoints_all_JVP", (plus - minus) / 2e-5, d[f"n{n}_tangent"])
                checked(checks, "old_native_error_analytic_derivative", tangent[-1], ledger[f"n{n}_tangent"], 1e-9)
                terms = np.concatenate((tangent[:-1], ledger[f"n{n}_remainder"][None]))
                checked(checks, "independent_intermediate_vector_closure", terms.sum(0), ledger[f"n{n}_tangent"] + ledger[f"n{n}_remainder"], 1e-8)
                count["directions"] += len(directions)
                count["finite_endpoints"] += 2 * len(directions)
            checked(checks, "independent_all_terminal_components", terms, d["terms"], 1e-9)
            cast = f["F_pred_mid"][15] - installed["own_input"].astype(float)
            target = o["true_mids"][15].astype(float) - installed["own_input"].astype(float)
            checked(checks, "known_cast", cast, d["cast"], 0.)
            checked(checks, "independent_target", target, d["target"], 0.)
            checked(checks, "independent_terminal_vector_closure", terms.sum(0) + cast, target, 1e-8)
            checked(checks, "independent_Gram", terms.reshape(len(NODES), -1) @ terms.reshape(len(NODES), -1).T, d["gram"], 1e-8)
            den = np.vdot(target, target)
            for index, n in enumerate(NODES):
                v, r = terms[index], ledger[f"n{n}_remainder"]
                transported_rows.append(dict(key=key, **meta["group"], node=n, kind="attention" if n % 2 == 0 else "mlp", layer=n // 2 + 1,
                    local_norm=float(np.linalg.norm(r)), terminal_norm=float(np.linalg.norm(v)),
                    norm_gain=float(np.linalg.norm(v) / np.linalg.norm(r)) if np.linalg.norm(r) else 0.,
                    signed_share=float(np.vdot(v, target) / den),
                    alone_remaining_relative=geometry(v + cast, target)[0]))
            cases.append(dict(key=key, group=meta["group"], terms=terms, target=target, cast=cast))
            print("independent vector transport", key[:8], flush=True)
    assert dict(count) == dict(directions=13920, finite_endpoints=27840)
    order, curve, residuals = [], [], [c["target"] - c["cast"] for c in cases]
    for step in range(len(NODES)):
        candidates = []
        for index, n in enumerate(NODES):
            if n in order:
                continue
            value = float(np.median([np.linalg.norm(r - c["terms"][index]) / np.linalg.norm(c["target"]) for r, c in zip(residuals, cases, strict=True)]))
            candidates.append((value, n, index))
        _, n, index = min(candidates)
        order.append(n)
        for c, residual in zip(cases, residuals, strict=True):
            residual -= c["terms"][index]
            reconstruction = c["target"] - residual
            metrics, passes = {}, []
            for scope, pick in (("all", slice(None)), ("verdict", -1)):
                r, cosine = geometry(reconstruction[pick], c["target"][pick])
                metrics.update({scope + "_relative": r, scope + "_cosine": cosine})
                passes.append(np.linalg.norm(c["target"][pick]) >= .001 and r <= .25 and cosine >= .95)
            curve.append(dict(key=c["key"], **c["group"], prefix=step + 1, added_node=n, **metrics, passes=bool(all(passes))))
    decisions = {}
    for k in range(1, len(NODES) + 1):
        subset = [r for r in curve if r["prefix"] == k]
        strata = {f"{s}_{rev}": sum(r["passes"] for r in subset if r["schema"] == s and r["reverse"] == rev)
            for s in ("generic", "descriptive") for rev in (0, 1)}
        decisions[str(k)] = dict(passes=sum(r["passes"] for r in subset), strata=strata)
        decisions[str(k)]["meets_rule"] = decisions[str(k)]["passes"] >= 24 and all(n >= 6 for n in strata.values())
    for name, rows in (("transported", transported_rows), ("reconstruction", curve)):
        with (run / (name + ".csv")).open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write(run / "selection.json", dict(retrospective=True, rule="Greedy shared median whole-vector residual; chronological tie-break.",
        order=order, primary_sites=order[:4], decisions=decisions,
        first_sufficient_prefix=next((k for k in range(1, len(NODES) + 1) if decisions[str(k)]["meets_rule"]), None)))
    write(run / "audit.json", dict(status="passed", retrospective=True, ordinary_model_calls=0, source_hashes=design["source_hashes"],
        checks=checks, independent_counts=dict(count), primary=decisions["4"], selection_sha256=sha(run / "selection.json")))
    print(json.dumps(dict(primary=decisions["4"], order=order, first_sufficient_prefix=next((k for k in range(1, len(NODES) + 1)
        if decisions[str(k)]["meets_rule"]), None)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.run.resolve())
    except Exception as exc:
        write(args.run / "failure-audit.json", dict(error=repr(exc)))
        raise
