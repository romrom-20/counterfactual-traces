"""Separate audit of finite native carry/read trajectories and state sufficiency."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .audit229 import checked, sha
from .audit242b import decision, geometry, head
from .core225b import write_json

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("F", "P", "R", "T")


def read(path):
    return json.loads(path.read_text())


def table(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(run, local, previous):
    assert not (run/"audit.json").exists()
    design, status, seal = (read(run/n) for n in ("design.json", "status.json", "seal.json"))
    assert design["experiment"] == 248 and design["retrospective"] and design["primary"] == "verdict_state_carry"
    assert status["status"] == "complete" and status["ordinary_true"] == 0 and status["reused_True_inputs"]
    assert seal["status"] == "sealed" and len(status["cases"]) == 32 and status["cases"] == seal["cases"]
    assert seal["design_sha256"] == sha(run/"design.json") and seal["status_sha256"] == sha(run/"status.json")
    for name, h in design["source_hashes"].items():
        assert sha(ROOT/name) == sha(run/"sources"/Path(name).name) == h
    for folder, label in ((local, "input247"), (previous, "input242")):
        for name, h in design[label].items():
            assert sha(folder/name) == h
    le = {e["key"]:e for e in read(local/"status.json")["cases"]}
    fe = {e["key"]:e for e in read(previous/"forecast-status.json")["cases"]}
    te = {e["key"]:e for e in read(previous/"status.json")["cases"]}
    weights = dict(np.load(previous/"weights.npz"))
    checks, outcomes, effects, negligible = [], [], [], []
    for case, ent in zip(read(run/"sources/cases242.json"), status["cases"], strict=True):
        key = case["key"]
        assert key == ent["key"]
        path = run/"cases"/(key+".npz")
        meta = read(path.with_suffix(".json"))
        assert meta["key"] == key and meta["group"] == case["group"]
        assert sha(path) == meta["arrays_sha256"] == ent["arrays_sha256"] and sha(path.with_suffix(".json")) == ent["record_sha256"]
        assert sha(local/"cases"/path.name) == meta["input247_sha256"] == le[key]["arrays_sha256"]
        assert sha(previous/"forecasts"/path.name) == meta["prior_forecast_sha256"] == fe[key]["arrays_sha256"]
        assert sha(previous/"observations"/path.name) == meta["prior_native_sha256"] == te[key]["arrays_sha256"]
        for c in meta["checks"]:
            assert np.isfinite(c["error"]) and c["error"] <= c["bound"]
        with np.load(path) as d, np.load(local/"cases"/path.name) as parent, np.load(previous/"forecasts"/path.name) as f, np.load(previous/"observations"/path.name) as t:
            states, margins = {}, {}
            for a in ARMS:
                checked(checks, "independent_boundary_installation", d[a+"_input"], parent[a+"_input"], 0.)
                states[a] = d[a+"_trajectory"].astype(float)
                assert states[a].shape == (19, *d["F_input"].shape) and np.isfinite(states[a]).all()
                score = d[a+"_scores"].astype(float)
                checked(checks, "independent_finite_head", head(weights, states[a][-1, -1]), score, .005)
                margins[a] = float(np.diff(score)[0])
                if a in ("F", "T"):
                    checked(checks, "native_suffix_state_replay", states[a], f["false_trajectory"][9:] if a == "F" else t["true_trajectory"][9:], 0.)
                    checked(checks, "native_suffix_score_replay", score, f["false_scores"] if a == "F" else t["true_scores"], 0.)
            for a, b in (("F", "R"), ("P", "T")):
                checked(checks, "native_causal_prior_rows", states[a][:, :-1], states[b][:, :-1], 0.)
            B = states["T"]-states["F"]
            terms = dict(context=states["P"]-states["F"], carry=states["R"]-states["F"], interaction=states["T"]-states["P"]-states["R"]+states["F"])
            checked(checks, "native_suffix_factorial_closure", sum(terms.values()), B, 1e-10)
            dm = margins["T"]-margins["F"]
            im = margins["T"]-margins["P"]-margins["R"]+margins["F"]
            for a in ("P", "R"):
                pred = states[a]-states["F"]
                row = dict(key=key, **case["group"], arm=a, native_margin_response=dm, margin=margins[a], native_margin=margins["T"], margin_error=abs(margins[a]-margins["T"]))
                valid = []
                for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                    target, value = B[-1, pick], pred[-1, pick]
                    rel, cosine = geometry(value, target)
                    success = np.linalg.norm(target) >= .001 and rel <= .05 and cosine >= .99
                    row.update({scope+"_relative":rel, scope+"_cosine":cosine, scope+"_target_norm":float(np.linalg.norm(target)), scope+"_passes":bool(success)})
                    valid.append(success)
                row["passes"] = bool(row["verdict_passes"] and row["margin_error"] <= .10)
                row["full_passes"] = bool(all(valid) and row["margin_error"] <= .10)
                outcomes.append(row)
            nrow = dict(key=key, **case["group"], passes=True, margin_interaction=im)
            for name, term in terms.items():
                for i in range(19):
                    for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                        target, value = B[i, pick], term[i, pick]
                        den = max(np.linalg.norm(target), 1e-30)
                        rel, cosine = geometry(value, target)
                        magnitude = float(np.linalg.norm(value)/den)
                        effects.append(dict(key=key, **case["group"], term=name, layer=i+10, scope=scope, target_norm=float(np.linalg.norm(target)),
                            relative_error=rel, cosine=cosine, relative_norm=magnitude, signed_share=float(np.vdot(value, target)/max(np.vdot(target, target), 1e-30))))
                        if name == "interaction" and i == 18:
                            nrow["passes"] = bool(nrow["passes"] and np.linalg.norm(target) >= .001 and magnitude <= .10)
            nrow["passes"] = bool(nrow["passes"] and abs(im) <= .10)
            negligible.append(nrow)
        print("native suffix audit", key[:8], flush=True)
    counts = status["counts"]
    assert counts["ordinary_false"] == 32 and counts["suffix_blocks"] == 2432 and counts["finite_heads"] == 128
    for name, rows in (("native-outcomes", outcomes), ("signed-effects", effects), ("interaction", negligible)):
        table(run/(name+".csv"), rows)
    decisions = {a:decision([r for r in outcomes if r["arm"] == a]) for a in ("P", "R")}
    write_json(run/"audit.json", dict(status="passed", retrospective=True, checks=checks, primary=decisions["R"], verdict_decisions=decisions,
        full_decisions={a:decision([dict(r, passes=r["full_passes"]) for r in outcomes if r["arm"] == a]) for a in ("P", "R")},
        negligible_interaction=decision(negligible), scorer_sha256=sha(__file__), design_sha256=sha(run/"design.json"), seal_sha256=sha(run/"seal.json"),
        csv_hashes={n:sha(run/(n+".csv")) for n in ("native-outcomes", "signed-effects", "interaction")},
        interpretation="Native observed-state carry/read sufficiency, not a False-only forecast or learned circuit."))
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--input-247", type=Path, required=True)
    parser.add_argument("--input-242", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.run.resolve(), args.input_247.resolve(), args.input_242.resolve())
    except Exception as exc:
        write_json(args.run/"failure-audit.json", dict(error=repr(exc)))
        raise
