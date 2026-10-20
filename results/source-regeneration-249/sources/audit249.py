"""Separate scoring of sealed source-only regeneration and oracle context controls."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .audit229 import checked, sha
from .audit242b import decision, geometry, head
from .core225b import write_json

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def table(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(run, previous, local, carried):
    assert not (run/"audit.json").exists()
    design, fs, seal, cd, status, final = (read(run/n) for n in ("design.json", "forecast-status.json", "forecast-seal.json", "control-design.json", "status.json", "seal.json"))
    assert design["experiment"] == 249 and design["retrospective"] and design["primary"] == "native_source_regeneration"
    assert fs["status"] == status["status"] == "complete" and seal["status"] == final["status"] == "sealed"
    assert not fs["native_True_arrays_opened"] and not seal["native_True_arrays_opened"] and status["oracle_controls"]
    assert len(fs["cases"]) == len(status["cases"]) == 32 and fs["cases"] == seal["cases"] and status["cases"] == final["cases"]
    assert seal["design_sha256"] == sha(run/"design.json") and seal["status_sha256"] == sha(run/"forecast-status.json")
    assert cd["forecast_seal_sha256"] == final["forecast_seal_sha256"] == sha(run/"forecast-seal.json")
    assert final["status_sha256"] == sha(run/"status.json") and final["control_design_sha256"] == sha(run/"control-design.json")
    assert datetime.fromisoformat(cd["started_utc"]) > datetime.fromisoformat(seal["created_utc"])
    for name, h in design["source_hashes"].items():
        assert sha(ROOT/name) == sha(run/"sources"/Path(name).name) == h
    for folder, refs in ((previous, design["input242"]), (local, cd["input247"]), (carried, cd["input248"])):
        for name, h in refs.items():
            assert sha(folder/name) == h
    fe = {e["key"]:e for e in read(previous/"forecast-status.json")["cases"]}
    te = {e["key"]:e for e in read(previous/"status.json")["cases"]}
    le = {e["key"]:e for e in read(local/"status.json")["cases"]}
    ce = {e["key"]:e for e in read(carried/"status.json")["cases"]}
    weights = dict(np.load(previous/"weights.npz"))
    checks, outcomes, recovery, effects = [], [], [], []
    for case, fore, ctrl in zip(read(run/"sources/cases242.json"), fs["cases"], status["cases"], strict=True):
        key = case["key"]
        assert key == fore["key"] == ctrl["key"]
        fp, cp = (run/n/(key+".npz") for n in ("forecasts", "controls"))
        fm, cm = read(fp.with_suffix(".json")), read(cp.with_suffix(".json"))
        assert sha(fp) == fm["arrays_sha256"] == fore["arrays_sha256"] and sha(fp.with_suffix(".json")) == fore["record_sha256"]
        assert sha(cp) == cm["arrays_sha256"] == ctrl["arrays_sha256"] and sha(cp.with_suffix(".json")) == ctrl["record_sha256"]
        assert not fm["native_True_arrays_opened"] and cm["forecast_arrays_sha256"] == sha(fp)
        assert fm["prior_forecast_sha256"] == fe[key]["arrays_sha256"] == sha(previous/"forecasts"/fp.name)
        assert sha(previous/"observations"/fp.name) == te[key]["arrays_sha256"]
        assert cm["input247_sha256"] == le[key]["arrays_sha256"] == sha(local/"cases"/fp.name)
        assert cm["input248_sha256"] == ce[key]["arrays_sha256"] == sha(carried/"cases"/fp.name)
        for c in (*fm["checks"], *cm["checks"]):
            assert np.isfinite(c["error"]) and c["error"] <= c["bound"]
        with np.load(fp) as f, np.load(cp) as c, np.load(carried/"cases"/fp.name) as old, np.load(previous/"observations"/fp.name) as native, np.load(local/"cases"/fp.name) as boundary:
            checked(checks, "known_edit_source_all9_native_replay", f["source_rows"], native["true_trajectory"][:9, 0], 0.)
            checked(checks, "source_boundary_replay", f["S_input"][0], boundary["T_input"][0], 0.)
            checked(checks, "receiving_boundary_no_True_input", f["S_input"][1:], boundary["F_input"][1:], 0.)
            states = dict(F=old["F_trajectory"].astype(float), P=old["P_trajectory"].astype(float), T=old["T_trajectory"].astype(float),
                S=f["S_trajectory"].astype(float), C=c["C_trajectory"].astype(float))
            scores = {a:old[a+"_scores"].astype(float) for a in ("F", "P", "T")}
            scores.update(S=f["S_scores"].astype(float), C=c["C_scores"].astype(float))
            for arm in ("S", "C"):
                assert states[arm].shape == states["F"].shape and np.isfinite(states[arm]).all()
                checked(checks, "independent_finite_head", head(weights, states[arm][-1, -1]), scores[arm], .005)
            checked(checks, "source_future_known_True", states["S"][:, 0], states["T"][:, 0], 0.)
            checked(checks, "control_source_known_False", states["C"][:, 0], states["F"][:, 0], 0.)
            margins = {a:float(np.diff(s)[0]) for a, s in scores.items()}
            targets = dict(native=states["T"]-states["F"], context=states["P"]-states["F"])
            for kind, target in targets.items():
                for arm in ("S", "C"):
                    pred = states[arm]-states["F"]
                    margin_target = margins["T" if kind == "native" else "P"]
                    row = dict(key=key, **case["group"], arm=arm, passes=False, margin=margins[arm], target_margin=margin_target, margin_error=abs(margins[arm]-margin_target))
                    valid = []
                    for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                        truth, value = target[-1, pick], pred[-1, pick]
                        rel, cosine = geometry(value, truth)
                        row.update({scope+"_relative":rel, scope+"_cosine":cosine, scope+"_target_norm":float(np.linalg.norm(truth))})
                        valid.append(np.linalg.norm(truth) >= .001 and rel <= .05 and cosine >= .99)
                    row["passes"] = bool(all(valid) and row["margin_error"] <= .10)
                    (outcomes if kind == "native" else recovery).append(row)
            B = targets["context"]
            terms = dict(source=states["S"]-states["F"], receivers=states["C"]-states["F"], interaction=states["P"]-states["S"]-states["C"]+states["F"])
            checked(checks, "source_receiver_factorial_closure", sum(terms.values()), B, 1e-10)
            for name, term in terms.items():
                for i in range(19):
                    for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                        target, value = B[i, pick], term[i, pick]
                        rel, cosine = geometry(value, target)
                        den = max(np.linalg.norm(target), 1e-30)
                        effects.append(dict(key=key, **case["group"], term=name, layer=i+10, scope=scope, target_norm=float(np.linalg.norm(target)),
                            relative_error=rel, cosine=cosine, relative_norm=float(np.linalg.norm(value)/den), signed_share=float(np.vdot(value, target)/max(np.vdot(target, target), 1e-30))))
        print("source regeneration audit", key[:8], flush=True)
    assert fs["counts"] == dict(ordinary_false=32, source_dense_blocks=288, suffix_blocks=608, finite_heads=32)
    assert status["counts"] == dict(suffix_blocks=608, finite_heads=32) and status["ordinary_true"] == 0
    for name, rows in (("native-outcomes", outcomes), ("context-recovery", recovery), ("signed-effects", effects)):
        table(run/(name+".csv"), rows)
    native_decisions = {a:decision([r for r in outcomes if r["arm"] == a]) for a in ("S", "C")}
    write_json(run/"audit.json", dict(status="passed", retrospective=True, checks=checks, primary=native_decisions["S"], native_decisions=native_decisions,
        context_decisions={a:decision([r for r in recovery if r["arm"] == a]) for a in ("S", "C")},
        scorer_sha256=sha(__file__), design_sha256=sha(run/"design.json"), forecast_seal_sha256=sha(run/"forecast-seal.json"), seal_sha256=sha(run/"seal.json"),
        csv_hashes={n:sha(run/(n+".csv")) for n in ("native-outcomes", "context-recovery", "signed-effects")},
        interpretation="Retrospective False-only source forecast with substantial dense component budget; C uses oracle inputs. No speedup, learned circuit or fresh confirmation."))
    print(json.dumps(native_decisions, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--input-242", type=Path, required=True)
    parser.add_argument("--input-247", type=Path, required=True)
    parser.add_argument("--input-248", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.run.resolve(), args.input_242.resolve(), args.input_247.resolve(), args.input_248.resolve())
    except Exception as exc:
        write_json(args.run/"failure-audit.json", dict(error=repr(exc)))
        raise
