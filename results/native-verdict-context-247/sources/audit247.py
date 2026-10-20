"""Separate saved-array / CPU64 primitive audit of native verdict coupling."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .audit229 import checked, sha, softmax
from .audit242b import decision, geometry, head
from .checkpoint238 import locate, weights
from .core225b import write_json

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("F", "P", "R", "T")
EDITS = ("earlier", "self", "coupling", "all", "weight", "value", "mixed", "random0", "random1", "random2")


def read(path):
    return json.loads(path.read_text())


def table(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(run, previous):
    assert not (run/"audit.json").exists()
    design, status, seal = (read(run/n) for n in ("design.json", "status.json", "seal.json"))
    assert design["experiment"] == 247 and design["retrospective"] and design["primary"] == "native_coupling_only"
    assert status["status"] == "complete" and seal["status"] == "sealed" and seal["cases"] == status["cases"] and len(status["cases"]) == 32
    assert seal["design_sha256"] == sha(run/"design.json") and seal["status_sha256"] == sha(run/"status.json")
    for name, h in design["source_hashes"].items():
        assert sha(ROOT/name) == sha(run/"sources"/Path(name).name) == h
    for name, h in design["inputs"].items():
        assert sha(previous/name) == h
    checkpoint, identity = locate(ROOT)
    assert identity == design["checkpoint"]
    with safe_open(checkpoint/"model.safetensors", framework="pt", device="cpu") as handle:
        w = weights(handle, read(checkpoint/"config.json"), 9, "attention")
    headw = dict(np.load(previous/"weights.npz"))
    first_entries = {e["key"]:e for e in read(previous/"forecast-status.json")["cases"]}
    true_entries = {e["key"]:e for e in read(previous/"status.json")["cases"]}
    checks, local, outcomes, primaries = [], [], [], []
    for case, ent in zip(read(run/"sources/cases242.json"), status["cases"], strict=True):
        key = case["key"]
        assert key == ent["key"]
        path = run/"cases"/(key+".npz")
        meta = read(path.with_suffix(".json"))
        assert meta["key"] == key and meta["group"] == case["group"]
        assert sha(path) == ent["arrays_sha256"] == meta["arrays_sha256"] and sha(path.with_suffix(".json")) == ent["record_sha256"]
        assert meta["prior_forecast_sha256"] == first_entries[key]["arrays_sha256"] == sha(previous/"forecasts"/path.name)
        assert meta["prior_native_sha256"] == true_entries[key]["arrays_sha256"] == sha(previous/"observations"/path.name)
        for c in meta["checks"]:
            assert np.isfinite(c["error"]) and c["error"] <= c["bound"]
        with np.load(path) as d, np.load(previous/"forecasts"/path.name) as old, np.load(previous/"observations"/path.name) as native:
            checked(checks, "False_input_replay", d["F_input"], old["false_trajectory"][8], 0.)
            checked(checks, "True_input_replay", d["T_input"], native["true_trajectory"][8], 0.)
            for arm, earlier, receiver in (("P", "T", "F"), ("R", "F", "T")):
                checked(checks, "crossed_tail_prior_rows", d[arm+"_input"][:-1], d[earlier+"_input"][:-1], 0.)
                checked(checks, "crossed_tail_receiver", d[arm+"_input"][-1], d[receiver+"_input"][-1], 0.)
            q, k, v, p, y = ({a:d[a+"_"+label].astype(float) for a in ARMS} for label in ("q", "k", "v", "p", "y"))
            for a in ARMS:
                groups = int(d[a+"_groups"])
                k[a], v[a] = (np.repeat(val, groups, axis=0) for val in (k[a], v[a]))
                raw = q[a]@k[a].swapaxes(-1, -2)*float(d[a+"_scaling"])
                checked(checks, "independent_QK", raw, d[a+"_raw"], 1e-4)
                checked(checks, "independent_softmax", softmax(d[a+"_raw"].astype(float)+d[a+"_mask"].astype(float)), p[a], 1e-4)
                checked(checks, "independent_weighted_value", p[a]@v[a], y[a], 1e-4)
            checked(checks, "True_fixed_query", q["T"], q["R"], 0.)
            checked(checks, "False_fixed_query", q["P"], q["F"], 0.)
            for name, bank in (("key", k), ("value", v)):
                checked(checks, "current_"+name+"_unchanged_T_R", bank["T"][:, -1], bank["R"][:, -1], 0.)
                checked(checks, "current_"+name+"_unchanged_P_F", bank["P"][:, -1], bank["F"][:, -1], 0.)
            terms = dict(earlier=y["T"]-y["R"], self=y["T"]-y["P"], coupling=y["T"]-y["P"]-y["R"]+y["F"], all=y["T"]-y["F"],
                weight=(p["T"]-p["R"])@v["R"], value=p["R"]@(v["T"]-v["R"]), mixed=(p["T"]-p["R"])@(v["T"]-v["R"]))
            for name, value in terms.items():
                checked(checks, "saved_component_"+name, d[name], value, 1e-12)
            remainder = terms["earlier"]-terms["weight"]-terms["value"]-terms["mixed"]
            checked(checks, "saved_multiplication_remainder", d["rounding"], remainder, 1e-12)
            checked(checks, "multiplication_remainder_bound", remainder, np.zeros_like(remainder), 1e-4)
            def project(value):
                return value.transpose(1, 0, 2).reshape(1, -1)@w["o_weight"].T
            for a in ARMS:
                checked(checks, "public_output_projection", project(y[a]), d[a+"_write"], 1e-4)
            target = project(terms["all"])
            local_pass = False
            for name, value in terms.items():
                projected = project(value)
                rel, cosine = geometry(projected, target)
                local.append(dict(key=key, **case["group"], term=name, target_norm=float(np.linalg.norm(target)),
                    relative_error=rel, cosine=cosine, relative_norm=float(np.linalg.norm(projected)/max(np.linalg.norm(target), 1e-30)),
                    signed_share=float(np.vdot(projected, target)/max(np.vdot(target, target), 1e-30))))
                if name == "coupling":
                    local_pass = np.linalg.norm(target) >= .001 and rel <= .10 and cosine >= .99
            f, t = d["F_final"].astype(float), d["T_final"].astype(float)
            checked(checks, "False_final_replay", f, old["false_trajectory"][-1], 0.)
            checked(checks, "True_final_replay", t, native["true_trajectory"][-1], 0.)
            mf, mt = (float(np.diff(d[a+"_scores"].astype(float))[0]) for a in ("F", "T"))
            response, dm = t-f, mt-mf
            records = {}
            for edit in EDITS:
                term = terms.get(edit)
                if edit.startswith("random"):
                    rng = np.random.default_rng(int(hashlib.sha256((key+edit).encode()).hexdigest()[:16], 16))
                    value = rng.normal(size=terms["coupling"].shape)
                    term = value*np.linalg.norm(terms["coupling"])/np.linalg.norm(value)
                expected = (y["T"]-term).astype(np.float32)
                checked(checks, "direct_installation", expected, d[edit+"_installed"], 0.)
                checked(checks, "actual_output_projection", project(expected.astype(float)), d[edit+"_write"], 1e-4)
                final, scores = d[edit+"_final"].astype(float), d[edit+"_scores"].astype(float)
                checked(checks, "independent_finite_head", head(headw, final[-1]), scores, .005)
                checked(checks, "unchanged_prior_final_rows", final[:-1], t[:-1], 0.)
                margin = float(np.diff(scores)[0])
                row = dict(key=key, **case["group"], edit=edit, baseline_margin_response=dm, remaining_margin_response=margin-mf,
                    margin_response_reduction=abs(dm)-abs(margin-mf), native_margin_error=abs(margin-mt))
                for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                    base, remain, changed = response[pick], (final-f)[pick], (final-t)[pick]
                    den = max(np.linalg.norm(base), 1e-30)
                    row[scope+"_remaining_ratio"] = float(np.linalg.norm(remain)/den)
                    row[scope+"_native_relative"] = float(np.linalg.norm(changed)/den)
                    row[scope+"_change_signed_share"] = float(np.vdot(changed, base)/max(np.vdot(base, base), 1e-30))
                outcomes.append(row)
                records[edit] = row
            coupling = records["coupling"]
            causal_pass = abs(dm) >= .10 and coupling["verdict_remaining_ratio"] <= .75 and coupling["margin_response_reduction"] >= .10
            causal_pass = causal_pass and all(coupling["verdict_remaining_ratio"] < records[c]["verdict_remaining_ratio"] for c in ("random0", "random1", "random2"))
            primaries.append(dict(key=key, **case["group"], local_pass=bool(local_pass), causal_pass=bool(causal_pass), passes=bool(local_pass and causal_pass)))
        print("native verdict audit", key[:8], flush=True)
    assert status["counts"]["ordinary_false"] == status["counts"]["ordinary_true"] == 32
    assert status["counts"]["local_crossed_attention_calls"] == 64 and status["counts"]["primitive_intervention_forwards"] == status["counts"]["primitive_installations"] == 320
    for name, rows in (("local-effects", local), ("native-outcomes", outcomes), ("primary", primaries)):
        table(run/(name+".csv"), rows)
    write_json(run/"audit.json", dict(status="passed", retrospective=True, checks=checks, primary=decision(primaries),
        local=decision([dict(r, passes=r["local_pass"]) for r in primaries]), causal=decision([dict(r, passes=r["causal_pass"]) for r in primaries]),
        scorer_sha256=sha(__file__), design_sha256=sha(run/"design.json"), seal_sha256=sha(run/"seal.json"),
        csv_hashes={n:sha(run/(n+".csv")) for n in ("local-effects", "native-outcomes", "primary")},
        interpretation="Native-input oracle context/receiver factorial; not a False-only forecast or learned circuit."))
    print(json.dumps(decision(primaries), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--input-242", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.run.resolve(), args.input_242.resolve())
    except Exception as exc:
        write_json(args.run/"failure-audit.json", dict(error=repr(exc)))
        raise
