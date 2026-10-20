"""Separate finite-kernel witness and outcome audit for native communication250."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .audit229 import checked, sha, softmax
from .audit242b import decision, geometry, head
from .core225b import write_json

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("F", "T", "Z", "K", "V", "B", "D", "L", "random0", "random1", "random2", "H")


def read(path):
    return json.loads(path.read_text())


def table(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(run, previous):
    assert not (run/"audit.json").exists()
    design, status, seal = (read(run/n) for n in ("design.json", "status.json", "seal.json"))
    assert design["experiment"] == 250 and design["retrospective"] and design["primary"] == "continued_source_communication"
    assert status["status"] == "complete" and seal["status"] == "sealed" and seal["cases"] == status["cases"] and len(status["cases"]) == 32
    assert seal["design_sha256"] == sha(run/"design.json") and seal["status_sha256"] == sha(run/"status.json")
    for path, h in design["source_hashes"].items():
        assert sha(ROOT/path) == sha(run/"sources"/Path(path).name) == h
    for path, h in design["inputs"].items():
        assert sha(previous/path) == h
    fe = {e["key"]:e for e in read(previous/"forecast-status.json")["cases"]}
    te = {e["key"]:e for e in read(previous/"status.json")["cases"]}
    weights = dict(np.load(previous/"weights.npz"))
    checks, outcomes, terms, primary, witness_count = [], [], [], [], 0
    for case, ent in zip(read(run/"sources/cases242.json"), status["cases"], strict=True):
        key = case["key"]
        assert key == ent["key"]
        path = run/"cases"/(key+".npz")
        meta = read(path.with_suffix(".json"))
        assert sha(path) == meta["arrays_sha256"] == ent["arrays_sha256"] and sha(path.with_suffix(".json")) == ent["record_sha256"]
        assert meta["prior_forecast_sha256"] == fe[key]["arrays_sha256"] == sha(previous/"forecasts"/path.name)
        assert meta["prior_native_sha256"] == te[key]["arrays_sha256"] == sha(previous/"observations"/path.name)
        for c in meta["checks"]:
            assert np.isfinite(c["error"]) and c["error"] <= c["bound"]
        if key in design["witness_cases"]:
            wp = run/"witnesses"/path.name
            assert sha(wp) == meta["witness_sha256"] == ent["witness_sha256"]
            with np.load(wp) as w:
                for arm in ARMS[:-1]:
                    for layer in design["witness_blocks"]:
                        prefix = f"{arm}_{layer}_"
                        q, k, v, raw, mask, prob, output = (w[prefix+n].astype(float) for n in ("q", "k", "v", "raw", "mask", "p", "y"))
                        groups, scale = int(w[prefix+"groups"]), float(w[prefix+"scaling"])
                        checked(checks, "CPU64_witness_QK", q@np.repeat(k, groups, axis=0).swapaxes(-1, -2)*scale, raw, 1e-4)
                        checked(checks, "CPU64_witness_softmax", softmax(raw+mask), prob, 1e-4)
                        checked(checks, "CPU64_witness_PV", prob@np.repeat(v, groups, axis=0), output, 1e-4)
                        witness_count += 1
        else:
            assert meta["witness_sha256"] is ent["witness_sha256"] is None
        with np.load(path) as d, np.load(previous/"forecasts"/path.name) as f, np.load(previous/"observations"/path.name) as t:
            state = {a:d[a+"_trajectory"].astype(float) for a in ARMS}
            scores = {a:d[a+"_scores"].astype(float) for a in ARMS}
            margins = {a:float(np.diff(s)[0]) for a, s in scores.items()}
            checked(checks, "False_native_suffix", state["F"], f["false_trajectory"][9:], 0.)
            checked(checks, "True_native_suffix", state["T"], t["true_trajectory"][9:], 0.)
            checked(checks, "zero_clamp_suffix", state["Z"], state["T"], 0.)
            checked(checks, "joint_clamp_H_receivers", state["B"][:, 1:], state["H"][:, 1:], 0.)
            checked(checks, "joint_clamp_H_scores", scores["B"], scores["H"], 0.)
            B, dm = state["T"]-state["F"], margins["T"]-margins["F"]
            for arm in ARMS:
                assert state[arm].shape == B.shape and np.isfinite(state[arm]).all() and np.isfinite(scores[arm]).all()
                checked(checks, "CPU64_finite_endpoint_head", head(weights, state[arm][-1, -1]), scores[arm], .005)
                for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                    truth, value = B[-1, pick], (state[arm]-state["F"])[-1, pick]
                    den = float(np.vdot(truth, truth))
                    rel, cosine = geometry(value, truth)
                    outcomes.append(dict(key=key, **case["group"], arm=arm, scope=scope,
                        target_norm=float(np.linalg.norm(truth)), relative_error=rel, cosine=cosine,
                        relative_norm=float(np.linalg.norm(value)/max(np.linalg.norm(truth), 1e-30)),
                        signed_share=float(np.vdot(value, truth)/max(den, 1e-30)),
                        native_margin_change=dm, margin_change=margins[arm]-margins["F"],
                        remaining_margin_fraction=(margins[arm]-margins["F"])/dm,
                        margin_error=abs(margins[arm]-margins["T"])))
            b = next(r for r in reversed(outcomes) if r["arm"] == "B" and r["scope"] == "verdict")
            primary.append(dict(key=key, **case["group"], passes=bool(b["target_norm"] >= .001 and abs(dm) >= .02
                and abs(b["signed_share"]) <= .10 and abs(b["remaining_margin_fraction"]) <= .10)))
            families = {
                "key_value":dict(key=state["T"]-state["K"], value=state["T"]-state["V"], interaction=state["K"]+state["V"]-state["T"]-state["B"]),
                "direct_relay":dict(direct=state["T"]-state["D"], relay=state["T"]-state["L"], interaction=state["D"]+state["L"]-state["T"]-state["B"])}
            # The interaction sign here allocates REMOVED effect T-B. The protocol's
            # raw second finite difference is its negative; save both explicitly.
            for family, allocations in families.items():
                checked(checks, "removed_effect_factorial_closure", sum(allocations.values()), state["T"]-state["B"], 1e-10)
                for term, value in allocations.items():
                    if term == "interaction":
                        allocations_to_save = ((term, value), ("raw_second_difference", -value))
                    else:
                        allocations_to_save = ((term, value),)
                    for label, vector in allocations_to_save:
                        for i in range(19):
                            for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                                target, effect = B[i, pick], vector[i, pick]
                                terms.append(dict(key=key, **case["group"], family=family, term=label, layer=i+10, scope=scope,
                                    signed_share=float(np.vdot(effect, target)/max(np.vdot(target, target), 1e-30)),
                                    relative_norm=float(np.linalg.norm(effect)/max(np.linalg.norm(target), 1e-30))))
        print("communication audit", key[:8], flush=True)
    assert witness_count == 132
    assert status["counts"] == dict(full_forwards=352, H_suffix_blocks=608, H_finite_heads=32, extra_QK_softmax_PV_each=5472, witness_extra_QK=132)
    for name, rows in (("outcomes", outcomes), ("signed-contrasts", terms), ("primary", primary)):
        table(run/(name+".csv"), rows)
    write_json(run/"audit.json", dict(status="passed", retrospective=True, primary=decision(primary), checks=checks,
        witness_coverage=dict(contexts=4, blocks=[10, 19, 28], arms=11, kernels=132, full_forward_reconstruction=False),
        scorer_sha256=sha(__file__), design_sha256=sha(run/"design.json"), seal_sha256=sha(run/"seal.json"),
        csv_hashes={n:sha(run/(n+".csv")) for n in ("outcomes", "signed-contrasts", "primary")},
        interpretation="Finite retrospective necessity diagnostic; no prospective predictor, learned circuit, speedup or fresh confirmation."))
    print(json.dumps(decision(primary), indent=2))


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
