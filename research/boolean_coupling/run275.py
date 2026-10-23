"""Freeze-first native source reading diagnostic on all32 old273 cases."""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from .audit229 import sha
from .audit272 import read
from .cache272 import digest, installed_sources, save_json
from .checkpoint238 import locate
from .core225b import array, load_model
from .reading275 import FACTORS, ROOT, SOURCES, VALUE_ARMS, SourceReading, packed
from .reanchor274 import ValueEngine, load_bank
from .run272 import runtime, stamp


def main(reference, out):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 2 * 1024**3
    design, status = read(reference / "design.json"), read(reference / "status.json")
    assert design["experiment"] == 273 and status["status"] == "complete"
    assert read(reference / "audit.json")["status"] == "passed"
    for name, value in design["source_hashes"].items():
        assert sha(ROOT / name) == value
    out.mkdir(parents=True)
    for name in ("cases", "sources"):
        (out / name).mkdir()
    sources = {name: sha(ROOT / name) for name in SOURCES}
    for name in sources:
        shutil.copy2(ROOT / name, out / "sources" / Path(name).name)
    _, checkpoint = locate(ROOT)
    save_json(out / "design.json", dict(experiment=275, retrospective=True, cases=design["cases"],
        factors=FACTORS, value_arms=VALUE_ARMS,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes=sources, checkpoint=checkpoint, installed_sources=installed_sources(),
        reference_hashes={n: sha(reference / n) for n in ("design.json", "status.json", "audit.json")}, began=stamp(),
        primary="V (False self-value, adaptive probabilities) combined emitter10%/.995/24layers + native value-effect25%/.95/margin passes >=24/32 and >=6/8 per stratum; W (False probabilities, adaptive self-value) combined <=8/32. Strict complete native response reported separately for full factorial and value-only arms.",
        boundary="Old native primitive intervention. Probability includes query and self-key changes, not isolated query attribution. Full-factor forecasts use observed False nodes plus known edit/weights and expensive full finite continuations; not a compact predictor. Value-only arms retain native source K. No fresh test or learned circuit."))
    model = load_model(True)
    save_json(out / "runtime.json", runtime(model))
    entries = {e["key"]: e for e in status["cases"]}
    completed, raw_bytes, all_counts = [], 0, dict(full_forwards=0, source_intervention_PV=0,
        native_corner_PV=0, native_corner_output_projections=0, value_intervention_PV=0)
    for case in design["cases"]:
        assert shutil.disk_usage(out).free >= 768 * 1024**2
        key, start = case["key"], case["position"]
        path = reference / "cases" / (key + ".npz")
        assert sha(path) == entries[key]["arrays_sha256"]
        own = load_bank(path)
        old_meta = read(path.with_suffix(".json"))
        values, changes = {}, {}

        def tail(states):
            assert [digest(array(s[0, :start])) for s in states] == old_meta["false_prefix_hashes"]
            return np.stack([array(s[0, start:]) for s in states])

        with SourceReading(model) as engine, torch.no_grad():
            fstates, context, score, f = engine.forward(case, "F")
            ftail = tail(fstates)
            assert np.array_equal(ftail, own["F_states"]) and np.array_equal(score, own["F_scores"])
            values.update(F_states=ftail, F_scores=score, **packed(f, "F"),
                          rope_cos=array(context["rope"][0]), rope_sin=array(context["rope"][1]))
            for arm in (*FACTORS, "FB"):
                states, _, score, records = engine.forward(case, arm)
                values.update({arm + "_states": tail(states), arm + "_scores": score}, **packed(records, arm))
                assert np.array_equal(values[arm + "_before"][0], array(model.model.embed_tokens.weight[895 if arm == "FB" else 830]))
                if arm == "N":
                    assert np.array_equal(values[arm + "_states"], own["T_states"])
                    assert np.array_equal(score, own["T_scores"])
                    assert np.array_equal(values[arm + "_k"], own["G_K"])
                    assert np.array_equal(values[arm + "_v"], own["G_V"])
                elif arm == "FB":
                    assert np.array_equal(values[arm + "_states"], values["F_states"])
                    assert np.array_equal(score, values["F_scores"])
                else:
                    assert all(r["only_source_query_changed"] for r in records)
                if arm != "FB":
                    changes[arm] = values[arm + "_v"].astype(float) - own["F_V"].astype(float)
            for k, v in engine.counts.items():
                all_counts[k] += v
        changes.update(U=changes["N"], Z=np.zeros_like(changes["N"]), FZ=np.zeros_like(changes["N"]))
        values.update({"delta_" + k: v for k, v in changes.items()})
        with ValueEngine(model) as receiver, torch.no_grad():
            receiver.bank, receiver.fields = own, changes
            for arm in VALUE_ARMS:
                states, _, score, records = receiver.forward(case, arm)
                values["value_" + arm + "_states"] = tail(states)
                values["value_" + arm + "_scores"] = score
                expected = own["F_states"][:, 0] if arm == "FZ" else own["G_source"]
                assert np.array_equal(values["value_" + arm + "_states"][:, 0], expected)
                if arm != "T":
                    values["value_" + arm + "_installed_V"] = np.stack([r["installed_V"] for r in records])
            for arm, original in (("U", "T"), ("T", "T"), ("FZ", "F"), ("Z", "Z")):
                assert np.array_equal(values["value_" + arm + "_states"], own[original + "_states"])
                assert np.array_equal(values["value_" + arm + "_scores"], own[original + "_scores"])
            all_counts["full_forwards"] += receiver.counts["full_forwards"]
            all_counts["value_intervention_PV"] += receiver.counts["intervention_PV"]
            assert receiver.counts["source_blocks"] == receiver.counts["diagnostic_PV"] == receiver.counts["diagnostic_output_projections"] == 0
        path = out / "cases" / (key + ".npz")
        np.savez_compressed(path, **values)
        save_json(path.with_suffix(".json"), dict(key=key, arrays_sha256=sha(path), input_sha256=entries[key]["arrays_sha256"],
            false_and_native_exact=True, unchanged_prefix_all_layers=True, source_query_only_install=True,
            native_corner_B_matches_F=True, native_value_KV_exact=True, self_zero_archived_exact=True,
            full_factor_zero_edit_exact=True))
        completed.append(dict(key=key, arrays_sha256=sha(path), metadata_sha256=sha(path.with_suffix(".json")), bytes=path.stat().st_size))
        raw_bytes += path.stat().st_size
        assert raw_bytes <= 1000 * 1024**2
        print("275", len(completed), "/32", case["group"]["schema"], case["group"]["reverse"], "complete", flush=True)
    assert all_counts == dict(full_forwards=416, source_intervention_PV=3584,
        native_corner_PV=2688, native_corner_output_projections=2688, value_intervention_PV=5376)
    save_json(out / "status.json", dict(status="complete", cases=completed, counts=all_counts,
        raw_bytes=raw_bytes, ended=stamp()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-273", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.reference_273.resolve(), args.out.resolve())
    except Exception as exc:
        if args.out.exists() and not (args.out / "failure.json").exists():
            save_json(args.out / "failure.json", dict(error=repr(exc), time=stamp()))
        raise
