"""Freeze-first retrospective source-residual reanchoring test on all32 old cases."""

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
from .reanchor274 import (
    ARMS,
    FAMILIES,
    ROOT,
    SOURCES,
    ValueEngine,
    check_own,
    fields,
    geometry,
    load_bank,
)
from .run272 import runtime, stamp
from .source273 import paired


def main(reference, out):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 1200 * 1024**2
    old, old_status = read(reference / "design.json"), read(reference / "status.json")
    assert read(reference / "audit.json")["status"] == "passed"
    assert old["experiment"] == 273 and old_status["status"] == "complete"
    for name, value in old["source_hashes"].items():
        assert sha(ROOT / name) == value
    cases = old["cases"]
    entries = {e["key"]: e for e in old_status["cases"]}
    partners = {a["key"]: b["key"] for pair in paired(cases) for a, b in (pair, pair[::-1])}
    fixed = cases[0]["key"]  # Ordinal, fixed before computation, never selected by result.
    out.mkdir(parents=True)
    for name in ("cases", "sources"):
        (out / name).mkdir()
    sources = {name: sha(ROOT / name) for name in SOURCES}
    for name in sources:
        shutil.copy2(ROOT / name, out / "sources" / Path(name).name)
    _, checkpoint = locate(ROOT)
    save_json(out / "design.json", dict(experiment=274, retrospective=True, cases=cases,
        partners=partners, fixed_reference=fixed, arms=ARMS, families=FAMILIES,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes=sources, installed_sources=installed_sources(), checkpoint=checkpoint,
        reference_hashes={n: sha(reference / n) for n in ("design.json", "status.json", "audit.json")}, began=stamp(),
        primary="P must pass native-value-effect25%/.95/finite-margin rule AND emitter10%/.995 aggregate and >=24/28 layers, in>=24/32 cases and>=6/8 per schema/order. Strict full native5%/.99/.10nat remains a separate endpoint.",
        boundary="Old-context conditional native value path; native source keys retained. Own-True residuals used only by U/D controls. Donor/reference steps are old calibrated trajectories. Not a cheap full False-only forecast, fresh test or learned circuit."))
    model = load_model(True)
    save_json(out / "runtime.json", runtime(model))
    embeddings = array(model.model.embed_tokens.weight[[895, 830]])
    fixed_path = reference / "cases" / (fixed + ".npz")
    assert sha(fixed_path) == entries[fixed]["arrays_sha256"]
    fixed_bank = load_bank(fixed_path)
    completed, raw_bytes = [], 0
    counts = dict(full_forwards=0, intervention_PV=0, source_norm_calls=0, source_V_projection_calls=0)
    for case in cases:
        assert shutil.disk_usage(out).free >= 768 * 1024**2
        key, start = case["key"], case["position"]
        paths = {k: reference / "cases" / (k + ".npz") for k in (key, partners[key])}
        for k, path in paths.items():
            assert sha(path) == entries[k]["arrays_sha256"]
        own, donor = load_bank(paths[key]), load_bank(paths[partners[key]])
        parameters = geometry(own, donor, fixed_bank, embeddings, key)
        with torch.no_grad():
            changes, nodes = fields(model, case, parameters)
        counts["source_norm_calls"] += 2 * len(FAMILIES) * 28
        counts["source_V_projection_calls"] += 2 * len(FAMILIES) * 28
        check_own(nodes, own)
        changes.update(X=donor["G_V"].astype(float) - donor["F_V"].astype(float),
                       Z=np.zeros_like(changes["U"]), FZ=np.zeros_like(changes["U"]))
        changes["J"] = changes["P"] + changes["D"] - changes["X"]
        values = dict(**nodes, embeddings=embeddings,
            **{"delta_" + arm: change for arm, change in changes.items()})
        with ValueEngine(model) as engine, torch.no_grad():
            engine.bank, engine.fields = own, changes
            for arm in ARMS:
                states, _, score, records = engine.forward(case, arm)
                prefix = [digest(array(s[0, :start])) for s in states]
                old_meta = read(paths[key].with_suffix(".json"))
                assert prefix == old_meta["false_prefix_hashes"]
                tail = np.stack([array(s[0, start:]) for s in states])
                expected_source = own["F_states"][:, 0] if arm in ("F", "FZ") else own["G_source"]
                assert np.array_equal(tail[:, 0], expected_source)
                values[arm + "_states"], values[arm + "_scores"] = tail, score
                if arm in ("F", "T", "X", "Z"):
                    for name, value in (("states", tail), ("scores", score)):
                        assert np.array_equal(value, own[arm + "_" + name])
                if arm not in ("F", "T"):
                    values[arm + "_installed_V"] = np.stack([r["installed_V"] for r in records])
            assert np.array_equal(values["U_states"], values["T_states"])
            assert np.array_equal(values["U_scores"], values["T_scores"])
            assert np.array_equal(values["FZ_states"], values["F_states"])
            assert np.array_equal(values["FZ_scores"], values["F_scores"])
        counts["full_forwards"] += engine.counts["full_forwards"]
        counts["intervention_PV"] += engine.counts["intervention_PV"]
        assert engine.counts["source_blocks"] == engine.counts["diagnostic_PV"] == engine.counts["diagnostic_output_projections"] == 0
        path = out / "cases" / (key + ".npz")
        np.savez_compressed(path, **values)
        save_json(path.with_suffix(".json"), dict(key=key, donor=partners[key], fixed_reference=fixed,
            arrays_sha256=sha(path), source_primal_replay_exact=True, source_KV_unchanged=True,
            prefix_and_source_exact=True, self_and_zero_exact=True, F_T_X_Z_archived_replay_exact=True,
            inputs={k: entries[k]["arrays_sha256"] for k in (key, partners[key], fixed)}))
        completed.append(dict(key=key, arrays_sha256=sha(path), metadata_sha256=sha(path.with_suffix(".json")), bytes=path.stat().st_size))
        raw_bytes += path.stat().st_size
        assert raw_bytes <= 600 * 1024**2
        print("274", len(completed), "/32", case["group"]["schema"], case["group"]["reverse"], "complete", flush=True)
    assert counts == dict(full_forwards=384, intervention_PV=8960,
                          source_norm_calls=10752, source_V_projection_calls=10752)
    save_json(out / "status.json", dict(status="complete", cases=completed, counts=counts,
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
