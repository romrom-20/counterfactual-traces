"""Frozen first-source-MLP gain diagnostic; all32 old275 contexts."""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .audit229 import sha
from .audit272 import read
from .cache272 import digest, installed_sources, save_json
from .checkpoint238 import locate, weights
from .core225b import array, load_model
from .firstmlp276 import ARMS, ROOT, SOURCES, SiteEngine, coordinates, finite, proposals, tangent
from .run272 import runtime, stamp
from .source273 import paired


def main(reference, out):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 1536 * 1024**2
    previous, status = read(reference / "design.json"), read(reference / "status.json")
    assert previous["experiment"] == 275 and status["status"] == "complete"
    assert read(reference / "audit.json")["status"] == "passed"
    for name, h in previous["source_hashes"].items():
        assert sha(ROOT / name) == h
    out.mkdir(parents=True)
    for name in ("cases", "sources"):
        (out / name).mkdir()
    sources = {name: sha(ROOT / name) for name in SOURCES}
    for name in sources:
        shutil.copy2(ROOT / name, out / "sources" / Path(name).name)
    folder, checkpoint = locate(ROOT)
    partners = {a["key"]: b["key"] for pair in paired(previous["cases"]) for a, b in (pair, pair[::-1])}
    save_json(out / "design.json", dict(experiment=276, retrospective=True, cases=previous["cases"], partners=partners, arms=ARMS,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes=sources, installed_sources=installed_sources(), checkpoint=checkpoint,
        reference_hashes={n: sha(reference / n) for n in ("design.json", "status.json", "audit.json")}, began=stamp(),
        primary="S joint original native5%/.99/.10nat + paired first-source and final-verdict contrast25%/.95/material/finite paired margin passes >=24/32 and >=6/8 each stratum. Separate gain explanation requires local and downstream U-L correction25%/.95/margin, >=2/3 matched controls failing, same24/32+6/8.",
        boundary="Retrospective first-source MLP diagnostic with old native attention inputs; all other blocks/receivers finite. S/L use finite carrier MLP and smooth CPU64 tangents, never native True MLP output as predictor input. No compact/efficient full predictor, learned feature or general QB solution."))
    model = load_model(True)
    save_json(out / "runtime.json", runtime(model))
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:
        w = weights(handle, read(folder / "config.json"), 0, "mlp")
    entries = {r["key"]: r for r in status["cases"]}
    completed, raw_bytes = [], 0
    for case in previous["cases"]:
        assert shutil.disk_usage(out).free >= 768 * 1024**2
        key, start, length = case["key"], case["position"], len(case["false_ids"])
        original = reference / "cases" / (key + ".npz")
        assert sha(original) == entries[key]["arrays_sha256"]
        with np.load(original) as old:
            points, components = coordinates(old)
            with torch.no_grad():
                nodes = {name: finite(model.model.layers[0], point, length, start) for name, point in points.items()}
            assert np.array_equal(nodes["F"]["write"], old["F_mlpwrite"][0])
            assert np.array_equal(nodes["T"]["write"], old["N_mlpwrite"][0])
            derivatives = {name: tangent(w, points[name], components["read"]) for name in ("F", "C")}
            writes, gain, random = proposals(nodes, derivatives, key)
            values = {**{"point_" + n: v for n, v in points.items()},
                      **{"component_" + n: v for n, v in components.items()}, "gain": gain,
                      **{"random_" + n: v for n, v in random.items()},
                      **{"proposal_" + n: v for n, v in writes.items()}}
            for name, result in nodes.items():
                values.update({"node_" + name + "_" + n: v for n, v in result.items()})
            for name, result in derivatives.items():
                values.update({"derivative_" + name + "_" + n: v for n, v in result.items()})
            prefixes = None
            with SiteEngine(model, case, points, nodes, writes) as engine, torch.no_grad():
                for arm in ARMS:
                    states, scores, record = engine.forward(arm)
                    hashes = [digest(array(s[0, :start])) for s in states]
                    if arm == "F":
                        prefixes = hashes
                    assert hashes == prefixes
                    values[arm + "_source"] = np.stack([array(s[0, start]) for s in states])
                    values[arm + "_final"], values[arm + "_scores"] = array(states[-1][0, start:]), scores
                    values.update({arm + "_" + n: v for n, v in record.items()})
                    if arm in ("F", "T", "U"):
                        label = "F" if arm == "F" else "N"
                        assert all(np.array_equal(array(s[0, start:]), old[label + "_states"][i]) for i, s in enumerate(states))
                        assert np.array_equal(scores, old[label + "_scores"])
            for a, b in (("U", "T"), ("FZ", "F")):
                for node in ("source", "final", "scores"):
                    assert np.array_equal(values[a + "_" + node], values[b + "_" + node])
        path = out / "cases" / (key + ".npz")
        np.savez_compressed(path, **values)
        save_json(path.with_suffix(".json"), dict(key=key, arrays_sha256=sha(path), input_sha256=entries[key]["arrays_sha256"],
            native_F_T_U_exact=True, full_prefix_hashes=prefixes, prefix_all_arms_exact=True,
            site_only_install=True, carrier_primal_for_S_L=True, self_and_zero_exact=True,
            stored="All source trajectories and final receiving rows; intermediate receiving trajectories deliberately not duplicated."))
        completed.append(dict(key=key, arrays_sha256=sha(path), metadata_sha256=sha(path.with_suffix(".json")), bytes=path.stat().st_size))
        raw_bytes += path.stat().st_size
        assert raw_bytes <= 200 * 1024**2
        print("276", len(completed), "/32", case["group"]["schema"], case["group"]["reverse"], "complete", flush=True)
    save_json(out / "status.json", dict(status="complete", cases=completed, raw_bytes=raw_bytes, ended=stamp(),
        counts=dict(full_forwards=448, finite_first_MLP_maps=128, CPU64_MLP_tangents=64,
                    CPU64_tangent_dense_matvec=320)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-275", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.reference_275.resolve(), args.out.resolve())
    except Exception as exc:
        if args.out.exists() and not (args.out / "failure.json").exists():
            save_json(args.out / "failure.json", dict(error=repr(exc), time=stamp()))
        raise
