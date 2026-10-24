"""Frozen one-site source MLP primitive factorial on32 old276 cases."""

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
from .core225b import array, capture_mlp, load_model, observe
from .primitive278 import ARMS, BITS, SOURCES, Engine, local, random_writes
from .run272 import runtime, stamp

ROOT = Path(__file__).resolve().parents[2]


def main(reference, out):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 1536 * 1024**2
    previous, prior_status = read(reference / "design.json"), read(reference / "status.json")
    assert previous["experiment"] == 276 and prior_status["status"] == "complete"
    assert read(reference / "audit.json")["status"] == "passed"
    for name, h in previous["source_hashes"].items():
        assert sha(ROOT / name) == h
    out.mkdir(parents=True)
    for name in ("cases", "sources"):
        (out / name).mkdir()
    source_hashes = {name: sha(ROOT / name) for name in SOURCES}
    for name in source_hashes:
        shutil.copy2(ROOT / name, out / "sources" / Path(name).name)
    save_json(out / "design.json", dict(experiment=278, retrospective=True, cases=previous["cases"],
        partners=previous["partners"], arms=ARMS, source_hashes=source_hashes,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        installed_sources=installed_sources(), checkpoint=locate(ROOT)[1], began=stamp(),
        reference_hashes={n: sha(reference / n) for n in ("design.json", "status.json", "audit.json")},
        primary="N100 sufficiency: original native5%/.99/.10nat + local MLPchange10%/.995 + material whole-first-MLP T-Z effect25%/.95/finite margin + paired first-source/verdict25%/.95/material/finite margin, >=24/32 and>=6/8 each stratum. Separate nonlinear T-N000 restoration25%/.95/material and random controls; no alternate-arm primary rescue.",
        boundary="One-site old-context primitive intervention, native and carrier incoming points, finite receivers/final readout. No fresh/compact predictor, learned feature/circuit, efficiency, priority or complete QB solution."))
    model = load_model(True)
    save_json(out / "runtime.json", runtime(model))
    layer = model.model.layers[0]
    originals = {r["key"]: r for r in prior_status["cases"]}
    entries, raw_bytes = [], 0
    for case in previous["cases"]:
        assert shutil.disk_usage(out).free >= 768 * 1024**2
        key, start, length = case["key"], case["position"], len(case["false_ids"])
        src = reference / "cases" / (key + ".npz")
        assert sha(src) == originals[key]["arrays_sha256"]
        with np.load(src) as old:
            points = {n: old["point_" + q].copy() for n, q in (("F", "F"), ("C", "C"), ("N", "T"))}
            anchor = {name: old["node_F_" + name].copy() for name in ("x", "z", "gate", "up", "phi", "product", "write")}
            with torch.no_grad():
                nodes = {p + bits: local(layer, points[p], anchor, bits, length, start) for p in ("F", "N", "C") for bits in BITS}
            for bits in BITS:
                for name in anchor:
                    assert np.array_equal(nodes["F" + bits][name], anchor[name]), (bits, name)
            for p, label in (("N", "T"), ("C", "C")):
                for name in anchor:
                    assert np.array_equal(nodes[p + "111"][name], old["node_" + label + "_" + name])
            random, draws = random_writes(nodes, key)
            values = {**{"point_" + n: v for n, v in points.items()},
                      **{"anchor_" + n: v for n, v in anchor.items()},
                      **{"random_" + n: v for n, v in draws.items()},
                      **{"proposal_" + n: v for n, v in random.items()}}
            for name, node in nodes.items():
                # Save every source primitive operand. No intermediate receiving trajectories.
                values.update({"node_" + name + "_" + n: v for n, v in node.items()})
            prefix, native_tails, records = None, {}, {}
            # Actual unhooked F/T controls, not merely a full-valued policy.
            for arm in ("F", "T"):
                with capture_mlp(layer) as captured:
                    states, _, score = observe(model, torch.tensor([case["false_ids" if arm == "F" else "true_ids"]], device="mps"))
                hashes = [digest(array(s[0, :start])) for s in states]
                if prefix is None:
                    prefix = hashes
                assert hashes == prefix
                source = np.stack([array(s[0, start]) for s in states])
                final, score = array(states[-1][0, start:]), array(score)
                assert np.array_equal(source, old[arm + "_source"])
                assert np.array_equal(final, old[arm + "_final"]) and np.array_equal(score, old[arm + "_scores"])
                native_tails[arm] = [digest(array(s[0, start:])) for s in states]
                values.update({arm + "_source": source[0], arm + "_final": final, arm + "_scores": score})
                records[arm] = {name: array(v[0, start]) for name, v in captured.items()}
                for name, v in records[arm].items():
                    expected = points["F" if arm == "F" else "N"] if name == "mid" else nodes[("F" if arm == "F" else "N") + "111"][name]
                    assert np.array_equal(v, expected), (arm, name)
                values.update({arm + "_actual_" + n: v for n, v in records[arm].items()})
            with Engine(model, case, points, anchor, nodes, random) as engine, torch.no_grad():
                for arm in (a for a in ARMS if a not in ("F", "T")):
                    states, score, record = engine.forward(arm)
                    assert [digest(array(s[0, :start])) for s in states] == prefix
                    source = np.stack([array(s[0, start]) for s in states])
                    final = array(states[-1][0, start:])
                    if arm in ("N111", "FZ"):
                        label = "T" if arm == "N111" else "F"
                        assert [digest(array(s[0, start:])) for s in states] == native_tails[label]
                        assert np.array_equal(score, values[label + "_scores"])
                    if arm in ("Z", "C111"):
                        label = "Z" if arm == "Z" else "C"
                        assert np.array_equal(source, old[label + "_source"])
                        assert np.array_equal(final, old[label + "_final"]) and np.array_equal(score, old[label + "_scores"])
                    values.update({arm + "_source": source[0], arm + "_final": final, arm + "_scores": score})
                    values.update({arm + "_actual_" + n: v for n, v in record.items()})
            for name in ("source", "final", "scores"):
                assert np.array_equal(values["N111_" + name], values["T_" + name])
                assert np.array_equal(values["FZ_" + name], values["F_" + name])
        p = out / "cases" / (key + ".npz")
        np.savez_compressed(p, **values)
        meta = dict(key=key, arrays_sha256=sha(p), input_sha256=originals[key]["arrays_sha256"],
            prefix_hashes=prefix, ordinary_F_T_unhooked=True, native_replay_exact=True,
            native_self_and_zero_full_tails_exact=True, archived_Z_C111_exact=True,
            all_F_primitive_zero_edits_exact=True, only_source_MLP1_primitives_changed=True,
            stored="All first-source primitive operands and first source output, final receiving tails and scores; intermediate receiving/source trajectories checked by hash/replay but not duplicated.")
        save_json(p.with_suffix(".json"), meta)
        entries.append(dict(key=key, arrays_sha256=sha(p), metadata_sha256=sha(p.with_suffix(".json")), bytes=p.stat().st_size))
        raw_bytes += p.stat().st_size
        assert raw_bytes <= 384 * 1024**2
        print("278", len(entries), "/32 complete", flush=True)
    save_json(out / "status.json", dict(status="complete", ended=stamp(), cases=entries, raw_bytes=raw_bytes,
        counts=dict(full_forwards=736, extra_full_shaped_local_MLP_maps=768,
                    source_norm_interventions=21 * 32, source_SiLU_interventions=21 * 32,
                    source_product_interventions=21 * 32, random_write_installs=96)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-276", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.reference_276.resolve(), args.out.resolve())
    except Exception as exc:
        if args.out.exists() and not (args.out / "failure.json").exists():
            save_json(args.out / "failure.json", dict(error=repr(exc), time=stamp()))
        raise
