"""Frozen native source-value interchange on all 32 already-observed 269 cases."""

import argparse
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import transformers

from .audit229 import sha
from .audit272 import read
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate
from .core225b import array, load_model
from .run272 import stamp
from .source273 import ARMS, ROOT, SOURCES, Engine, bank_arrays, deltas, digest, paired


def main(out, reference):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 1024 * 1024**2
    cases = read(ROOT / "research/boolean_coupling/cases269.json")
    pairs = paired(cases)
    out.mkdir(parents=True)
    for name in ("cases", "sources", "witnesses"):
        (out / name).mkdir()
    sources = {name: sha(ROOT / name) for name in SOURCES}
    for name in sources:
        shutil.copy2(ROOT / name, out / "sources" / Path(name).name)
    _, checkpoint = locate(ROOT)
    reference_status = None
    if reference is not None:
        reference_status = read(reference / "status.json")
        assert reference_status["status"] == "complete"
    save_json(out / "design.json", dict(experiment=273, retrospective=True, cases=cases,
        pairs=[[a["key"], b["key"]] for a, b in pairs], arms=list(ARMS),
        source_hashes=sources, checkpoint=checkpoint, installed_sources=installed_sources(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        began=stamp(), reference_status_sha256=sha(reference / "status.json") if reference else None,
        native_receiving_oracle=False, source_generation="28 full-shaped hybrid blocks; only source output carried, all receivers reset to observed False at each input. Not a speedup.",
        primary="X accounts for U-Z native source-value effect in >=24/32 and >=6/8 per schema/order, both vector scopes <=25%, cosine>=.95; finite margin effect error <=.005+.25*abs(mU-mZ); materiality .001 and .02nat; U must replay T exactly."))
    model = load_model(True)
    save_json(out / "runtime.json", dict(python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, transformers=transformers.__version__, device="mps", dtype="float32",
        metric_dtype="cpu_float64", checkpoint_revision=model.config._commit_hash,
        attention=model.config._attn_implementation, training=model.training))
    entries, total_bytes = [], 0
    with Engine(model) as engine, torch.no_grad():
        for pair in pairs:
            parents = {}
            for case in pair:
                fs, context, score, frecord = engine.forward(case, "F")
                source, grecord = engine.source(case, fs, context)
                parents[case["key"]] = dict(states=fs, context=context, scores=score,
                    bank=bank_arrays(frecord, grecord, source), prefix=[digest(array(s[0, :case["position"]])) for s in fs])
            for case, donor in (pair, pair[::-1]):
                assert shutil.disk_usage(out).free >= 768 * 1024**2
                key, start = case["key"], case["position"]
                own, other = parents[key], parents[donor["key"]]
                bank = own["bank"]
                engine.bank = bank
                engine.fields = deltas(bank, other["bank"], key)
                engine.local = {arm + "_" + node: [] for arm in ("T", "Z", "X") for node in ("y", "write")}
                engine.witness = {}
                values = dict(**bank, donor_F_V=other["bank"]["F_V"], donor_G_V=other["bank"]["G_V"],
                    F_states=np.stack([array(s[0, start:]) for s in own["states"]]), F_scores=own["scores"],
                    **{"delta_" + a: d for a, d in engine.fields.items()})
                controls = []
                for arm in ARMS:
                    full_states, _, scores, record = engine.forward(case, arm)
                    assert [digest(array(s[0, :start])) for s in full_states] == own["prefix"]
                    states = np.stack([array(s[0, start:]) for s in full_states])
                    values[arm + "_states"], values[arm + "_scores"] = states, scores
                    values[arm + "_source_V"] = np.stack([r["V"] for r in record])
                    values[arm + "_source_K"] = np.stack([r["K"] for r in record])
                    if arm not in ("FZ", "T"):
                        values[arm + "_installed_V"] = np.stack([r["installed_V"] for r in record])
                    controls.append(dict(arm=arm, source_KV_exact=True, installation_exact=True,
                                         unchanged_prefix_and_source_output=True))
                    # Causality and these full-shaped operations must preserve the source exactly.
                    expected = values["F_states"][:, 0] if arm == "FZ" else bank["G_source"]
                    assert np.array_equal(states[:, 0], expected)
                    assert all(r.get("prefix_source_unchanged", True) for r in record)
                    if arm == "T":
                        values["T_norm"] = np.stack([r["norm"] for r in record])
                for node, rows in engine.local.items():
                    values["local_" + node] = np.stack(rows)
                assert np.array_equal(values["U_states"], values["T_states"])
                assert np.array_equal(values["U_scores"], values["T_scores"])
                assert np.array_equal(values["FZ_states"], values["F_states"])
                assert np.array_equal(values["FZ_scores"], values["F_scores"])
                assert np.array_equal(values["T_norm"], bank["G_norm"])
                archived = None
                if reference is not None:
                    old = reference / "cases" / (key + ".npz")
                    ent = next(e for e in reference_status["cases"] if e["key"] == key)
                    assert sha(old) == ent["arrays_sha256"]
                    with np.load(old) as prior:
                        for arm in ("F", "T"):
                            for node in ("states", "scores"):
                                assert np.array_equal(values[arm + "_" + node], prior[arm + "_" + node])
                    archived = dict(arrays_sha256=sha(old), F_T_replay_exact=True)
                path = out / "cases" / (key + ".npz")
                witness_path = out / "witnesses" / path.name
                np.savez_compressed(path, **values)
                np.savez_compressed(witness_path, **engine.witness)
                save_json(path.with_suffix(".json"), dict(key=key, donor=donor["key"], group=case["group"],
                    arrays_sha256=sha(path), witness_sha256=sha(witness_path), controls=controls,
                    source_generator_replay_exact=True, self_install_replay_exact=True,
                    zero_edit_replay_exact=True, historical=archived,
                    false_prefix_hashes=own["prefix"]))
                entries.append(dict(key=key, donor=donor["key"], arrays_sha256=sha(path),
                    metadata_sha256=sha(path.with_suffix(".json")), witness_sha256=sha(witness_path),
                    arrays_bytes=path.stat().st_size, witness_bytes=witness_path.stat().st_size))
                total_bytes += path.stat().st_size + witness_path.stat().st_size
                assert total_bytes <= 600 * 1024**2
                print("273", len(entries), "/32", case["group"]["schema"], case["group"]["reverse"], "complete", flush=True)
    assert len(entries) == 32
    assert engine.counts == dict(full_forwards=288, source_blocks=896, intervention_PV=6272,
                                 diagnostic_PV=1792, diagnostic_output_projections=2688)
    save_json(out / "status.json", dict(status="complete", cases=entries, counts=engine.counts,
        raw_bytes=total_bytes, ended=stamp()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-272", type=Path)
    args = parser.parse_args()
    try:
        main(args.out.resolve(), args.reference_272.resolve() if args.reference_272 else None)
    except Exception as exc:
        if args.out.exists() and not (args.out / "failure.json").exists():
            save_json(args.out / "failure.json", dict(error=repr(exc), time=stamp()))
        raise
