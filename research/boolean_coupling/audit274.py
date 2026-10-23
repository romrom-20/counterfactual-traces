"""Independent operand reconstruction, finite primitive checks and decisions for274."""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as F
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import sha
from .audit272 import margin, read, table, vector
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate, read_head
from .core225b import array
from .fields267 import PublicHead

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("U", "P", "C", "D", "E", "R")
EVALUATED = (*FAMILIES, "J", "X", "Z")
ALL = ("F", *EVALUATED, "FZ", "T")


def maximum(a, b):
    return float(np.max(abs(a.astype(float) - b.astype(float))))


def main(run, reference):
    assert not (run / "audit.json").exists() and not (run / "failure-audit.json").exists()
    design, status = read(run / "design.json"), read(run / "status.json")
    assert design["experiment"] == 274 and design["retrospective"] and status["status"] == "complete"
    assert tuple(design["arms"]) == ("F", *FAMILIES, "J", "X", "Z", "FZ", "T")
    assert design["installed_sources"] == installed_sources()
    for name, value in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == value
    for name, value in design["reference_hashes"].items():
        assert sha(reference / name) == value
    old_status = read(reference / "status.json")
    old_entries = {r["key"]: r for r in old_status["cases"]}
    case_by_key = {c["key"]: c for c in design["cases"]}
    assert design["cases"] == read(reference / "design.json")["cases"]
    assert design["fixed_reference"] == design["cases"][0]["key"]
    folder, checkpoint = locate(ROOT)
    assert checkpoint == design["checkpoint"]
    config, cpu_head = read(folder / "config.json"), read_head(folder)
    checks, raw, emission, layerwise, outcomes, effects, heads = [], [], [], [], [], [], []
    counts = Counter()

    def check(name, a, b, bound=0.):
        error = maximum(a, b)
        checks.append(dict(name=name, error=error, bound=bound))
        assert np.isfinite(error) and error <= bound, checks[-1]

    def old(key):
        p = reference / "cases" / (key + ".npz")
        assert sha(p) == old_entries[key]["arrays_sha256"]
        with np.load(p) as d:
            return {n: d[n].copy() for n in ("F_states", "G_source", "F_norm", "G_norm", "F_V", "G_V", "F_scores", "T_scores", "T_states", "X_states", "X_scores", "Z_states", "Z_scores")}

    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle, torch.no_grad():
        head = PublicHead(handle, config)
        sliced = handle.get_slice("model.embed_tokens.weight")
        embeddings = np.concatenate([sliced[n:n + 1].float().numpy() for n in (895, 830)])
        matrices, norms = [], []
        for i in range(28):
            prefix = f"model.layers.{i}."
            gamma = handle.get_tensor(prefix + "input_layernorm.weight").float().numpy()
            w = handle.get_tensor(prefix + "self_attn.v_proj.weight").float().numpy()
            bias = handle.get_tensor(prefix + "self_attn.v_proj.bias").float().numpy()
            norm = qwen.Qwen2RMSNorm(1536, config["rms_norm_eps"]).to(device="mps", dtype=torch.float32)
            norm.weight.copy_(torch.tensor(gamma, device="mps"))
            norms.append(norm)
            matrices.append((gamma, w, bias, torch.tensor(w, device="mps"), torch.tensor(bias, device="mps")))
        calibration = old(design["fixed_reference"])
        for entry in status["cases"]:
            key = entry["key"]
            case = case_by_key[key]
            start, length = case["position"], len(case["false_ids"])
            path = run / "cases" / (key + ".npz")
            assert sha(path) == entry["arrays_sha256"] and sha(path.with_suffix(".json")) == entry["metadata_sha256"]
            meta = read(path.with_suffix(".json"))
            assert all(meta[k] for k in ("source_primal_replay_exact", "source_KV_unchanged", "prefix_and_source_exact", "self_and_zero_exact", "F_T_X_Z_archived_replay_exact"))
            paired_key = design["partners"][key]
            assert meta["donor"] == paired_key
            a, b = case["group"], case_by_key[paired_key]["group"]
            assert design["partners"][paired_key] == key and a["schema"] != b["schema"]
            assert {n: v for n, v in a.items() if n != "schema"} == {n: v for n, v in b.items() if n != "schema"}
            for k, s in meta["inputs"].items():
                assert s == old_entries[k]["arrays_sha256"]
            own, donor = old(key), old(paired_key)

            def coordinates(bank):
                center = np.vstack((embeddings[0], bank["F_states"][:-1, 0])).astype(float)
                endpoint = np.vstack((embeddings[1], bank["G_source"][:-1])).astype(float)
                return center, endpoint - center

            x, h = coordinates(own)
            y, k = coordinates(donor)
            _, fixed = coordinates(calibration)
            expected = dict(U=(x, h), P=(x, k), C=(x, fixed), D=(y, h),
                            E=(x, np.tile((embeddings[1].astype(float) - embeddings[0].astype(float)), (28, 1))))
            with np.load(path) as data:
                check("checkpoint_embeddings", data["embeddings"], embeddings)
                for arm, (center, step) in expected.items():
                    check(arm + "_center_provenance", data["geom_" + arm + "_center"], center)
                    check(arm + "_step_provenance", data["geom_" + arm + "_step"], step)
                check("random_center", data["geom_R_center"], x)
                check("random_input_norm", np.linalg.norm(data["geom_R_step"], axis=-1), np.linalg.norm(k, axis=-1), 1e-10)
                rng = np.random.default_rng(int(hashlib.sha256((key + ":reanchor274").encode()).hexdigest()[:16], 16))
                draw = rng.normal(size=k.shape)
                draw = draw / np.linalg.norm(draw, axis=-1, keepdims=True) * np.linalg.norm(k, axis=-1, keepdims=True)
                check("random_direction_provenance", data["geom_R_step"], draw)
                for arm in FAMILIES:
                    for i in range(28):
                        gamma, weight, bias, tw, tb = matrices[i]
                        center, step = data["geom_" + arm + "_center"][i], data["geom_" + arm + "_step"][i]
                        for suffix, value in (("0", center), ("1", center + step)):
                            state = torch.zeros((1, length, 1536), device="mps")
                            cast = value.astype(np.float32)
                            state[0, start] = torch.tensor(cast, device="mps")
                            normalized = norms[i](state)
                            public_v = array(F.linear(normalized, tw, tb)[0, start]).reshape(2, 128)
                            check("public_RMS", array(normalized[0, start]), data["geom_" + arm + "_z" + suffix][i])
                            check("public_V", public_v, data["geom_" + arm + "_v" + suffix][i])
                            z = cast.astype(float) / np.sqrt(np.mean(cast.astype(float) ** 2) + config["rms_norm_eps"]) * gamma.astype(float)
                            v = (z @ weight.astype(float).T + bias.astype(float)).reshape(2, 128)
                            for node, a, b in (("RMS", z, data["geom_" + arm + "_z" + suffix][i]), ("V", v, data["geom_" + arm + "_v" + suffix][i])):
                                error = maximum(a, b)
                                raw.append(dict(key=key, arm=arm, layer=i + 1, side=suffix, node=node,
                                                error=error, original_bound=.0001, passes=error <= .0001))
                    check(arm + "_emission_identity", data["delta_" + arm], data["geom_" + arm + "_v1"].astype(float) - data["geom_" + arm + "_v0"].astype(float))
                for node, original in (("z0", "F_norm"), ("z1", "G_norm"), ("v0", "F_V"), ("v1", "G_V")):
                    check("native_source_" + node, data["geom_U_" + node], own[original])
                check("raw_donor_emission", data["delta_X"], donor["G_V"].astype(float) - donor["F_V"].astype(float))
                check("additive_corners", data["delta_J"], data["delta_P"] + data["delta_D"] - data["delta_X"])
                for arm in ("F", "T", "X", "Z"):
                    for node in ("states", "scores"):
                        check("archived_" + arm + "_" + node, data[arm + "_" + node], own[arm + "_" + node])
                for a, b in (("U", "T"), ("FZ", "F")):
                    for node in ("states", "scores"):
                        check("replay_" + a + "_" + node, data[a + "_" + node], data[b + "_" + node])
                for arm in ALL:
                    expected_source = own["F_states"][:, 0] if arm in ("F", "FZ") else own["G_source"]
                    check("native_source_state_" + arm, data[arm + "_states"][:, 0], expected_source)
                    if arm not in ("F", "T"):
                        candidate = (own["F_V"].astype(float) + data["delta_" + arm]).astype(np.float32)
                        check("actual_install_" + arm, data[arm + "_installed_V"], candidate)
                    scores = head(data[arm + "_states"][-1, -1], length)
                    check("finite_public_head_" + arm, scores, data[arm + "_scores"])
                    endpoint = data[arm + "_states"][-1, -1].astype(float)
                    z = endpoint / np.sqrt(np.mean(endpoint ** 2) + float(cpu_head["final_eps"])) * cpu_head["final_norm"]
                    cpu = z @ cpu_head["final_head"].T
                    check("CPU64_head_bridge", cpu, data[arm + "_scores"], .005)
                    heads.append(dict(key=key, arm=arm, public_error=maximum(scores, data[arm + "_scores"]), cpu_error=maximum(cpu, data[arm + "_scores"])))
                target = data["T_states"][-1].astype(float) - data["F_states"][-1].astype(float)
                effect_target = data["U_states"][-1].astype(float) - data["Z_states"][-1].astype(float)
                mu, mz, mt = (margin(data[a + "_scores"]) for a in ("U", "Z", "T"))
                for arm in EVALUATED:
                    row = dict(key=key, **case["group"], arm=arm)
                    nrow, erow = dict(row), dict(row)
                    nv, ev = [], []
                    for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                        n = vector((data[arm + "_states"][-1].astype(float) - data["F_states"][-1].astype(float))[pick], target[pick])
                        e = vector((data[arm + "_states"][-1].astype(float) - data["Z_states"][-1].astype(float))[pick], effect_target[pick])
                        nrow.update({scope + "_" + k: v for k, v in n.items()})
                        erow.update({scope + "_" + k: v for k, v in e.items()})
                        nv.append(n["norm"] >= .001 and n["relative"] <= .05 and n["cosine"] >= .99)
                        ev.append(e["norm"] >= .001 and e["relative"] <= .25 and e["cosine"] >= .95)
                    m = margin(data[arm + "_scores"])
                    nrow.update(margin_error=abs(m - mt), passes=bool(all(nv) and abs(m - mt) <= .1))
                    erow.update(margin_error=abs(m - mu), target_margin=mu - mz, margin_bound=.005 + .25 * abs(mu - mz),
                        passes=bool(all(ev) and abs(mu - mz) >= .02 and abs(m - mu) <= .005 + .25 * abs(mu - mz)))
                    outcomes.append(nrow)
                    effects.append(erow)
                    if arm == "Z":
                        continue
                    m = vector(data["delta_" + arm].astype(float), data["delta_U"].astype(float))
                    good = 0
                    for i in range(28):
                        v = vector(data["delta_" + arm][i].astype(float), data["delta_U"][i].astype(float))
                        ok = v["norm"] >= .001 and v["relative"] <= .10 and v["cosine"] >= .995
                        good += ok
                        layerwise.append(dict(row, layer=i + 1, **v, passes=bool(ok)))
                    emission.append(dict(row, **m, good_layers=int(good), passes=bool(m["norm"] >= .001 and m["relative"] <= .10 and m["cosine"] >= .995 and good >= 24)))
                counts["cases"] += 1
            print("274 audit", counts["cases"], "/32", flush=True)
    for name, rows in (("native.csv", outcomes), ("effects.csv", effects), ("emission.csv", emission),
                       ("emission-layers.csv", layerwise), ("raw-primitives.csv", raw), ("heads.csv", heads)):
        table(run / name, rows)
    combined = []
    for e in emission:
        f = next(r for r in effects if r["key"] == e["key"] and r["arm"] == e["arm"])
        combined.append(dict(key=e["key"], schema=e["schema"], reverse=e["reverse"], arm=e["arm"],
                             emitter_pass=e["passes"], effect_pass=f["passes"], passes=e["passes"] and f["passes"]))
    table(run / "combined.csv", combined)
    nc = {a: sum(r["passes"] for r in outcomes if r["arm"] == a) for a in EVALUATED}
    ec = {a: sum(r["passes"] for r in effects if r["arm"] == a) for a in EVALUATED}
    cc = {a: sum(r["passes"] for r in combined if r["arm"] == a) for a in EVALUATED if a != "Z"}
    strata = {f"{s}/{rev}": sum(r["passes"] for r in combined if r["arm"] == "P" and r["schema"] == s and r["reverse"] == rev)
              for s in ("generic", "descriptive") for rev in (0, 1)}
    save_json(run / "audit.json", dict(status="passed", experiment=274, retrospective=True,
        design_sha256=sha(run / "design.json"), status_sha256=sha(run / "status.json"), source_sha256=sha(__file__),
        numerical_checks=len(checks), numerical_max_error=max(r["error"] for r in checks),
        raw_checks=dict(total=len(raw), failed=sum(not r["passes"] for r in raw), original_bound=.0001),
        native_counts=nc, effect_counts=ec, combined_counts=cc, P_strata=strata,
        primary_passes=cc["P"] >= 24 and min(strata.values()) >= 6,
        csv_sha256={p.name: sha(p) for p in run.glob("*.csv")},
        boundary="Old native value-path geometry diagnostic, native source keys retained; no fresh/cheap full forecast, learned feature or general Question B solution. Original numerical failures not recertified."))
    print(dict(native=nc, effects=ec, combined=cc, primary=cc["P"] >= 24 and min(strata.values()) >= 6), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference-273", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.run.resolve(), args.reference_273.resolve())
    except Exception as exc:
        if args.run.exists() and not (args.run / "failure-audit.json").exists():
            save_json(args.run / "failure-audit.json", dict(error=repr(exc)))
        raise
