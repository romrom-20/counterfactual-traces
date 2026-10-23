"""Separate finite-primitive replay and scoring for native value interchange273."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as F

from .audit229 import sha
from .audit272 import margin, read, table, vector
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate, read_head
from .core225b import array
from .fields267 import PublicHead
from .source273 import ARMS, ROOT, paired


def metric(value, target):
    result = vector(value, target)
    result["signed_projection"] = float(np.vdot(value.ravel(), target.ravel())) / max(result["norm"] ** 2, 1e-30)
    return result


def check(records, name, error, bound=0.):
    row = dict(name=name, error=float(error), bound=bound, passes=bool(np.isfinite(error) and error <= bound))
    records.append(row)
    assert row["passes"], row


def maximum(a, b):
    return float(np.max(np.abs(a.astype(float) - b.astype(float))))


def main(run):
    assert not (run / "audit.json").exists() and not (run / "failure-audit.json").exists()
    design, status = read(run / "design.json"), read(run / "status.json")
    assert design["experiment"] == 273 and design["retrospective"] and status["status"] == "complete"
    assert design["installed_sources"] == installed_sources()
    for name, value in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == value
    cases = {c["key"]: c for c in design["cases"]}
    partners = {a["key"]: b["key"] for pair in paired(design["cases"]) for a, b in (pair, pair[::-1])}
    assert design["pairs"] == [[a["key"], b["key"]] for a, b in paired(design["cases"])]
    folder, checkpoint = locate(ROOT)
    assert checkpoint == design["checkpoint"]
    config, cpu_head = read(folder / "config.json"), read_head(folder)
    native, effects, source, local, raw, checks, heads = [], [], [], [], [], [], []
    counts = Counter()
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle, torch.no_grad():
        head = PublicHead(handle, config)
        matrices = []
        for i in range(28):
            prefix = f"model.layers.{i}.self_attn."
            matrices.append({n + "_" + kind: handle.get_tensor(prefix + n + "." + kind).float().numpy()
                             for n, kind in (("v_proj", "weight"), ("v_proj", "bias"), ("o_proj", "weight"))})
        for entry in status["cases"]:
            key = entry["key"]
            case = cases[key]
            start, length = case["position"], len(case["false_ids"])
            path = run / "cases" / (key + ".npz")
            wp = run / "witnesses" / path.name
            assert sha(path) == entry["arrays_sha256"] and sha(wp) == entry["witness_sha256"]
            assert sha(path.with_suffix(".json")) == entry["metadata_sha256"]
            meta = read(path.with_suffix(".json"))
            assert entry["donor"] == meta["donor"] == partners[key]
            assert all(meta[n] for n in ("source_generator_replay_exact", "self_install_replay_exact", "zero_edit_replay_exact"))
            assert len(meta["controls"]) == 8 and all(all(r[n] for n in ("source_KV_exact", "installation_exact", "unchanged_prefix_and_source_output")) for r in meta["controls"])
            if design["reference_status_sha256"]:
                assert meta["historical"]["F_T_replay_exact"]
                counts["archived_endpoint_and_score_equalities"] += 4
            with np.load(path) as data, np.load(wp) as witness:
                donor_path = run / "cases" / (partners[key] + ".npz")
                with np.load(donor_path) as donor_data:
                    for node in ("F_V", "G_V"):
                        check(checks, "paired_donor_" + node, maximum(data["donor_" + node], donor_data[node]))
                for a, b in (("U_states", "T_states"), ("U_scores", "T_scores"),
                             ("FZ_states", "F_states"), ("FZ_scores", "F_scores"), ("G_norm", "T_norm")):
                    check(checks, a + "=" + b, maximum(data[a], data[b]))
                check(checks, "generated_native_source", maximum(data["G_source"], data["T_states"][:, 0]))
                own = data["G_V"].astype(float) - data["F_V"].astype(float)
                donor = data["donor_G_V"].astype(float) - data["donor_F_V"].astype(float)
                check(checks, "own_delta", maximum(data["delta_U"], own))
                check(checks, "donor_delta", maximum(data["delta_X"], donor))
                nx, ny = np.linalg.norm(own, axis=-1), np.linalg.norm(donor, axis=-1)
                for arm, target in (("D", nx), ("A", ny), ("R", nx)):
                    check(checks, arm + "_head_norm", maximum(np.linalg.norm(data["delta_" + arm], axis=-1), target), 1e-10)
                for arm in ARMS:
                    baseline = "F" if arm == "FZ" else "G"
                    for node in ("K", "V"):
                        check(checks, arm + "_source_" + node, maximum(data[arm + "_source_" + node], data[baseline + "_" + node]))
                    if arm not in ("FZ", "T"):
                        candidate = (data["F_V"].astype(float) + data["delta_" + arm]).astype(np.float32)
                        check(checks, arm + "_installation", maximum(candidate, data[arm + "_installed_V"]))
                truth = data["T_states"][-1].astype(float) - data["F_states"][-1].astype(float)
                target_effect = data["U_states"][-1].astype(float) - data["Z_states"][-1].astype(float)
                mf, mt, mu, mz = (margin(data[a + "_scores"]) for a in ("F", "T", "U", "Z"))
                assert abs(mt - mf) > 1e-8
                for arm in ("U", "X", "D", "A", "Z", "R"):
                    pred = data[arm + "_states"][-1].astype(float) - data["F_states"][-1].astype(float)
                    response = data[arm + "_states"][-1].astype(float) - data["Z_states"][-1].astype(float)
                    nrow = dict(key=key, donor=partners[key], **case["group"], arm=arm)
                    erow = dict(nrow)
                    native_ok, effect_ok = [], []
                    for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                        v, e = metric(pred[pick], truth[pick]), metric(response[pick], target_effect[pick])
                        nrow.update({scope + "_" + k: v for k, v in v.items()})
                        erow.update({scope + "_" + k: v for k, v in e.items()})
                        native_ok.append(v["norm"] >= .001 and v["relative"] <= .05 and v["cosine"] >= .99)
                        effect_ok.append(e["norm"] >= .001 and e["relative"] <= .25 and e["cosine"] >= .95)
                    m = margin(data[arm + "_scores"])
                    nrow.update(margin_response=m - mf, native_margin_response=mt - mf,
                        margin_ratio=(m - mf) / (mt - mf), margin_error=abs(m - mt),
                        passes=bool(all(native_ok) and abs(m - mt) <= .1))
                    erow.update(margin_effect=m - mz, target_margin_effect=mu - mz,
                        margin_effect_error=abs(m - mu), margin_bound=.005 + .25 * abs(mu - mz),
                        passes=bool(all(effect_ok) and abs(mu - mz) >= .02 and abs(m - mu) <= .005 + .25 * abs(mu - mz)))
                    native.append(nrow)
                    effects.append(erow)
                for arm in ("F", *ARMS):
                    endpoint = data[arm + "_states"][-1, -1]
                    expected = head(endpoint, length)
                    error = maximum(expected, data[arm + "_scores"])
                    check(checks, arm + "_finite_head", error)
                    norm = endpoint.astype(float) / np.sqrt(np.mean(endpoint.astype(float) ** 2) + float(cpu_head["final_eps"]))
                    cpu = (norm * cpu_head["final_norm"]) @ cpu_head["final_head"].T
                    cpu_error = maximum(cpu, data[arm + "_scores"])
                    check(checks, arm + "_cpu_head_bridge", cpu_error, .005)
                    heads.append(dict(key=key, arm=arm, public_error=error, cpu_error=cpu_error, cpu_bound=.005))
                for i in range(28):
                    for h in range(2):
                        v = metric(donor[i, h], own[i, h])
                        source.append(dict(key=key, donor=partners[key], **case["group"], layer=i + 1,
                            head=h, **v, own_norm=nx[i, h], donor_norm=ny[i, h], norm_ratio=ny[i, h] / nx[i, h]))
                    target = data["local_T_write"][i].astype(float) - data["local_Z_write"][i].astype(float)
                    alt = data["local_X_write"][i].astype(float) - data["local_Z_write"][i].astype(float)
                    for scope, pick in (("all", slice(None)), ("verdict", -1)):
                        local.append(dict(key=key, **case["group"], layer=i + 1, scope=scope,
                                          **metric(alt[pick], target[pick])))
                    w = matrices[i]
                    weight = torch.tensor(w["v_proj_weight"], device="mps")
                    bias = torch.tensor(w["v_proj_bias"], device="mps")
                    for arm in ("F", "G"):
                        z = data[arm + "_norm"][i]
                        full = torch.zeros((1, length, 1536), device="mps")
                        full[0, start] = torch.tensor(z, device="mps")
                        value = array(F.linear(full, weight, bias)[0, start]).reshape(2, 128)
                        check(checks, arm + "_public_source_V", maximum(value, data[arm + "_V"][i]))
                        cpu = (z.astype(float) @ w["v_proj_weight"].astype(float).T + w["v_proj_bias"].astype(float)).reshape(2, 128)
                        error = maximum(cpu, data[arm + "_V"][i])
                        raw.append(dict(key=key, layer=i + 1, node=arm + "_cpu64_source_V", error=error,
                                        original_bound=.0001, passes=error <= .0001))
                    if i not in (0, 9, 27):
                        continue
                    pre = f"layer{i + 1}_"
                    q, k, v = (witness[pre + n].astype(float) for n in ("q", "k", "v"))
                    groups, scaling = int(witness[pre + "groups"]), float(witness[pre + "scaling"])
                    raw_qk = (q @ np.repeat(k, groups, axis=0).transpose(0, 2, 1)) * scaling
                    logits = raw_qk + witness[pre + "mask"].astype(float)
                    prob = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                    prob /= np.sum(prob, axis=-1, keepdims=True)
                    check(checks, "cpu64_softmax", maximum(prob, witness[pre + "p"]), .0001)
                    tq = torch.zeros((1, 12, length, 128), device="mps")
                    tq[0, :, start + 1:] = torch.tensor(q.astype(np.float32), device="mps")
                    tk = torch.tensor(np.repeat(k.astype(np.float32), groups, axis=0)[None], device="mps")
                    tm = torch.zeros((1, 1, length, length), device="mps")
                    tm[0, :, start + 1:] = torch.tensor(witness[pre + "mask"], device="mps")
                    tp = torch.softmax((tq @ tk.transpose(2, 3)) * scaling + tm, dim=-1, dtype=torch.float32)
                    check(checks, "public_softmax", maximum(array(tp[0, :, start + 1:]), witness[pre + "p"]))
                    for arm in ("T", "Z", "X"):
                        changed = v.copy()
                        if arm != "T":
                            changed[:, start] = (data["F_V"][i].astype(float) + data["delta_" + arm][i]).astype(np.float32)
                        cpu_y = (prob @ np.repeat(changed, groups, axis=0)).transpose(1, 0, 2)
                        check(checks, "cpu64_PV_" + arm, maximum(cpu_y, data["local_" + arm + "_y"][i]), .0001)
                        tv = torch.tensor(np.repeat(changed.astype(np.float32), groups, axis=0)[None], device="mps")
                        ty = (tp @ tv).transpose(1, 2).contiguous()
                        check(checks, "public_PV_" + arm, maximum(array(ty[0, start + 1:]), data["local_" + arm + "_y"][i]))
                        tw = F.linear(ty.reshape(1, length, 1536), torch.tensor(w["o_proj_weight"], device="mps"))
                        check(checks, "public_output_" + arm, maximum(array(tw[0, start + 1:]), data["local_" + arm + "_write"][i]))
                    counts["complete_attention_witnesses"] += 1
            print("273 audit", sum(r["arm"] == "U" for r in native), "/32", flush=True)
    for name, rows in (("native.csv", native), ("effects.csv", effects), ("source-geometry.csv", source),
                       ("local-vectors.csv", local), ("raw-source-projections.csv", raw), ("heads.csv", heads)):
        table(run / name, rows)
    native_counts = {a: sum(r["passes"] for r in native if r["arm"] == a) for a in ("U", "X", "D", "A", "Z", "R")}
    effect_counts = {a: sum(r["passes"] for r in effects if r["arm"] == a) for a in native_counts}
    strata = {f"{schema}/{rev}": sum(r["passes"] for r in effects if r["arm"] == "X" and r["schema"] == schema and r["reverse"] == rev)
              for schema in ("generic", "descriptive") for rev in (0, 1)}
    save_json(run / "audit.json", dict(status="passed", experiment=273, retrospective=True,
        source_sha256=sha(__file__), design_sha256=sha(run / "design.json"), status_sha256=sha(run / "status.json"),
        counts=dict(counts), numerical_checks=len(checks), numerical_max_error=max(r["error"] for r in checks),
        native_counts=native_counts, effect_counts=effect_counts, X_strata=strata,
        primary_passes=effect_counts["X"] >= 24 and min(strata.values()) >= 6,
        raw_source_checks=dict(total=len(raw), failed=sum(not r["passes"] for r in raw), original_bound=.0001),
        csv_sha256={p.name: sha(p) for p in run.glob("*.csv")},
        boundary="Native source-value interventions with finite source/receiver computation. Old contexts; no pure semantic-schema isolation, fresh forecast, speedup, learned circuit or general Question B solution. No disputed derivatives used or recertified."))
    print(dict(native=native_counts, effects=effect_counts, primary=effect_counts["X"] >= 24 and min(strata.values()) >= 6), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.run.resolve())
    except Exception as exc:
        if args.run.exists() and not (args.run / "failure-audit.json").exists():
            save_json(args.run / "failure-audit.json", dict(error=repr(exc)))
        raise
