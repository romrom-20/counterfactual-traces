"""Evaluate272's cached-suffix reference, finite heads, cost and timing controls."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .audit229 import sha
from .cache272 import ROOT, installed_sources, save_json
from .checkpoint238 import locate
from .fields267 import PublicHead


def read(path):
    return json.loads(path.read_text())


def table(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def vector(value, target):
    norm, size = float(np.linalg.norm(target)), float(np.linalg.norm(value))
    return dict(norm=norm, relative=float(np.linalg.norm(value - target)) / max(norm, 1e-30),
                cosine=float(np.vdot(value.ravel(), target.ravel())) / max(norm * size, 1e-30))


def margin(scores):
    return float(scores[1]) - float(scores[0])


def main(run, reference):
    assert not (run / "audit.json").exists() and not (run / "failure-audit.json").exists()
    design, status = read(run / "design.json"), read(run / "status.json")
    assert design["experiment"] == 272 and design["retrospective"] and status["status"] == "complete"
    assert design["installed_sources"] == installed_sources()
    for name, value in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == value
    for name, value in design["reference_hashes"].items():
        assert sha(reference / name) == value
    folder, checkpoint = locate(ROOT)
    assert checkpoint == design["checkpoint"]
    config = read(folder / "config.json")
    h, intermediate, layers = config["hidden_size"], config["intermediate_size"], config["num_hidden_layers"]
    head_width = h // config["num_attention_heads"]
    kv_width = head_width * config["num_key_value_heads"]
    macs_per_row = layers * (2 * h * h + 2 * h * kv_width + 3 * h * intermediate)
    head_macs = h * config["vocab_size"]
    outcomes, coordinates, heads, budgets, timing, checks = [], [], [], [], [], []
    counts = Counter()
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:
        head = PublicHead(handle, config)
        for case, entry in zip(design["cases"], status["cases"], strict=True):
            key, start, length = case["key"], case["position"], len(case["false_ids"])
            tail = length - start
            path = run / "cases" / (key + ".npz")
            assert sha(path) == entry["arrays_sha256"]
            assert sha(path.with_suffix(".json")) == entry["metadata_sha256"]
            meta = read(path.with_suffix(".json"))
            assert all(s == "complete" for s in meta["statuses"].values())
            assert meta["records"]["F"]["prefix_states"] == meta["records"]["T"]["prefix_states"]
            for arm in ("F", "T", "P", "Z"):
                for point, prefix in zip(meta["records"][arm]["attention"], meta["shared_prefix"], strict=True):
                    assert point["prefix_key"] == prefix["key"]
                    assert point["prefix_value"] == prefix["value"]
                    counts["prefix_KV_equalities"] += 2
            counts["prefix_residual_equalities"] += 28
            for arm in ("P", "Z", "S"):
                stop = start + (arm == "S")
                assert meta["records"][arm]["cache_before"] == stop
                assert meta["records"][arm]["cache_after"] == length
                assert all(v["shape"][-2] == stop for v in meta["records"][arm]["copied_prefix"])
            with np.load(path) as data:
                truth = data["T_states"][-1].astype(float) - data["F_states"][-1].astype(float)
                for arm in ("P", "Z", "S"):
                    pred = data[arm + "_states"][-1].astype(float) - data["F_states"][-1].astype(float)
                    row = dict(key=key, **case["group"], arm=arm)
                    ok = []
                    for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                        m = vector(pred[pick], truth[pick])
                        row.update({scope + "_" + k: v for k, v in m.items()})
                        ok.append(m["norm"] >= .001 and m["relative"] <= .05 and m["cosine"] >= .99)
                    row["margin_error"] = abs(margin(data[arm + "_scores"]) - margin(data["T_scores"]))
                    row["passes"] = bool(all(ok) and row["margin_error"] <= .1)
                    outcomes.append(row)
                    target = "T" if arm == "P" else "F"
                    for i in range(28):
                        error = float(np.max(abs(data[arm + "_states"][i].astype(float) - data[target + "_states"][i].astype(float))))
                        coordinates.append(dict(key=key, arm=arm, layer=i + 1, target=target,
                            error=error, original_bound=.0001, passes=error <= .0001))
                for arm in ("F", "Z", "P", "S", "T"):
                    executed_length = length if arm in ("F", "T") else tail - (arm == "S")
                    expected = head(data[arm + "_states"][-1, -1], executed_length)
                    error = float(np.max(abs(expected.astype(float) - data[arm + "_scores"].astype(float))))
                    heads.append(dict(key=key, arm=arm, node="same-shape-public-head", error=error,
                                      original_bound=0., passes=error == 0., bridge_passes=error <= .005))
                for bench in meta["benchmark"]:
                    error = float(np.max(abs(np.array(bench["scores"]) - data[bench["arm"] + "_scores"].astype(float))))
                    heads.append(dict(key=key, arm=bench["arm"], node="unprofiled-benchmark-output", error=error,
                                      original_bound=.0001, passes=error <= .0001, bridge_passes=error <= .005))
            for arm, profile in meta["records"].items():
                m = length if arm in ("F", "T") else tail - (arm == "S")
                expected_dense = macs_per_row * m + head_macs
                expected_pairs = layers * config["num_attention_heads"] * m * length
                expected_attention = 2 * expected_pairs * head_width
                b = meta["budgets"][arm]
                assert b["dense_macs"] == expected_dense
                assert b["dense_calls"] == 7 * layers + 1
                assert b["non_head_rows"] == 7 * layers * m
                assert b["head_macs"] == head_macs
                assert b["attention_dense_pairs"] == expected_pairs and b["attention_macs"] == expected_attention
                counts["independent_profile_formula_checks"] += 6
                assert len(profile["attention"]) == 28
                assert all(p["query_shape"][2] == m and p["key_shape"][2] == length for p in profile["attention"])
            full, partial = meta["budgets"]["T"], meta["budgets"]["P"]
            ratio = partial["dense_macs"] / full["dense_macs"]
            dense_plus_attn = (partial["dense_macs"] + partial["attention_macs"]) / (full["dense_macs"] + full["attention_macs"])
            budgets.append(dict(key=key, **case["group"], length=length, edit=start, suffix=tail,
                prefix_bytes=meta["prefix_bytes"], token_rows_ratio=tail / length,
                dense_macs_full=full["dense_macs"], dense_macs_suffix=partial["dense_macs"],
                dense_ratio=ratio, dense_plus_attention_ratio=dense_plus_attn,
                attention_pair_ratio=partial["attention_dense_pairs"] / full["attention_dense_pairs"],
                full_dense_calls=full["dense_calls"], suffix_dense_calls=partial["dense_calls"],
                dense_budget_passes=ratio <= .05))
            assert meta["prefix_bytes"] == 2 * layers * config["num_key_value_heads"] * start * head_width * 4
            pairs = {}
            for repeat in range(3):
                pair = {r["arm"]: r for r in meta["benchmark"] if r["repetition"] == repeat}
                assert len(pair) == 2 and pair["P"]["clone_prefix_included"] and not pair["T"]["clone_prefix_included"]
                assert pair["P"]["seconds"] > 0 and pair["T"]["seconds"] > 0
                pairs[repeat] = pair
                timing.append(dict(key=key, repetition=repeat, full_seconds=pair["T"]["seconds"],
                    suffix_seconds=pair["P"]["seconds"], speedup=pair["T"]["seconds"] / pair["P"]["seconds"],
                    order="/".join(pair["P"]["order"])))
            checks.append(dict(key=key, all_structural_checks_passed=True, prefix_arrays_immutable=True))
            print("272 audit", len(checks), "/32", flush=True)
    for name, rows in (("native.csv", outcomes), ("raw-coordinates.csv", coordinates),
                       ("raw-heads.csv", heads), ("budget.csv", budgets), ("timing.csv", timing)):
        table(run / name, rows)
    native_counts = {a: sum(r["passes"] for r in outcomes if r["arm"] == a) for a in ("P", "Z", "S")}
    per_case_speedup = [float(np.median([r["full_seconds"] for r in timing if r["key"] == c["key"]])) /
                        float(np.median([r["suffix_seconds"] for r in timing if r["key"] == c["key"]]))
                        for c in design["cases"]]
    save_json(run / "audit.json", dict(status="passed", experiment=272, retrospective=True,
        source_sha256=sha(__file__), design_sha256=sha(run / "design.json"), status_sha256=sha(run / "status.json"),
        checkpoint=checkpoint, checks=checks, counts=dict(counts), native_counts=native_counts,
        primary_prediction_budget_passes=native_counts["P"] == 32 and all(r["dense_budget_passes"] for r in budgets),
        raw_coordinates=dict(total=len(coordinates), failed=sum(not r["passes"] for r in coordinates), original_bound=.0001),
        raw_heads=dict(total=len(heads), failed=sum(not r["passes"] for r in heads),
                       bridge_failed=sum(not r["bridge_passes"] for r in heads)),
        latency=dict(repetitions=3, median_case_speedup=float(np.median(per_case_speedup)),
                     cases_faster=sum(v > 1 for v in per_case_speedup),
                     secondary_material_speedup_passes=float(np.median(per_case_speedup)) >= 1.25 and sum(v > 1 for v in per_case_speedup) >= 24),
        max_dense_ratio=max(r["dense_ratio"] for r in budgets),
        csv_sha256={p.name: sha(p) for p in run.glob("*.csv")},
        boundary="One observed False KV cache, finite recomputation of every affected suffix operation. Changed-shape raw checks preserved. No derivatives, learned circuit, fresh confirmation or full Question B solution."))
    print(dict(native=native_counts, max_dense_ratio=max(r["dense_ratio"] for r in budgets),
               median_speedup=float(np.median(per_case_speedup)), raw_coordinate_failures=sum(not r["passes"] for r in coordinates)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.run.resolve(), args.reference.resolve())
    except Exception as exc:
        if args.run.exists() and not (args.run / "failure-audit.json").exists():
            save_json(args.run / "failure-audit.json", dict(error=repr(exc)))
        raise
