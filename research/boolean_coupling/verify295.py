"""Stdlib295 source/record/arithmetic checks; no omitted-vector reconstruction."""

import json
from pathlib import Path

try:
    from .verify285 import boolean, close, number, read, relative, require, sha
    from .verify286 import source_seal
    from .verify288 import rows
    from .verify293 import vstats
except ImportError:
    from verify285 import boolean, close, number, read, relative, require, sha
    from verify286 import source_seal
    from verify288 import rows
    from verify293 import vstats

ROOT = Path(__file__).resolve().parents[2]
DEST = Path("results/finite-drive-295")
ARMS = (
    "ENDPOINT_DRIVE",
    "MIDPOINT_DRIVE",
    "SIMPSON_DRIVE",
    "INTEGRAL16",
    "SHUFFLED_DRIVE",
    "ORACLE_DIRECTION",
    "CHANGE_RANGE",
)


def verify(root):
    dest = root / DEST
    release = read(dest / "release-manifest.json")
    require(
        release["experiment"] == 295
        and release["retrospective"]
        and release["conditional_P_inputs"],
        "Release identity/scope",
    )
    for name, h in release["files"].items():
        require(sha(relative(root, name)) == h, "Release bytes: " + name)
        require(
            not ({".context", "drafts"} & set(Path(name).parts))
            and Path(name).suffix not in (".npz", ".npy", ".pt", ".safetensors"),
            "Public scope",
        )
    actual = {
        str(p.relative_to(root))
        for p in dest.rglob("*")
        if p.is_file() and p.name != "release-manifest.json"
    }
    require(
        actual == {n for n in release["files"] if n.startswith(str(DEST) + "/")},
        "Exact result inventory",
    )
    d, st, a = (read(dest / (n + ".json")) for n in ("design", "status", "audit"))
    source_seal(root, d, "drive295")
    require(len(d["source_hashes"]) == 117, "117 immutable sources")
    require(
        d["retrospective"]
        and d["conditional_P_inputs"]
        and d["model_forwards"] == d["native_true_calls"] == 0,
        "No new native forecast",
    )
    require(
        d["primary"] == dict(norm=0.001, relative=0.25, cosine=0.95, population=36, per_scope=18),
        "Frozen scientific criteria",
    )
    require(
        d["bounds"]
        == dict(
            absolute=1e-8,
            relative=1e-10,
            bridge_absolute=1e-4,
            bridge_relative=1e-5,
            bridge_norm=0.01,
        ),
        "Frozen bounds",
    )
    require(
        d["arms"] == list(ARMS)
        and d["sites"] == [0, 14, 27]
        and d["guards_MiB"] == [512, 384, 32]
        and "no automatic" in d["stopping_rule"],
        "Arms, sites, guards and stopping rule",
    )
    for name, h in d["reference_hashes"].items():
        require(sha(relative(root, name)) == h, "Parent identity")
    old = root / "results/response-subspace-293"
    prior = root / "results/response-change-294"
    require(
        len(d["cases"]) == 8
        and d["cases"]
        == read(old / "design.json")["cases"]
        == read(prior / "design.json")["cases"],
        "Exact old cases",
    )
    require(
        d["parent_entries"] == read(old / "status.json")["cases"]
        and d["previous_entries"] == read(prior / "status.json")["cases"],
        "Parent archive binding",
    )
    require(d["checkpoint"] == read(prior / "design.json")["checkpoint"], "Pinned checkpoint")
    require(
        st["status"] == "complete"
        and a["status"] == "passed"
        and st["model_forwards"]
        == st["native_true_calls"]
        == a["model_forwards"]
        == a["native_true_calls"]
        == 0
        and not a["scientific_outcomes_withheld"]
        and not a["general_question_B_solved"]
        and not a["learned_circuit_identified"],
        "Qualified scope",
    )
    require(
        a["design_sha256"] == sha(dest / "design.json")
        and a["status_sha256"] == sha(dest / "status.json"),
        "Audit inputs",
    )
    for name, h in a["csv_sha256"].items():
        require(sha(dest / name) == h, "Audit table bytes")
    checks = rows(dest / "checks.csv")
    require(
        len(checks) == a["required_checks"] == 936 and a["failed_checks"] == 0,
        "Required numerical denominator",
    )
    require(
        len({(r["key"], r["layer"], r["name"]) for r in checks}) == 936, "Unique required checks"
    )
    for r in checks:
        require(
            0 <= number(r, "error") <= number(r, "bound") and boolean(r, "passes"),
            "Required numerical decision",
        )
    inv = read(dest / "omitted-array-inventory.json")
    require(
        inv["case_bytes"]
        == st["raw_bytes"]
        == sum(e["bytes"] for e in st["cases"])
        <= 32 * 1024**2,
        "Raw storage budget",
    )
    require(
        [e["key"] for e in st["cases"]] == [c["key"] for c in d["cases"]], "Ordered case coverage"
    )
    metrics = rows(dest / "metrics.csv")
    index = {(r["key"], r["layer"], r["scope"], r["arm"]): r for r in metrics}
    expected = {
        (c["key"], str(layer), scope, arm)
        for c in d["cases"]
        for layer in [1, 15, 28]
        for scope in ["source", "verdict"]
        for arm in ARMS
    }
    require(
        len(metrics) == len(index) == 336 and set(index) == expected,
        "All48 probes/seven candidates",
    )
    for e, v in zip(st["cases"], inv["arrays"], strict=True):
        p = dest / "cases" / (e["key"] + ".json")
        m = read(p)
        require(
            sha(p) == e["metadata_sha256"]
            and m["key"] == e["key"]
            and m["arrays_sha256"] == v["sha256"] == e["arrays_sha256"]
            and v["bytes"] == e["bytes"]
            and v["path"] == "cases/" + e["key"] + ".npz",
            "Archive/metadata binding",
        )
        require(len(m["metrics"]) == 42 and len(v["arrays"]) == 51, "Case array/metric coverage")
        for r in m["metrics"]:
            saved = index[r["key"], str(r["layer"]), r["scope"], r["arm"]]
            for k, val in r.items():
                if isinstance(val, (int, float)):
                    close(number(saved, k), val, "Numeric case metric")
                else:
                    require(saved[k] == val, "Case metric identity")
    stats = {}
    for key, r in index.items():
        v = {
            k: number(r, k)
            for k in [
                "prediction_squared_norm",
                "target_squared_norm",
                "prediction_target_dot",
                "error_squared_norm",
                "norm",
                "relative",
                "cosine",
                "signed_target_fraction",
            ]
        }
        stats[key] = vstats(v)
        close(
            v["norm"],
            number(index[*key[:3], "ENDPOINT_DRIVE"], "norm"),
            "Same target across candidates",
        )
        if r["arm"] in ("ORACLE_DIRECTION", "CHANGE_RANGE"):
            close(
                v["prediction_squared_norm"],
                v["prediction_target_dot"],
                "Orthogonal projection identity",
            )
    priorrows = {
        (r["key"], r["layer"], r["scope"]): r
        for r in rows(prior / "metrics.csv")
        if r["arm"] == "CHANGE_RANGE"
    }
    for key, r in index.items():
        if key[-1] == "CHANGE_RANGE":
            for k in [
                "prediction_squared_norm",
                "target_squared_norm",
                "prediction_target_dot",
                "error_squared_norm",
                "norm",
                "relative",
                "cosine",
                "signed_target_fraction",
            ]:
                close(number(r, k), number(priorrows[key[:3]], k), "Unchanged294 comparator")
        if key[-1] == "SHUFFLED_DRIVE":
            close(
                number(r, "prediction_squared_norm"),
                number(index[*key[:3], "ENDPOINT_DRIVE"], "prediction_squared_norm"),
                "Matched control norm",
            )
    for arm in ARMS:
        scopes = {
            s: sum(
                n >= 0.001 and rel <= 0.25 and cos >= 0.95
                for key, (n, rel, cos) in stats.items()
                if key[-1] == arm and key[2] == s
            )
            for s in ["source", "verdict"]
        }
        n = sum(scopes.values())
        require(
            a["populations"][arm]
            == dict(
                passes=n,
                total=48,
                scopes=scopes,
                population_pass=n >= 36 and min(scopes.values()) >= 18,
            ),
            "Frozen population rule",
        )
    require(
        a["primary_pass"] == a["populations"]["ENDPOINT_DRIVE"]["population_pass"]
        and a["oracle_direction_pass"] == a["populations"]["ORACLE_DIRECTION"]["population_pass"],
        "Primary/oracle distinction",
    )
    cpu = read(dest / "cpu-qualification.json")
    require(
        cpu["passed"]
        and cpu["synthetic"]
        and not cpu["checkpoint_read"]
        and cpu["model_forwards"] == 0
        and cpu["required_checks"] == 90
        and cpu["failed_checks"] == 0
        and cpu["zero_drive"]
        and cpu["shapes"] == [[17, 23], [19, 29]]
        and cpu["rejected"] == ["17_sign", "17_drive", "19_sign", "19_drive"]
        and cpu["source_hashes"] == d["source_hashes"]
        and cpu["scientific_outcomes_not_required"],
        "Final tiny qualification",
    )
    cost = read(dest / "cost.json")
    for k, val in dict(
        model_instantiations=0,
        model_forwards=0,
        native_true_calls=0,
        checkpoint_down_weight_sets=3,
        conditional_points=48,
        quadrature_nodes=[16, 32],
        partition_intervals=12,
        quadrature_product_kernel_batches=24 * 12 * 48,
        producer_down_batches=24 * 7,
        independent_down_batches=24 * 6,
    ).items():
        require(cost[k] == val, "Calculated loop budget")
    require(
        cost["calculated_from_frozen_loop_counts"] and not cost["speedup_claimed"], "Cost scope"
    )
    fig = root / "docs/figures/finite-drive-295"
    prov = read(fig / "provenance.json")
    require(
        prov["all_48_probes"]
        and prov["script_sha256"] == sha(root / "research/boolean_coupling/describe295.py"),
        "Figure provenance",
    )
    for n, h in prov["inputs"].items():
        require(sha(relative(root, n)) == h, "Figure input bytes")
    for n, h in prov["outputs"].items():
        require(sha(fig / n) == h, "Figure output bytes")
    print(
        json.dumps(
            dict(
                passed=True,
                experiment=295,
                required_checks=936,
                populations=a["populations"],
                conditional_P_inputs=True,
                omitted_arrays_replicated=False,
                general_question_B_solved=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    verify(ROOT)
