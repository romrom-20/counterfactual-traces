"""Stdlib294 record/source/arithmetic audit; omitted vectors are not replicated."""

import json
import math
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
DEST = Path("results/response-change-294")
ARMS = ("FALSE_RANGE", "ENDPOINT_RANGE", "CHANGE_RANGE", "RANDOM_RANGE")


def verify(root):
    dest = root / DEST
    release = read(dest / "release-manifest.json")
    require(
        release["experiment"] == 294
        and release["retrospective"]
        and release["conditional_P_inputs"],
        "Conditional retrospective release",
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
    source_seal(root, d, "change294")
    require(len(d["source_hashes"]) == 112, "112 immutable scientific sources")
    require(
        d["retrospective"]
        and d["conditional_P_inputs"]
        and d["model_forwards"] == d["native_true_calls"] == 0,
        "No new native prediction",
    )
    require(
        d["arms"] == list(ARMS)
        and d["sites"] == [0, 14, 27]
        and d["guards_MiB"] == [512, 384, 192],
        "Frozen arms/sites/storage",
    )
    require(
        d["primary"]
        == dict(
            rank=32,
            power_steps=2,
            norm=0.001,
            improvement_factor=0.5,
            relative=0.25,
            cosine=0.95,
            population=36,
            per_scope=18,
        ),
        "Frozen scientific criteria",
    )
    require(
        d["bounds"]
        == dict(operator=1e-8, range=1e-6, original_range=1e-3, absolute=1e-10, relative=1e-11),
        "Frozen numerical bounds",
    )
    old = root / "results/response-subspace-293"
    for name, h in d["reference_hashes"].items():
        require(sha(old / name) == h, "Parent identity")
    require(d["qualified_reference_sha256"] == sha(old / "audit.json"), "Qualified293c reference")
    require(
        d["parent_entries"] == read(old / "status.json")["cases"]
        and d["cases"] == read(old / "design.json")["cases"]
        and len(d["cases"]) == 8,
        "Exact old cases and parent archives",
    )
    require(
        st["status"] == "complete"
        and st["model_forwards"] == st["native_true_calls"] == a["model_forwards"] == 0
        and a["status"] == "passed"
        and not a["scientific_outcomes_withheld"]
        and not a["general_question_B_solved"],
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
        len(checks) == a["required_checks"] == 1824 and a["failed_checks"] == 0,
        "Numerical denominator",
    )
    expected = {
        (c["key"], str(layer), scope, arm)
        for c in d["cases"]
        for layer in (1, 15, 28)
        for scope in ("source", "verdict")
        for arm in ARMS
    }
    for r in checks:
        require(
            0 <= number(r, "error") <= number(r, "bound") and boolean(r, "passes"),
            "Numerical check",
        )
    require(len({(r["key"], r["layer"], r["name"]) for r in checks}) == 1824, "Unique checks")
    inv = read(dest / "omitted-array-inventory.json")
    require(
        inv["case_bytes"]
        == st["raw_bytes"]
        == sum(e["bytes"] for e in st["cases"])
        <= 192 * 1024**2,
        "Archive budget",
    )
    require([e["key"] for e in st["cases"]] == [c["key"] for c in d["cases"]], "Ordered cases")
    metrics = rows(dest / "metrics.csv")
    index = {(r["key"], r["layer"], r["scope"], r["arm"]): r for r in metrics}
    require(len(metrics) == len(index) == 192 and set(index) == expected, "All48 probes/four arms")
    for e, v in zip(st["cases"], inv["arrays"], strict=True):
        p = dest / "cases" / (e["key"] + ".json")
        m = read(p)
        require(
            sha(p) == e["metadata_sha256"]
            and m["key"] == e["key"]
            and m["arrays_sha256"] == v["sha256"] == e["arrays_sha256"]
            and v["bytes"] == e["bytes"]
            and v["path"] == "cases/" + e["key"] + ".npz",
            "Metadata/archive binding",
        )
        require(len(v["arrays"]) == 57 and len(m["metrics"]) == 24, "Archive/metric coverage")
        for r in m["metrics"]:
            saved = index[r["key"], str(r["layer"]), r["scope"], r["arm"]]
            for k, val in r.items():
                if isinstance(val, bool):
                    require(boolean(saved, k) == val, "Case Boolean metric")
                elif isinstance(val, (int, float)):
                    close(number(saved, k), val, "Case numeric metric")
                else:
                    require(saved[k] == val, "Case metric identity")
    for r in metrics:
        v = {
            k: number(r, k)
            for k in (
                "prediction_squared_norm",
                "target_squared_norm",
                "prediction_target_dot",
                "error_squared_norm",
                "norm",
                "relative",
                "cosine",
                "signed_target_fraction",
            )
        }
        n, rel, _ = vstats(v)
        require(boolean(r, "eligible") == (n >= 0.001), "Norm guard")
        close(
            v["prediction_squared_norm"],
            v["prediction_target_dot"],
            "Orthogonal projection signed identity",
        )
        close(v["signed_target_fraction"], 1 - rel**2, "Projection captured fraction")
        require(all(math.isfinite(x) for x in v.values()), "Finite statistics")
        close(
            v["norm"],
            number(index[r["key"], r["layer"], r["scope"], "CHANGE_RANGE"], "norm"),
            "Same target across bases",
        )
    dec = rows(dest / "decisions.csv")
    require(
        len(dec) == 48 and len({(r["key"], r["layer"], r["scope"]) for r in dec}) == 48,
        "Decision denominator",
    )
    for r in dec:
        key = r["key"], r["layer"], r["scope"]
        f, e, c = (index[*key, arm] for arm in ARMS[:3])
        best = min(number(f, "relative"), number(e, "relative"))
        n, rel, cos = (number(c, k) for k in ("norm", "relative", "cosine"))
        for k, val in [
            ("norm", n),
            ("best_endpoint_error", best),
            ("change_error", rel),
            ("change_cosine", cos),
        ]:
            close(number(r, k), val, "Decision inputs")
        require(
            boolean(r, "improvement") == (n >= 0.001 and rel <= 0.5 * best), "Twofold criterion"
        )
        require(
            boolean(r, "vector_account") == (n >= 0.001 and rel <= 0.25 and cos >= 0.95),
            "Complete vector criterion",
        )
    for field, name in [("improvement", "H1_improvement"), ("vector_account", "H2_vector_account")]:
        scopes = {
            s: sum(boolean(r, field) for r in dec if r["scope"] == s) for s in ("source", "verdict")
        }
        n = sum(scopes.values())
        require(
            a["populations"][name]
            == dict(
                passes=n,
                total=48,
                scopes=scopes,
                population_pass=n >= 36 and min(scopes.values()) >= 18,
            ),
            "Frozen population rule",
        )
    require(
        a["primary_pass"] == a["populations"]["H1_improvement"]["population_pass"],
        "Primary priority",
    )
    cpu = read(dest / "cpu-qualification.json")
    require(
        cpu["passed"]
        and cpu["synthetic"]
        and not cpu["checkpoint_read"]
        and cpu["model_forwards"] == 0
        and cpu["required_checks"] == 152
        and cpu["failed_checks"] == 0
        and cpu["shapes"] == [[17, 23], [19, 29]]
        and cpu["rejected"] == ["17_sign", "17_range", "19_sign", "19_range"]
        and cpu["source_hashes"] == d["source_hashes"]
        and cpu["scientific_outcomes_not_required"],
        "Unequal-shape/negative fixtures",
    )
    cost = read(dest / "cost.json")
    for name, value in dict(
        model_instantiations=0,
        model_forwards=0,
        native_true_calls=0,
        checkpoint_MLP_weight_sets=3,
        conditional_points=48,
        rank=32,
        power_steps=2,
        learned_basis_batches=72,
        learned_basis_row_matrices=144,
        random_basis_batches=24,
        random_basis_row_matrices=48,
        producer_operator_applications=360,
        producer_operator_row_applications=720,
        producer_QR_batches=384,
        producer_QR_row_matrices=768,
        independent_operator_applications=528,
        independent_operator_row_applications=1056,
        independent_QR_batches=384,
        independent_QR_row_matrices=768,
    ).items():
        require(cost[name] == value, "Declared loop-count budget")
    require(
        cost["calculated_from_frozen_loop_counts"]
        and cost["independent_operator_and_QR_checks_additional"]
        and cost["weight_reads_and_host_arithmetic_count"]
        and not cost["speedup_claimed"],
        "Work accounting scope",
    )
    fig = root / "docs/figures/response-change-294"
    prov = read(fig / "provenance.json")
    require(
        prov["all_48_probes"]
        and prov["script_sha256"] == sha(root / "research/boolean_coupling/describe294.py"),
        "Figure provenance",
    )
    for name, h in prov["inputs"].items():
        require(sha(relative(root, name)) == h, "Figure inputs")
    for name, h in prov["outputs"].items():
        require(sha(fig / name) == h, "Figure outputs")
    print(
        json.dumps(
            dict(
                passed=True,
                experiment=294,
                required_checks=len(checks),
                populations=a["populations"],
                retrospective=True,
                conditional_P_inputs=True,
                omitted_arrays_replicated=False,
                general_question_B_solved=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    verify(ROOT)
