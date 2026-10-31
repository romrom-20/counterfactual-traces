"""Stdlib source, parent, numerical-scope and conditional-decision integrity."""

import hashlib
import json
import subprocess
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
DEST = Path("results/context-messages-298")
GROUPS = (
    "chosen",
    "swapped",
    "union",
    "complement",
    "neighbor",
    "random1",
    "random2",
    "all",
    "zero",
)


def verify(root):
    dest = root / DEST
    m = read(dest / "release-manifest.json")
    require(m["experiment"] == 298 and m["retrospective"], "Release identity")
    for n, h in m["files"].items():
        require(sha(relative(root, n)) == h, "Release bytes " + n)
        require(
            not ({".context", "drafts"} & set(Path(n).parts))
            and Path(n).suffix not in (".npz", ".npy", ".pt", ".safetensors"),
            "Public scope",
        )
    actual = {
        str(p.relative_to(root))
        for p in dest.rglob("*")
        if p.is_file() and p.name != "release-manifest.json"
    }
    require(
        actual == {n for n in m["files"] if n.startswith(str(DEST) + "/")}, "Exact result inventory"
    )
    d, s, a = [read(dest / n) for n in ["design.json", "status.json", "audit.json"]]
    source_seal(root, d, "messages298")
    require(len(d["source_hashes"]) == 140, "140frozen sources")
    require(
        d["retrospective"] and d["native_inputs_supplied"] and not d["fresh_inputs"],
        "Known native input scope",
    )
    require(
        d["primary"]
        == dict(
            norm=0.001,
            relative=0.25,
            cosine=0.95,
            material=0.02,
            margin_absolute=0.1,
            margin_relative=0.25,
            population=6,
            per_schema=3,
            per_layout=1,
            control_max=2,
        ),
        "Frozen scientific bounds",
    )
    require(
        d["bounds"]
        == dict(
            group_map=1e-8,
            message=1e-4,
            final_max=1e-3,
            final_relative=1e-5,
            head=0.005,
            raw_diagnostic=1e-4,
            random_norm_relative=1e-6,
        ),
        "Declared numerical boundary",
    )
    require(
        a["status"] == "passed"
        and a["failed_checks"] == 0
        and not a["scientific_outcomes_withheld"]
        and a["design_sha256"] == sha(dest / "design.json")
        and a["status_sha256"] == sha(dest / "status.json"),
        "Numerical qualification",
    )
    require(
        s["status"] == "complete"
        and len(d["cases"]) == len(s["cases"]) == 8
        and s["full_transformer_forwards"] == s["prospective_native_forecasts"] == 0,
        "Retrospective case coverage",
    )
    require(
        s["counts"]
        == dict(full_shape_output_projections=16, terminal_MLPs=160, full_vocabulary_heads=160),
        "Actual terminal work",
    )
    cpu = read(dest / "cpu-qualification.json")
    require(
        cpu["passed"]
        and not cpu["checkpoint_loaded"]
        and cpu["required_checks"] == 712
        and cpu["full_tiny_forwards"] == 36
        and cpu["terminal_MLPs"] == 101
        and cpu["failed_checks"] == 0
        and len(cpu["rejected"]) == 4
        and cpu["source_hashes"] == d["source_hashes"],
        "Tiny CPU qualification",
    )
    require(sha(dest / "cpu-qualification.json") == d["cpu_qualification_sha256"], "CPU binding")
    parent = read(root / "research/boolean_coupling/parents298.json")
    require(
        sha(root / "research/boolean_coupling/parents298.json") == d["parent_source_sha256"],
        "Parent definition binding",
    )
    require(d["cases"] == parent["cases"], "Immutable cases and positions")
    old = root / "results/query-routes-297"
    require(
        parent["source_design_sha256"] == sha(old / "pre-native/design.json")
        and parent["source_native_status_sha256"] == sha(old / "native-phase/native-status.json")
        and parent["source_native_audit_sha256"] == sha(old / "native-phase/audit.json"),
        "Original297 parent seals",
    )
    original = {e["key"]: e for e in read(old / "native-phase/native-status.json")["cases"]}
    raw_inventory = read(dest / "omitted-array-inventory.json")["arrays"]
    require(
        sum(e["bytes"] for e in raw_inventory) == s["raw_bytes"] <= 64 * 1024**2,
        "Bounded new raw data",
    )
    for c, e, inv in zip(d["cases"], s["cases"], raw_inventory, strict=True):
        key = c["key"]
        require(key == e["key"], "Ordered case binding")
        case = read(dest / "cases" / (key + ".json"))
        require(
            sha(dest / "cases" / (key + ".json")) == e["metadata_sha256"]
            and case["arrays_sha256"] == e["arrays_sha256"] == inv["sha256"]
            and e["bytes"] == inv["bytes"],
            "Array/metadata binding",
        )
        require(
            case["complete"]
            and all(r["passes"] and r["error"] <= r["bound"] for r in case["checks"]),
            "Installation/native/zero/calibration checks",
        )
        require(
            c["parent_hashes"]
            == dict(
                native=original[key]["arrays_sha256"],
                witness=original[key]["witness_sha256"],
                metadata=original[key]["metadata_sha256"],
            ),
            "Native/witness provenance",
        )
        ids = c["ids"]
        src = c["position"]
        ends = [j for j, x in enumerate(ids[:src]) if x == parent["eom_id"]]
        ib = [j for j in range(ends[0]) if ids[j] in parent["boolean_ids"]]
        chosen, swapped = (ends, ib) if c["group"]["schema"] == "generic" else (ib, ends)
        union = set(ends) | set(ib)
        neighbors = []
        for j in chosen:
            k = j - 1
            while k in union or k in neighbors:
                k -= 1
            require(k >= 0, "Valid matched neighbor")
            neighbors.append(k)
        require(
            c["positions"]
            == dict(
                chosen=chosen,
                swapped=swapped,
                union=sorted(union),
                complement=[j for j in range(len(ids)) if j not in chosen],
                neighbor=sorted(neighbors),
                all=list(range(len(ids))),
                zero=[],
            ),
            "Independent token role/neighbor map",
        )
    required = rows(dest / "checks.csv")
    raw = rows(dest / "raw-coordinate.csv")
    require(
        len(required) == a["required_checks"] == 1424
        and all(
            boolean(r, "passes") and number(r, "error") <= number(r, "bound") for r in required
        ),
        "Required numerical decisions",
    )
    require(
        len(raw) == a["raw_coordinate_checks"] == 160
        and sum(not boolean(r, "passes") for r in raw) == a["raw_coordinate_failed"] == 160,
        "Original narrow failures retained",
    )
    for r in raw:
        require(
            boolean(r, "passes") == (number(r, "error") <= 1e-4) and number(r, "bound") == 1e-4,
            "Raw diagnostic criterion unchanged",
        )
    for n, h in a["csv_sha256"].items():
        require(sha(dest / n) == h, "Audit table binding")
    review = read(dest / "numerical-review.json")
    require(
        review["posthoc"]
        and review["passed"]
        and review["original_diagnostics_unchanged"]
        and review["operation_decisions"] == 144
        and review["joint_decisions"] == 72
        and review["required_checks"] == 720
        and review["failed_checks"] == 0,
        "Explicit posthoc numerical review",
    )
    require(
        review["audit_sha256"] == sha(dest / "audit.json")
        and review["design_sha256"] == sha(dest / "design.json"),
        "Review target binding",
    )
    for n, h in review["source_hashes"].items():
        require(
            sha(root / n) == h
            and hashlib.sha256(
                subprocess.check_output(
                    ["git", "show", review["source_commit"] + ":" + n], cwd=root
                )
            ).hexdigest()
            == h,
            "Frozen review source",
        )
    require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", review["source_commit"], "HEAD"],
            cwd=root,
            capture_output=True,
        ).returncode
        == 0,
        "Review ancestry",
    )
    require(
        all(r["passes"] and r["error"] <= r["bound"] for r in review["checks"])
        and review["max_scaled_contrast_error"] <= 1e-3,
        "Conditional numerical stability",
    )
    stats = read(dest / "vectors.json")["statistics"]
    index = {(r["key"], r["group"], r["operation"], r["scope"]): r for r in stats}
    require(len(stats) == len(index) == 288, "All conditional vector records")
    for r in stats:
        vstats(r)
    for r in read(dest / "message-records.json")["statistics"]:
        vstats(r)
    scores = {r["key"]: r["scores"] for r in read(dest / "scores.json")["cases"]}
    effects = rows(dest / "effects.csv")
    joints = rows(dest / "joint.csv")
    scalar = rows(dest / "scalar-component.csv")
    require(
        len(effects) == 144 and len(joints) == len(scalar) == 72, "All group/operation decisions"
    )
    op = {}
    for r in effects:
        key, g, o = r["key"], r["group"], r["operation"]
        ss = scores[key]

        def margin(name):
            return ss[name + "_scores"][1] - ss[name + "_scores"][0]

        dm = margin("native_reference") - margin("frozen_query_reference")
        effect = (
            margin(g + "_add") - margin("frozen_query_reference")
            if o == "add"
            else margin("native_reference") - margin(g + "_remove")
        )
        error = abs(effect - dm)
        bound = min(0.1, 0.25 * abs(dm))
        ok = abs(dm) >= 0.02 and error <= bound
        for field, val in [
            ("native_margin_effect", dm),
            ("candidate_margin_effect", effect),
            ("margin_error", error),
            ("margin_bound", bound),
        ]:
            close(number(r, field), val, "Finite signed margin " + field)
        for scope in ["receiving", "verdict"]:
            norm, rel, cos = vstats(index[key, g, o, scope])
            ok = ok and norm >= 0.001 and rel <= 0.25 and cos >= 0.95
            close(number(r, scope + "_relative"), rel, "Vector criterion")
        require(boolean(r, "passes") == ok, "Operation decision")
        op[key, g, o] = r
    for r in joints:
        key, g = r["key"], r["group"]
        rr = [op[key, g, o] for o in ["add", "remove"]]
        require(
            boolean(r, "passes") == all(boolean(x, "passes") for x in rr)
            and boolean(r, "eligible_margin")
            == (abs(number(rr[0], "native_margin_effect")) >= 0.02),
            "Joint case decision",
        )
    for r in scalar:
        require(
            boolean(r, "passes")
            == all(
                abs(number(op[r["key"], r["group"], o], "native_margin_effect")) >= 0.02
                and number(op[r["key"], r["group"], o], "margin_error")
                <= number(op[r["key"], r["group"], o], "margin_bound")
                for o in ["add", "remove"]
            ),
            "Scalar component remains separate",
        )
    pop = {}
    for g in GROUPS:
        rr = [r for r in joints if r["group"] == g]
        schemas = {
            s: sum(boolean(r, "passes") for r in rr if r["schema"] == s)
            for s in ["descriptive", "generic"]
        }
        layouts = {
            s: sum(boolean(r, "passes") for r in rr if r["layout"] == s)
            for s in ["ini", "jsonlines", "memo", "sexpr"]
        }
        n = sum(schemas.values())
        pop[g] = dict(
            passes=n,
            total=8,
            schemas=schemas,
            layouts=layouts,
            sufficient=n >= 6 and min(schemas.values()) >= 3 and min(layouts.values()) >= 1,
        )
    require(
        pop == a["populations"]
        and a["H1"]
        == (
            pop["chosen"]["sufficient"]
            and all(
                pop[g]["passes"] <= 2
                for g in ["swapped", "complement", "neighbor", "random1", "random2"]
            )
        ),
        "Frozen population hypothesis",
    )
    require(
        a["eligible_margin_cases"]
        == sum(boolean(r, "eligible_margin") for r in joints if r["group"] == "chosen"),
        "Full materiality denominator",
    )
    cost = read(dest / "cost.json")
    require(
        cost["counts"] == s["counts"]
        and cost["checkpoint_full_transformer_forwards"]
        == cost["new_false_or_true_full_passes"]
        == 0
        and cost["raw_bytes"] == s["raw_bytes"],
        "Work accounting",
    )
    prov = read(root / "docs/figures/context-messages-298/provenance.json")
    require(
        prov["all_eight_cases"]
        and prov["both_directions"]
        and prov["script_sha256"] == sha(root / "research/boolean_coupling/describe298.py"),
        "Figure scope",
    )
    for n, h in prov["inputs"].items():
        require(sha(root / n) == h, "Figure input")
    for n, h in prov["outputs"].items():
        require(sha(root / "docs/figures/context-messages-298" / n) == h, "Figure output")
    print(
        json.dumps(
            dict(
                passed=True,
                experiment=298,
                retrospective=True,
                H1=a["H1"],
                eligible_cases=a["eligible_margin_cases"],
                populations=pop,
                required_checks=a["required_checks"],
                raw_failures_retained=a["raw_coordinate_failed"],
                posthoc_review_passed=True,
                omitted_arrays_replicated=False,
                general_question_B_solved=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    verify(ROOT)
