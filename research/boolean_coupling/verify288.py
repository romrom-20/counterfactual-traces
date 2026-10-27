"""Stdlib checks of 288 records and signed ledgers; no omitted-array replication."""

import csv
import itertools
import json
from pathlib import Path

try:
    from .verify285 import boolean, close, number, read, relative, require, sha
    from .verify286 import margin, source_seal
    from .verify287 import vector
except ImportError:
    from verify285 import boolean, close, number, read, relative, require, sha
    from verify286 import margin, source_seal
    from verify287 import vector

ROOT = Path(__file__).resolve().parents[2]
DEST = Path("results/tangent-routing-288")
ARMS = tuple("".join(b) for b in itertools.product("01", repeat=3))
HYPOTHESES = dict(SOURCE="100", RECEIVING_AND_HEAD="011", RECEIVING_BLOCKS="010", HEAD="001", ADD="ADD")


def rows(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def verify(root):
    dest = root / DEST
    release = read(dest / "release-manifest.json")
    require(release["experiment"] == 288 and release["retrospective"], "Release identity")
    for name, h in release["files"].items():
        require(sha(relative(root, name)) == h, "Release bytes: " + name)
        require(not ({".context", "drafts"} & set(Path(name).parts))
                and Path(name).suffix not in (".npz", ".npy", ".pt", ".safetensors"), "Public scope")
    actual = {str(p.relative_to(root)) for p in dest.rglob("*") if p.is_file() and p.name != "release-manifest.json"}
    require(actual == {n for n in release["files"] if n.startswith(str(DEST) + "/")}, "Exact inventory")
    d, status, audit = (read(dest / (n + ".json")) for n in ("design", "status", "audit"))
    require(d["experiment"] == 288 and d["retrospective"] and not d["synthetic"], "Production design")
    source_seal(root, d, "routing288")
    require(len(d["source_hashes"]) == 78 and release["scientific_sources"] == d["source_hashes"], "Scientific coverage")
    require(d["arms"] == list(ARMS) and d["FD_steps"] == [0.05, 0.025], "Eight corners/steps")
    require(d["resource_guards_MiB"] == [640, 384, 128], "Storage guards")
    require(d["factors"] == ["source_blocks", "receiving_blocks", "final_head"], "Factor order")
    require(d["anchors"] == ["False", "source-only-MF"], "Anchor scope")
    require(d["primary"] == dict(target="g111-g000 and finite E111-E000",
            hypotheses={"SOURCE": "100", "RECEIVING_AND_HEAD": "011"}, vector_norm_min=0.001,
            relative_max=0.25, cosine_min=0.95, finite_error_absolute=0.005,
            finite_error_relative=0.25, finite_material_min=0.02, population_min=24, per_layout_min=6), "Frozen endpoints")
    require(d["numerical_bounds"] == dict(boundary_adjoint=1e-4, matched_old_cached_gradient=1e-4,
            matched_old_full_gradient=1e-4, old_cached_primal=1e-4, native_finite_coordinate=0.002,
            native_finite_score=0.001, CPU64_head=0.005, projected_AD_FD_absolute=0.001,
            projected_AD_FD_relative=0.02, FD_halving_absolute=0.002, FD_halving_relative=0.03), "Frozen numeric bounds")
    require(status["status"] == "complete" and audit["status"] == "passed" and audit["failed_checks"] == 0, "Audit passed")
    require(audit["design_sha256"] == sha(dest / "design.json") and audit["status_sha256"] == sha(dest / "status.json"), "Audit bindings")
    require(audit["audit_source_sha256"] == sha(root / "research/boolean_coupling/audit288.py"), "Auditor bytes")
    require(not audit["general_question_B_solved"] and audit["old_287_failed_gate_preserved"], "Claim boundary")
    previous = root / "results/source-receiver-readout-287"
    old, qualifier = read(previous / "audit.json"), read(previous / "boundary-qualification/audit.json")
    require(old["status"] == "failed" and old["failed_checks"] == 1, "Original 287 failure retained")
    require(qualifier["status"] == "qualified_boundary_only" and not qualifier["original_numeric_gate_passed"], "287b narrowed scope")
    old285 = root / "results/native-structure-285"
    for name, phase in (("design", "pre-native"), ("forecast-status", "pre-native"), ("native-status", "native-phase")):
        require(d["reference_hashes"]["285_" + name] == sha(old285 / phase / (name + ".json")), "285 phase reference " + name)
    for tag, directory, names in (
        ("287", previous, ("design", "status", "audit", "head")),
        ("287b", previous / "boundary-qualification", ("design", "audit")),
    ):
        for name in names:
            require(d["reference_hashes"][tag + "_" + name] == sha(directory / (name + ".json")), "Reference " + tag + name)
    for name, h in audit["csv_sha256"].items():
        require(sha(dest / name) == h, "Audited table " + name)
    cases = {c["key"]: c for c in d["cases"]}
    require(len(cases) == len(status["cases"]) == 32, "Cohort denominator")
    require(list(cases) == [e["key"] for e in status["cases"]], "Case order")
    inventory = read(dest / "omitted-array-inventory.json")
    require(inventory["case_bytes"] == status["raw_bytes"] == sum(e["bytes"] for e in status["cases"]) <= 128 * 1024**2, "Array budget")
    scores = {e["key"]: e["scores"] for e in read(dest / "scores.json")["cases"]}
    checks = rows(dest / "checks.csv")
    derivatives = rows(dest / "derivatives.csv")
    require(audit["required_checks"] == len(checks) + len(derivatives) == 4384, "Required coverage")
    require(len(checks) == 2080 and len(derivatives) == 2304, "Numeric denominator")
    for row in checks + derivatives:
        require(0 <= number(row, "error") <= number(row, "bound") and boolean(row, "passes"), "Required numeric decision")
    index = {(r["key"], r["arm"], r["direction"], r["step"]): r for r in derivatives}
    for key, entry, item in zip(cases, status["cases"], inventory["arrays"], strict=True):
        require(item["path"] == "cases/" + key + ".npz" and item["sha256"] == entry["arrays_sha256"] and item["bytes"] == entry["bytes"], "Omitted array binding")
        path = dest / "cases" / (key + ".json")
        require(sha(path) == entry["metadata_sha256"], "Metadata hash")
        meta = read(path)
        require(meta["key"] == key and meta["arrays_sha256"] == entry["arrays_sha256"], "Metadata identity")
        require(meta["prefix_unchanged"] and meta["first_source_policy"]["False_primal_exact"], "Source/prefix witnesses")
        require(meta["first_source_policy"]["activation_calls"] == 1, "Pre-SiLU source clamp")
        require(meta["counts"] == dict(full_F=1, first_block=1, anchor_suffixes=2, anchor_blocks=54,
                modified_forwards=112, modified_blocks=4536, modified_heads=112, tape_backwards=8,
                triangular_blocks=324, triangular_block_VJPs=324, triangular_head_VJPs=8,
                native_suffixes=2, native_blocks=54), "All call counts")
        for c in meta["checks"]:
            rr = [r for r in checks if r["key"] == key and r["name"] == c["name"]]
            require(len(rr) == 1 and number(rr[0], "error") == c["error"] and number(rr[0], "bound") == c["bound"], "Numeric metadata/table identity")
        require(len(meta["derivatives"]) == 72, "Per-case FD coverage")
        for row in meta["derivatives"]:
            arm, label, step, ad = row["arm"], row["direction"], row["step"], row["AD"]
            published = index[key, arm, label, str(step)]
            close(number(published, "AD"), ad, "AD binding")
            if step == "halving":
                error = abs(number(index[key, arm, label, "0.05"], "FD") - number(index[key, arm, label, "0.025"], "FD"))
                bound = 0.002 + 0.03 * abs(ad)
            else:
                fd = (margin(row["plus_scores"]) - margin(row["minus_scores"])) / (2 * step)
                close(fd, row["FD"], "Finite score derivative")
                error, bound = abs(ad - fd), 0.001 + 0.02 * abs(ad)
            close(number(published, "error"), error, "FD error")
            close(number(published, "bound"), bound, "FD bound")
        sc = scores[key]
        for arm in ARMS:
            lift = [a + b - c for a, b, c in zip(sc["anchor1_scores"], sc[arm + "_finite_raw_scores"], sc[f"anchor{arm[-1]}_scores"], strict=True)]
            require(lift == sc[arm + "_finite_lifted_scores"], "Exact score lifting")
    totals = {n: sum(read(dest / "cases" / (k + ".json"))["counts"][n] for k in cases) for n in status["counts"]}
    require(totals == status["counts"], "Aggregate cost")
    traces, raw = rows(dest / "intermediate_diagnostics.csv"), rows(dest / "native_coordinate_diagnostics.csv")
    require(len(traces) == 32 * 8 * 27 and len(raw) == 128, "Diagnostic coverage")
    require(sum(not boolean(r, "passes") for r in traces) == audit["intermediate_diagnostic_failures"], "Trace failures retained")
    require(sum(not boolean(r, "passes") for r in raw) == audit["original_coordinate_diagnostic_failures"], "Raw failures retained")
    for r in traces:
        require(boolean(r, "passes") == (number(r, "error") <= 1e-4), "Trace diagnostic threshold")
    for r in raw:
        require(number(r, "original_bound") == 1e-4 and boolean(r, "passes") == (number(r, "error") <= 1e-4), "Original coordinate threshold")
    hypotheses, effects, ledger, native = (rows(dest / (n + ".csv")) for n in ("hypotheses", "effects", "ledger", "native_scalar"))
    require((len(hypotheses), len(effects), len(ledger), len(native)) == (160, 256, 224, 256), "Scientific coverage")
    require({(r["key"], r["hypothesis"]) for r in hypotheses} == set(itertools.product(cases, HYPOTHESES)), "Every hypothesis/case")
    require({(r["key"], r["arm"]) for r in effects} == {(r["key"], r["arm"]) for r in native} == set(itertools.product(cases, ARMS)), "Every arm/case")
    require({(r["key"], r["term"]) for r in ledger} == set(itertools.product(cases, ARMS[1:])), "Every interaction/case")
    ei = {(r["key"], r["arm"]): r for r in effects}
    for key, sc in scores.items():
        for arm in ARMS:
            e = ei[key, arm]
            close(number(e, "effect"), margin(sc[arm + "_finite_lifted_scores"]) - margin(sc["anchor1_scores"]), "Finite effect")
            close(number(e, "local_linear_error"), abs(number(e, "effect") - number(e, "dot_delta")), "Local scalar remainder")
    for row in hypotheses:
        key, name = row["key"], row["hypothesis"]
        norm, rel, cos = vector(row)
        target = number(ei[key, "111"], "effect") - number(ei[key, "000"], "effect")
        arm = HYPOTHESES[name]
        pred = (number(ei[key, arm], "effect") - number(ei[key, "000"], "effect") if arm != "ADD"
                else number(ei[key, "100"], "effect") + number(ei[key, "011"], "effect") - 2 * number(ei[key, "000"], "effect"))
        err = abs(pred - target)
        for n, v in (("finite_target", target), ("finite_prediction", pred), ("finite_error", err)):
            close(number(row, n), v, "Signed finite contrast")
        vp, fp, material = norm >= .001 and rel <= .25 and cos >= .95, err <= .005 + .25 * abs(target), abs(target) >= .02
        for n, v in (("vector_pass", vp), ("finite_pass", fp), ("passes", vp and fp), ("material_eligible", material), ("material_pass", vp and fp and material)):
            require(boolean(row, n) == v, "Scientific decision " + n)
    for name in HYPOTHESES:
        rr = [r for r in hypotheses if r["hypothesis"] == name]
        layout = {b: sum(boolean(r, "passes") for r in rr if r["block"] == b) for b in ("table", "dialogue", "xml", "user_rule")}
        count = sum(boolean(r, "passes") for r in rr)
        require(audit["hypotheses"][name] == dict(passes=count, total=32, by_layout=layout,
                population_pass=count >= 24 and all(v >= 6 for v in layout.values()),
                material_passes=sum(boolean(r, "material_pass") for r in rr), material_eligible=sum(boolean(r, "material_eligible") for r in rr)), "Population decision " + name)
    for row in ledger:
        key, term = row["key"], row["term"]
        subset = [j for j, bit in enumerate(term) if bit == "1"]
        fe, scalar = 0.0, 0.0
        for bits in itertools.product((0, 1), repeat=len(subset)):
            corner = ["0"] * 3
            for j, b in zip(subset, bits, strict=True):
                corner[j] = str(b)
            corner = "".join(corner)
            sign = (-1) ** (len(subset) - sum(bits))
            fe += sign * number(ei[key, corner], "effect")
            scalar += sign * number(ei[key, corner], "dot_delta")
        close(number(row, "finite_contribution"), fe, "Finite Mobius ledger")
        close(number(row, "projection_on_delta"), scalar, "Signed gradient projection ledger")
    old_targets = {r["key"]: number(r, "native_target") for r in rows(previous / "boundary-qualification/readouts.csv")}
    for row in native:
        key, arm = row["key"], row["arm"]
        y, p = old_targets[key], number(ei[key, arm], "dot_delta")
        close(number(row, "target_effect"), y, "Old native target")
        close(number(row, "prediction"), p, "Native scalar prediction")
        close(number(row, "error"), abs(y - p), "Native scalar error")
        require(boolean(row, "passes") == (abs(y - p) <= .1), "Native scalar decision")
    cpu = read(dest / "cpu-qualification/preflight-summary.json")
    cdesign = read(dest / "cpu-qualification/complete/design.json")
    caudit = read(dest / "cpu-qualification/complete/audit.json")
    require(cdesign["source_hashes"] == d["source_hashes"], "Final CPU-qualified scientific sources")
    require(caudit["status"] == "passed" and caudit["required_checks"] == 548, "CPU saved-array gate")
    require(cpu["passed"] and cpu["synthetic"] and not cpu["checkpoint_loaded"] and cpu["all_coordinate_FD_checks"] == 512, "CPU qualification")
    require(cpu["tamper_rejected"] == ["bounds", "FD", "required"], "Negative fixtures")
    allcoord = read(dest / "cpu-qualification/all-coordinates.json")
    require(len(allcoord["checks"]) == 512, "All tiny coordinates")
    for row in allcoord["checks"]:
        require(all(abs(row["AD"] - f) <= .001 + .02 * abs(row["AD"]) for f in row["FD"]), "Tiny coordinate FD")
        require(row["halving"] == abs(row["FD"][0] - row["FD"][1]) <= .002 + .03 * abs(row["AD"]), "Tiny FD halving")
    figure = root / "docs/figures/tangent-routing-288"
    provenance = read(figure / "provenance.json")
    for name, h in provenance["inputs"].items():
        require(sha(root / name) == h, "Figure input")
    for name, h in provenance["outputs"].items():
        require(sha(figure / name) == h, "Figure output")
    require(provenance["all_32_cases"] and provenance["script_sha256"] == sha(root / "research/boolean_coupling/describe288.py"), "Figure provenance")
    print(json.dumps(dict(passed=True, experiment=288, retrospective=True,
                          required_checks=audit["required_checks"], hypotheses=audit["hypotheses"],
                          omitted_vectors_independently_verified=False, general_question_B_solved=False), indent=2))


if __name__ == "__main__":
    verify(ROOT)
