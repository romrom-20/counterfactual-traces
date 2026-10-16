"""Run experiment225: separate a product baseline shift from its SiLU interaction.

Example: PYTHONPATH=research python -m boolean_coupling.run --phase qualify --offline
Run the separate audit before proceeding to --phase intervene.
"""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
import platform
from pathlib import Path

import numpy as np
import torch
import transformers

from .core225b import (HERE, SETTINGS, array, capture_mlp, check, continue_product,
                   corner, false_anchor, load_model, observe, sha, write_json)

ROOT = HERE.parents[1]
SOURCES = [HERE / name for name in ("core225b.py", "run225b.py", "audit225b.py", "__init__.py", "cases.json", "settings.json")]
SOURCES += [ROOT / "docs/experiments/225b-baseline-interaction-separation.md", HERE/"core.py", HERE/"run.py", HERE/"audit.py", ROOT/"docs/experiments/225-baseline-interaction-separation.md"]
CORNERS = {"b0": (False, False, "FLL"), "s00": (True, False, "FFL"),
           "b1": (False, True, "FLF"), "s11": (True, True, "FFF")}


def store(output, prefix, values):
    output.update({prefix + key: value for key, value in values.items()})


def expected(phase):
    return (dict(ordinary_models=64, local_mlps=256, continuations=256,
                 remaining_blocks=1408, finite_heads=256, primitive_checks=512)
            if phase == "qualify" else
            dict(ordinary_models=0, local_mlps=0, continuations=448,
                 remaining_blocks=2464, finite_heads=448, primitive_checks=0))


def advance(counts, site):
    counts.update(continuations=1, remaining_blocks=28-site, finite_heads=1)


def save_case(folder, case, arrays, checks, jobs, extra=None):
    path = folder / (case["key"] + ".npz")
    np.savez_compressed(path, **arrays)
    write_json(path.with_suffix(".json"), dict(
        key=case["key"], group=case["group"], position=case["position"],
        arrays_sha256=sha(path), checks=checks, jobs=jobs, **(extra or {})))


def qualify(args):
    out = args.out
    assert not out.exists(), "Use a new output directory; never overwrite evidence."
    cases = json.loads((HERE / "cases.json").read_text())
    assert len(cases) == 32
    out.mkdir(parents=True)
    (out / "qualification").mkdir()
    (out / "interventions").mkdir()
    design = dict(settings=SETTINGS, contexts=[c["key"] for c in cases],
                  source_hashes={str(p.relative_to(ROOT)): sha(p) for p in SOURCES},
                  reference_root=str(args.reference_root) if args.reference_root else None,
                  expected_qualify=expected("qualify"), expected_intervene=expected("intervene"),
                  retrospective=True, primary_site=23, comparison_site=22)
    write_json(out / "design.json", design)
    model = load_model(args.offline)
    write_json(out / "runtime.json", dict(python=platform.python_version(), torch=torch.__version__,
               transformers=transformers.__version__, numpy=np.__version__, device="mps",
               dtype="float32", attention="eager", training=model.training,
               revision=model.config._commit_hash))
    weights = {str(site)+"_down": array(model.model.layers[site-1].mlp.down_proj.weight)
               for site in (22, 23)}
    weights.update(norm=array(model.model.norm.weight),
                   head=array(model.lm_head.weight[[895, 830]]),
                   eps=np.array(model.model.norm.variance_epsilon))
    np.savez_compressed(out / "weights.npz", **weights)
    counts = Counter({key: 0 for key in expected("qualify")})
    for number, case in enumerate(cases, 1):
        start = case["position"] + 1
        different = np.flatnonzero(np.array(case["false_ids"]) != case["true_ids"])
        assert different.tolist() == [start-1]
        assert case["false_ids"][start-1] == 895 and case["true_ids"][start-1] == 830
        checks, arrays, jobs, anchors = [], {}, [], {}
        for bit in (False, True):
            ids = torch.tensor([case["true_ids" if bit else "false_ids"]], device="mps")
            with ExitStack() as stack:
                captures = {s: stack.enter_context(capture_mlp(model.model.layers[s-1])) for s in (22, 23)}
                states, context, scores = observe(model, ids)
            counts["ordinary_models"] += 1
            if not bit:
                for site in (22, 23):
                    anchors[site] = false_anchor(captures[site], start)
                    store(arrays, f"c{site}_", anchors[site])
                arrays["false_final"] = array(states[-1][0, start:])
                arrays["false_scores"] = array(scores)
                false_context = context
            else:
                reference = torch.stack([s[0] for s in states[21:]])
                arrays["native_tail"] = array(reference)
                arrays["native_scores"] = array(scores)
                arrays["mask"] = array(context["mask"])
                arrays["rope_cos"], arrays["rope_sin"] = map(array, context["rope"])
                check(checks, "same_mask", 0. if torch.equal(context["mask"], false_context["mask"]) else float("inf"), 0.)
                for a, b in zip(context["rope"], false_context["rope"]):
                    check(checks, "same_rope", (a-b).abs().max(), 0.)
                for site in (22, 23):
                    arrays[f"mid{site}"] = array(captures[site]["mid"])
                    arrays[f"native_product{site}"] = array(captures[site]["product"])
        legacy = None
        if args.reference_root:
            ep = args.reference_root / "endpoints" / (case["key"] + ".npz")
            hp = args.reference_root / "hybrids" / ep.name
            for p in (ep, hp):
                assert sha(p) == json.loads(p.with_suffix(".json").read_text())["arrays_sha256"]
            legacy = {**dict(np.load(ep)), **dict(np.load(hp))}
        for site in (22, 23):
            mid = torch.tensor(arrays[f"mid{site}"], device="mps")
            for name, (s_finite, p_finite, old_mode) in CORNERS.items():
                product, local = corner(model.model.layers[site-1], mid, anchors[site],
                                        start, s_finite, p_finite, checks)
                counts["local_mlps"] += 1
                counts["primitive_checks"] += 2
                value = continue_product(model, site, mid, product, reference, context, start, checks)
                advance(counts, site)
                check(checks, "isolated_down_replay", np.max(abs(value["write"]-local["original_write"])))
                if name == "s11":
                    check(checks, "native_trajectory", np.max(abs(value["trajectory"]-arrays["native_tail"][site-22:, start:])))
                    check(checks, "native_scores", np.max(abs(value["scores"]-arrays["native_scores"])), .005)
                if legacy is not None:
                    for field, old_field in (("product", "product_after"), ("write", "write"), ("final", "final"), ("scores", "scores")):
                        check(checks, "legacy_"+field, np.max(abs(value[field]-legacy[f"s{site}_{old_mode}_{old_field}"])), .005 if field == "scores" else 1e-4)
                prefix = f"s{site}_{name}_"
                store(arrays, prefix, {**value, **local})
                jobs.append(dict(site=site, name=name, prefix=prefix, silu=s_finite, product=p_finite))
        save_case(out / "qualification", case, arrays, checks, jobs)
        write_json(out / "qualification-status.json", dict(completed=number, counts=dict(counts)))
        print("qualification", number, case["key"], flush=True)
    assert dict(counts) == expected("qualify"), counts
    assert all(sha(ROOT/p) == h for p, h in design["source_hashes"].items())
    write_json(out / "qualification-status.json", dict(status="complete", counts=dict(counts)))


def intervene(args):
    out = args.out
    design = json.loads((out / "design.json").read_text())
    seal = json.loads((out / "qualification-seal.json").read_text())
    assert all(sha(out/p) == h for p, h in seal["files"].items())
    assert all(sha(ROOT/p) == h for p, h in design["source_hashes"].items())
    assert not any((out / "interventions").iterdir())
    model = load_model(args.offline)
    counts = Counter({key: 0 for key in expected("intervene")})
    cases = json.loads((HERE / "cases.json").read_text())
    for number, case in enumerate(cases, 1):
        path = out / "qualification" / (case["key"] + ".npz")
        record = json.loads(path.with_suffix(".json").read_text())
        assert sha(path) == record["arrays_sha256"]
        q = dict(np.load(path))
        checks, arrays, jobs = [], {}, []
        start = case["position"] + 1
        context = dict(mask=torch.tensor(q["mask"], device="mps"),
                       rope=tuple(torch.tensor(q[k], device="mps") for k in ("rope_cos", "rope_sin")))
        reference = torch.tensor(q["native_tail"], device="mps")
        for site in (22, 23):
            product = lambda name: q[f"s{site}_{name}_product"].astype(np.float64)
            p0 = product("b0")
            a, b = product("s00")-p0, product("b1")-p0
            k = product("s11")-product("b1")-product("s00")+p0
            for label, v in (("A", a), ("B", b), ("K", k)):
                arrays[f"s{site}_{label}"] = v
            units = {}
            for label, vector in (("B", b), ("K", k)):
                seed = int(__import__("hashlib").sha256((case["key"]+str(site)+label+"225").encode()).hexdigest()[:16], 16)
                unit = np.random.default_rng(seed).standard_normal(vector.shape)
                unit /= np.linalg.norm(unit)
                units[label] = unit * np.linalg.norm(vector)
                arrays[f"s{site}_random{label}"] = units[label]
                check(checks, "random_product_norm", abs(np.linalg.norm(units[label])-np.linalg.norm(vector)), 1e-9)
            recipes = dict(B_only=p0+b+a, K_only=p0+a+k,
                           random_K_add=p0+a+units["K"],
                           random_K_remove=product("s11")-units["K"],
                           random_B_base=p0+units["B"],
                           random_B_0=p0+a+units["B"],
                           random_B_1=p0+a+k+units["B"])
            mid = torch.tensor(q[f"mid{site}"], device="mps")
            for name, proposed in recipes.items():
                full = torch.tensor(q[f"native_product{site}"], device="mps")
                original = full.clone()
                full[:, start:] = torch.tensor(proposed[None], device="mps", dtype=torch.float32)
                check(checks, "product_prefix", (full[:, :start]-original[:, :start]).abs().max(), 0.)
                value = continue_product(model, site, mid, full, reference, context, start, checks)
                advance(counts, site)
                prefix = f"s{site}_{name}_"
                store(arrays, prefix, value)
                arrays[prefix+"proposed"] = proposed
                jobs.append(dict(site=site, name=name, prefix=prefix))
        save_case(out / "interventions", case, arrays, checks, jobs,
                  dict(qualification_sha256=sha(path)))
        write_json(out / "status.json", dict(completed=number, counts=dict(counts)))
        print("interventions", number, case["key"], flush=True)
    assert dict(counts) == expected("intervene"), counts
    assert all(sha(ROOT/p) == h for p, h in design["source_hashes"].items())
    write_json(out / "status.json", dict(status="complete", counts=dict(counts)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("qualify", "intervene"), required=True)
    parser.add_argument("--out", type=Path, default=ROOT/"runs/baseline-interaction-225b")
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    qualify(args) if args.phase == "qualify" else intervene(args)


if __name__ == "__main__":
    main()
