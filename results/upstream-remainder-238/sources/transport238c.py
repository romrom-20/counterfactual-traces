"""Native local-remainder transport beforeMLP20; sealed diagnosis then targeted repair."""

import argparse
import csv
import hashlib
import json
import platform
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers

from .alignment228 import mids, source_block
from .checkpoint238 import locate
from .core225b import array, check, load_model, observe, sha
from .ledger235 import states
from .smooth228 import attention_map, mlp_map, promote, qualified_jvp
from .tail226b import block

ROOT = Path(__file__).resolve().parents[2]
NODES = tuple(range(2, 39))
SOURCES = [ROOT/p for p in ("research/boolean_coupling/transport238.py", "research/boolean_coupling/transport238b.py", "research/boolean_coupling/transport238c.py", "research/boolean_coupling/audit238.py",
    "research/boolean_coupling/checkpoint238.py", "research/boolean_coupling/smooth228.py",
    "research/boolean_coupling/core225b.py", "research/boolean_coupling/alignment228.py",
    "research/boolean_coupling/ledger235.py", "research/boolean_coupling/tail226b.py",
    "research/boolean_coupling/audit229.py", "research/boolean_coupling/audit235.py",
    "research/boolean_coupling/audit236.py", "research/boolean_coupling/cases228.json",
    "research/boolean_coupling/settings.json", "docs/experiments/238-upstream-remainder-transport.md", "docs/experiments/238b-bootstrap-correction.md", "docs/experiments/238c-bootstrap-schema.md")]
ARMS = ("base", "oracle4", "own4", "random4", "native")
PROGRESS = {}


def spec(n):
    return n//2, "attention" if n % 2 == 0 else "mlp"


def rule(f, i, kind):
    if kind == "attention":
        return f["false_boundaries"][i].astype(float), f["false_mids"][i].astype(float)+f["source_mid"][i]
    return f["false_mids"][i].astype(float)+f["source_mid"][i], f["false_boundaries"][i+1].astype(float)+f["source_out"][i]


def jacobian(model, boundaries, middle, context, start, n):
    i, kind = spec(n)
    if kind == "attention":
        return attention_map(model.model.layers[i], boundaries[i], context, start), promote(boundaries[i][:, start:])
    return mlp_map(model.model.layers[i]), promote(middle[i][:, start:])


def qj(function, anchor, direction, label, data, witnesses, checks, counts):
    _, value, plus, minus, error = qualified_jvp(function, anchor, direction)
    check(checks, "used_JVP_FD", error)
    data[label+"_direction"], data[label+"_tangent"] = array(direction[0]), array(value[0])
    data[label+"_fd"] = array(((plus-minus)/2e-5)[0])
    witnesses.append(dict(label=label, plus_sha256=hashlib.sha256(array(plus).tobytes()).hexdigest(),
        minus_sha256=hashlib.sha256(array(minus).tobytes()).hexdigest(), error=error, bound=1e-4))
    counts["jvps"] += 1
    counts["fd_endpoints"] += 2
    return value


def saved(out, folder, case, data, checks, witnesses, **extra):
    assert shutil.disk_usage(out).free > 1024**3, "Preserve evidence; less than1GiB free."
    path = out/folder/(case["key"]+".npz")
    assert not path.exists()
    np.savez_compressed(path, **data)
    meta = dict(key=case["key"], group=case["group"], start=case["position"]+1,
        arrays_sha256=sha(path), checks=checks, fd_witnesses=witnesses, **extra)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
    return dict(key=case["key"], arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json")))


def ledger(previous, out):
    assert not out.exists(), "Never overwrite evidence."
    prior = json.loads((previous/"design.json").read_text())
    old = Path(prior["input228"])
    old_audit = json.loads((old/"audit.json").read_text())
    assert old_audit["status"] == "complete" and old_audit["global_forecast_order_passed"]
    assert set(old_audit["checks"]) == set(old_audit["maximum_errors"])
    assert all(v == 32 for v in old_audit["checks"].values())
    assert all(np.isfinite(v) and 0 <= v <= (.005 if k == "native_head" else 1e-4)
        for k, v in old_audit["maximum_errors"].items())
    for name, h in prior["input_hashes"].items():
        assert sha(old/name) == h
    out.mkdir(parents=True)
    for name in ("sources", "ledger", "cases"):
        (out/name).mkdir()
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in SOURCES}
    for p in SOURCES:
        (out/"sources"/p.name).write_bytes(p.read_bytes())
    _, checkpoint = locate(ROOT)
    (out/"design.json").write_text(json.dumps(dict(experiment="238c", retrospective=True, source_hashes=hashes,
        input237=str(previous), input228=str(old), input_hashes=prior["input_hashes"],
        checkpoint=checkpoint, original_derivative_recertified=False), indent=2)+"\n")
    model = load_model(True)
    (out/"runtime.json").write_text(json.dumps(dict(python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, transformers=transformers.__version__, device="mps", public_dtype="float32",
        derivative_dtype="cpu_float64", revision=model.config._commit_hash), indent=2)+"\n")
    counts, entries, rows = Counter(), [], []
    cases = json.loads((ROOT/"research/boolean_coupling/cases228.json").read_text())
    for number, case in enumerate(cases, 1):
        key, start = case["key"], case["position"]+1
        f, o = (dict(np.load(old/phase/(key+".npz"))) for phase in ("forecasts", "observations"))
        checks, data, witnesses = [], {}, []
        PROGRESS.update(key=key, checks=checks, data=data, fd_witnesses=witnesses)
        with mids(model) as middle:
            false_states, context, _ = observe(model, torch.tensor([case["false_ids"]], device="mps"))
        counts["ordinary_false"] += 1
        boundaries = [None, *false_states]
        data.update(mask=array(context["mask"]), rope_cos=array(context["rope"][0]), rope_sin=array(context["rope"][1]))
        check(checks, "exact_initial_receiving_state", np.max(abs(o["true_states"][0].astype(float)
            -f["false_boundaries"][1].astype(float)-f["F_delta_out"][0])), 0.)
        for n in NODES:
            i, kind = spec(n)
            ni, no, pi, po = states(f, o, i, kind)
            e, following = ni-pi, no-po
            function, anchor = jacobian(model, boundaries, middle, context, start, n)
            target = f["false_mids"][i] if kind == "attention" else f["false_boundaries"][i+1]
            check(checks, "false_primal_bridge", np.max(abs(array(function(anchor)[0])-target)))
            if kind == "attention":
                data[f"a{i}_false"] = array(boundaries[i][0])
                check(checks, "false_replay", np.max(abs(data[f"a{i}_false"][start:]-f["false_boundaries"][i])), 0.)
            tangent = qj(function, anchor, torch.tensor(e[None]), f"n{n}", data, witnesses, checks, counts)
            data[f"n{n}_remainder"] = following-array(tangent[0])
            check(checks, "local_error_recurrence", np.max(abs(array(tangent[0])+data[f"n{n}_remainder"]-following)), 1e-9)
            del function
        target = o["true_mids"][19].astype(float)-f["F_pred_mid"][19]
        adjoint = torch.tensor(target[None])
        den, signed_total = float(np.vdot(target, target)), 0.
        for n in reversed(NODES):
            function, anchor = jacobian(model, boundaries, middle, context, start, n)
            _, pullback = torch.func.vjp(function, anchor)
            before = pullback(adjoint)[0].detach()
            counts["vjps"] += 1
            direction = torch.tensor(data[f"n{n}_direction"][None])
            tangent = torch.tensor(data[f"n{n}_tangent"][None])
            if not torch.count_nonzero(direction):
                grid = torch.arange(anchor.numel(), dtype=torch.float64).reshape_as(anchor)
                direction = torch.sin(grid*.013+n*.29)
                direction *= .05*torch.linalg.vector_norm(anchor)/torch.linalg.vector_norm(direction)
                tangent = qj(function, anchor, direction, f"n{n}_dual", data, witnesses, checks, counts)
            check(checks, "reverse_bilinear_identity", abs(float((before*direction).sum()-(adjoint*tangent).sum())), 1e-9)
            r = data[f"n{n}_remainder"]
            score = float(np.vdot(array(adjoint[0]), r)/den)
            signed_total += score
            data[f"n{n}_adjoint_out"], data[f"n{n}_adjoint_in"] = array(adjoint[0]), array(before[0])
            rows.append(dict(key=key, **case["group"], node=n, layer=n//2+1, kind=spec(n)[1],
                local_remainder_norm=float(np.linalg.norm(r)), incoming_error_norm=float(np.linalg.norm(data[f"n{n}_direction"])),
                terminal_error_norm=float(np.sqrt(den)), adjoint_norm=float(torch.linalg.vector_norm(adjoint)),
                signed_share=score, projected_gain=score*np.sqrt(den)/np.linalg.norm(r) if np.linalg.norm(r) else 0.))
            adjoint = before
            del function, pullback
        check(checks, "complete_adjoint_scalar_closure", abs(signed_total-1), 1e-9)
        data["target_error"] = target
        entries.append(saved(out, "ledger", case, data, checks, witnesses))
        print(f"remainder ledger {number}/32", flush=True)
    rank = sorted(NODES, key=lambda n:(-float(np.median([abs(r["signed_share"]) for r in rows if r["node"] == n])), n))
    selection = dict(rule="Median absolute per-case signed terminal projection; chronological tie-break; before intervention outcomes.",
        selected=rank[:4], ranking=[dict(node=n, layer=n//2+1, kind=spec(n)[1],
            median_absolute_share=float(np.median([abs(r["signed_share"]) for r in rows if r["node"] == n])),
            median_signed_share=float(np.median([r["signed_share"] for r in rows if r["node"] == n]))) for n in rank])
    with (out/"ledger.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out/"selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    (out/"ledger-status.json").write_text(json.dumps(dict(status="complete", counts=dict(counts), cases=entries,
        selection_sha256=sha(out/"selection.json")), indent=2)+"\n")
    print(json.dumps(selection, indent=2), flush=True)


def intervene(out):
    assert not (out/"status.json").exists()
    assert json.loads((out/"ledger-audit.json").read_text())["status"] == "passed"
    design, selection = (json.loads((out/name).read_text()) for name in ("design.json", "selection.json"))
    for p, h in design["source_hashes"].items():
        assert sha(ROOT/p) == sha(out/"sources"/Path(p).name) == h
    old, previous = Path(design["input228"]), Path(design["input237"])
    selected = selection["selected"]
    model, counts, entries = load_model(True), Counter(), []
    cases = json.loads((ROOT/"research/boolean_coupling/cases228.json").read_text())
    with torch.no_grad():
        for number, case in enumerate(cases, 1):
            key, start = case["key"], case["position"]+1
            f, o = (dict(np.load(old/phase/(key+".npz"))) for phase in ("forecasts", "observations"))
            old237 = dict(np.load(previous/"cases"/(key+".npz")))
            ledger_data = dict(np.load(out/"ledger"/(key+".npz")))
            checks, data, witnesses = [], {}, []
            PROGRESS.update(key=key, checks=checks, data=data, fd_witnesses=witnesses)
            with mids(model) as middle:
                false_states, context, _ = observe(model, torch.tensor([case["false_ids"]], device="mps"))
            counts["ordinary_false"] += 1
            boundaries = [None, *false_states]
            parents, source_mid, source_out = {}, {}, {}
            for i in range(1, 20):
                operand = boundaries[i].clone()
                operand[:, start-1] = torch.tensor(f["source_rows"][i-1][None], device="mps")
                source_out[i], source_mid[i] = source_block(model, i, operand, context)
                parents[i] = operand
                counts["source_blocks"] += 1
                check(checks, "source_mid_replay", np.max(abs(array(source_mid[i][0, start:]).astype(float)
                    -f["false_mids"][i].astype(float)-f["source_mid"][i])))
                check(checks, "source_out_replay", np.max(abs(array(source_out[i][0, start:]).astype(float)
                    -f["false_boundaries"][i+1].astype(float)-f["source_out"][i])))
            proposed = {arm:torch.tensor(o["true_states"][0][None], dtype=torch.float64) for arm in ("own4", "random4")}
            correction = torch.zeros_like(proposed["own4"])
            for n in NODES:
                i, kind = spec(n)
                function, anchor = jacobian(model, boundaries, middle, context, start, n)
                tangent = qj(function, anchor, correction, f"oracle_n{n}", data, witnesses, checks, counts)
                correction = tangent+(torch.tensor(ledger_data[f"n{n}_remainder"][None]) if n in selected else 0)
                data[f"oracle_n{n}_output"] = array(correction[0])
                source_input, base = rule(f, i, kind)
                for arm in ("own4", "random4"):
                    label = f"{arm}_n{n}"
                    current = proposed[arm]
                    tangent = qj(function, anchor, current-torch.tensor(source_input[None]), label, data, witnesses, checks, counts)
                    linear = torch.tensor(base[None])+tangent
                    data[label+"_input"], data[label+"_linear"] = array(current[0]), array(linear[0])
                    value = linear
                    if n in selected:
                        operand = (parents[i] if kind == "attention" else source_mid[i]).clone()
                        operand[:, start:] = current.to("mps", torch.float32)
                        layer = model.model.layers[i]
                        if kind == "attention":
                            finite = operand+layer.self_attn(layer.input_layernorm(operand), attention_mask=context["mask"],
                                position_embeddings=context["rope"])[0]
                        else:
                            finite = operand+layer.mlp(layer.post_attention_layernorm(operand))
                        counts["selected_finite"] += 1
                        data[label+"_finite_input"], data[label+"_finite_output"] = array(operand[0, start:]), array(finite[0, start:])
                        value = promote(finite[:, start:])
                        if arm == "random4":
                            delta = array((value-linear)[0])
                            seed = int(hashlib.sha256((key+str(n)+"238").encode()).hexdigest()[:16], 16)
                            random = np.random.default_rng(seed).standard_normal(delta.shape)
                            random *= np.linalg.norm(delta, axis=-1, keepdims=True)/np.linalg.norm(random, axis=-1, keepdims=True)
                            data[label+"_random"] = random
                            value = linear+torch.tensor(random[None])
                    proposed[arm] = value
                    data[label+"_output"] = array(value[0])
                del function
            data["oracle_correction"] = array(correction[0])
            values = dict(base=f["F_pred_mid"][19], oracle4=f["F_pred_mid"][19]+array(correction[0]),
                own4=array(proposed["own4"][0]), random4=array(proposed["random4"][0]), native=o["true_mids"][19])
            target = ledger_data["target_error"]
            shares = sum(float(np.vdot(ledger_data[f"n{n}_adjoint_out"], ledger_data[f"n{n}_remainder"])/np.vdot(target, target)) for n in selected)
            check(checks, "selected_transport_projection", abs(float(np.vdot(array(correction[0]), target)/np.vdot(target, target))-shares), 1e-9)
            for arm, value in values.items():
                parent = source_mid[19].clone()
                parent[:, start:] = torch.tensor(value[None], device="mps", dtype=torch.float32)
                data[arm+"_parent"] = array(parent[0, start:])
                layer = model.model.layers[19]
                parent = parent+layer.mlp(layer.post_attention_layernorm(parent))
                counts["suffix_mlp20"] += 1
                for i in range(20, 28):
                    parent = block(model, i, parent, context)
                    counts["suffix_blocks"] += 1
                data[arm+"_final"] = array(parent[0, start:])
                data[arm+"_scores"] = array(torch.nn.functional.linear(model.model.norm(parent)[0, -1], model.lm_head.weight[[895, 830]]))
                counts["heads"] += 1
            check(checks, "base_237_replay", np.max(abs(data["base_final"]-old237["forecast_public_final"])))
            check(checks, "base_237_score_replay", np.max(abs(data["base_scores"]-old237["forecast_public_scores"])), .005)
            check(checks, "native_replay", np.max(abs(data["native_final"]-o["true_final"])))
            check(checks, "native_score_replay", np.max(abs(data["native_scores"]-o["true_scores"])), .005)
            entries.append(saved(out, "cases", case, data, checks, witnesses,
                selection_sha256=sha(out/"selection.json"), ledger_arrays_sha256=sha(out/"ledger"/(key+".npz"))))
            print(f"selected upstream intervention {number}/32", flush=True)
    (out/"status.json").write_text(json.dumps(dict(status="complete", counts=dict(counts), cases=entries,
        selection_sha256=sha(out/"selection.json"), ledger_audit_sha256=sha(out/"ledger-audit.json")), indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("ledger", "intervene"), required=True)
    parser.add_argument("--input-237", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.phase == "ledger":
            assert args.input_237
            ledger(args.input_237.resolve(), args.out.resolve())
        else:
            intervene(args.out.resolve())
    except Exception as exc:
        if args.out.exists():
            if PROGRESS:
                np.savez_compressed(args.out/("partial-"+args.phase+".npz"), **PROGRESS["data"])
                (args.out/("partial-"+args.phase+".json")).write_text(json.dumps({k:v for k, v in PROGRESS.items() if k != "data"}, indent=2)+"\n")
            (args.out/("failure-"+args.phase+".json")).write_text(json.dumps(dict(error=repr(exc)), indent=2)+"\n")
        raise
