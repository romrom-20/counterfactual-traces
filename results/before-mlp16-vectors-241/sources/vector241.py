"""Complete transported-remainder vectors before MLP16; no new model forwards."""

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
from safetensors import safe_open

from .audit229 import checked, sha
from .checkpoint238 import locate, weights
from .maps241 import maps

ROOT = Path(__file__).resolve().parents[2]
NODES = tuple(range(2, 31))
SOURCES = [ROOT / p for p in (
    "research/boolean_coupling/vector241.py", "research/boolean_coupling/maps241.py",
    "research/boolean_coupling/audit241.py", "research/boolean_coupling/checkpoint238.py",
    "research/boolean_coupling/audit229.py", "research/boolean_coupling/audit235.py",
    "research/boolean_coupling/curvature239.py", "research/boolean_coupling/settings.json",
    "docs/experiments/241-before-mlp16-vectors.md")]


def digest(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def inputs(run):
    design = json.loads((run / "design.json").read_text())
    previous, old, last = (Path(design[k]) for k in ("input238", "input228", "input240"))
    return design, previous, old, last


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main(previous, last, out):
    assert not out.exists(), "Never overwrite evidence."
    assert json.loads((previous / "audit.json").read_text())["status"] == "passed"
    assert json.loads((last / "audit.json").read_text())["status"] == "passed"
    old = Path(json.loads((previous / "design.json").read_text())["input228"])
    folder, checkpoint = locate(ROOT)
    assert checkpoint == json.loads((previous / "design.json").read_text())["checkpoint"]
    config = json.loads((folder / "config.json").read_text())
    out.mkdir(parents=True)
    for name in ("cases", "sources"):
        (out / name).mkdir()
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in SOURCES}
    for p in SOURCES:
        (out / "sources" / p.name).write_bytes(p.read_bytes())
    write(out / "design.json", dict(experiment=241, retrospective=True, input238=str(previous), input228=str(old),
        input240=str(last), checkpoint=checkpoint, source_hashes=hashes, nodes=NODES,
        input_audits={"238": sha(previous / "audit.json"), "240": sha(last / "audit.json")}))
    write(out / "runtime.json", dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        dtype="float64", device="cpu", ordinary_model_calls=0))
    torch.set_num_threads(2)
    inventory = {}
    for tag, release in (("228", "false-only-alignment-228"), ("238", "upstream-remainder-238"), ("240", "target-blind-silu-240")):
        inventory[tag] = {r["path"]: r["sha256"] for r in json.loads((ROOT / "results" / release / "local-array-inventory.json").read_text())["arrays"]}
    entries, energy, count = [], [], Counter()
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:
        for ent in json.loads((previous / "status.json").read_text())["cases"]:
            key, checks, data, witnesses = ent["key"], [], {}, []
            paths = dict(ledger=previous / "ledger" / (key + ".npz"), forecasts=old / "forecasts" / (key + ".npz"),
                observations=old / "observations" / (key + ".npz"), mask=last / "masks" / (key + ".npz"))
            for label, tag, phase in (("ledger", "238", "ledger"), ("forecasts", "228", "forecasts"),
                ("observations", "228", "observations"), ("mask", "240", "masks")):
                assert sha(paths[label]) == inventory[tag][phase + "/" + key + ".npz"]
            ledger, f, o, installed = (dict(np.load(paths[k])) for k in ("ledger", "forecasts", "observations", "mask"))
            meta = json.loads(paths["ledger"].with_suffix(".json").read_text())
            start, group = meta["start"], meta["group"]
            context = {k: ledger[k].astype(float) for k in ("mask", "rope_cos", "rope_sin")}
            shape = ledger["n2_direction"].shape
            terms = np.empty((0, *shape), dtype=float)
            for n in NODES:
                i, kind = n // 2, "attention" if n % 2 == 0 else "mlp"
                w = weights(handle, config, i, kind)
                full = ledger[f"a{i}_false"].astype(float) if kind == "attention" else f["false_mids"][i].astype(float)
                function, analytic, numpy_function = maps(w, context, full, start, kind)
                anchor = full[start:] if kind == "attention" else full
                t = torch.from_numpy(anchor)
                directions = np.concatenate((terms, ledger[f"n{n}_direction"][None]))
                d = torch.from_numpy(directions)
                tangent = torch.vmap(lambda v, fn=function, a=t: torch.func.jvp(fn, (a,), (v,))[1])(d).numpy()
                plus, minus = (torch.vmap(lambda v, fn=function, a=t, s=sign: fn(a + s * 1e-5 * v))(d).numpy() for sign in (1, -1))
                fd = (plus - minus) / 2e-5
                checked(checks, "all_used_JVP_FD", tangent, fd)
                checked(checks, "all_used_JVP_analytic", tangent, analytic(directions), 1e-9)
                checked(checks, "saved_native_error_JVP", tangent[-1], ledger[f"n{n}_tangent"], 1e-9)
                primal = function(t).numpy()
                checked(checks, "independent_primal", primal, numpy_function(anchor), 1e-9)
                public = f["false_mids"][i] if kind == "attention" else f["false_boundaries"][i + 1]
                checked(checks, "false_public_primal", primal, public)
                e, jt, r = (ledger[f"n{n}_" + k] for k in ("direction", "tangent", "remainder"))
                after = jt + r
                amplification = float(np.vdot(jt, jt) - np.vdot(e, e))
                generation, cross = float(np.vdot(r, r)), float(2 * np.vdot(jt, r))
                growth = float(np.vdot(after, after) - np.vdot(e, e))
                checked(checks, "squared_error_increment", np.array(amplification + generation + cross), np.array(growth), 1e-8)
                energy.append(dict(key=key, **group, node=n, kind=kind, layer=i + 1,
                    error_before=float(np.linalg.norm(e)), transported_error=float(np.linalg.norm(jt)),
                    error_after=float(np.linalg.norm(after)), remainder_norm=float(np.linalg.norm(r)),
                    transport_energy_change=amplification, remainder_energy=generation, cross_energy=cross, net_energy_change=growth))
                terms = np.concatenate((tangent[:-1], r[None]))
                checked(checks, "intermediate_vector_closure", terms.sum(0), after, 1e-8)
                data[f"n{n}_tangent"] = tangent
                witnesses.append(dict(node=n, directions_sha256=digest(directions), plus_sha256=digest(plus), minus_sha256=digest(minus),
                    directions=len(directions), max_fd_error=float(np.max(abs(tangent - fd)))))
                count["JVPs"] += len(directions)
                count["FD_endpoints"] += 2 * len(directions)
                del function, numpy_function, w, d, t
            raw_target = o["true_mids"][15].astype(float) - f["F_pred_mid"][15]
            cast = f["F_pred_mid"][15] - installed["own_input"].astype(float)
            target = o["true_mids"][15].astype(float) - installed["own_input"].astype(float)
            checked(checks, "raw_terminal_vector_closure", terms.sum(0), raw_target, 1e-8)
            checked(checks, "installed_terminal_vector_closure", terms.sum(0) + cast, target, 1e-8)
            checked(checks, "energy_profile_closure", np.array(sum(r["net_energy_change"] for r in energy if r["key"] == key)),
                np.array(np.vdot(raw_target, raw_target)), 1e-8)
            data.update(terms=terms, cast=cast, target=target, raw_target=raw_target,
                gram=np.einsum("irh,jrh->ij", terms, terms))
            path = out / "cases" / (key + ".npz")
            assert shutil.disk_usage(out).free > 1024**3
            np.savez_compressed(path, **data)
            write(path.with_suffix(".json"), dict(key=key, group=group, start=start, arrays_sha256=sha(path), checks=checks,
                witnesses=witnesses, inputs={k: sha(p) for k, p in paths.items()}))
            entries.append(dict(key=key, arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json"))))
            print(f"transported complete remainder vectors {len(entries)}/32", flush=True)
    with (out / "energy.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(energy[0]))
        writer.writeheader()
        writer.writerows(energy)
    assert dict(count) == dict(JVPs=13920, FD_endpoints=27840)
    write(out / "status.json", dict(status="complete", cases=entries, counts=dict(count), ordinary_model_calls=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-238", type=Path, required=True)
    parser.add_argument("--input-240", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.input_238.resolve(), args.input_240.resolve(), args.out.resolve())
    except Exception as exc:
        if args.out.exists():
            write(args.out / "failure.json", dict(error=repr(exc)))
        raise
