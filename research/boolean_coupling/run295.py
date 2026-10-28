"""Read pinned down weights and saved probes; no model or native observations."""

import argparse
import hashlib
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .audit229 import sha
from .audit272 import read, table
from .audit295 import qualify
from .cache272 import save_json
from .checkpoint238 import locate
from .drive295 import ARMS, BOUNDS, PRIMARY, ROOT, SOURCES, decisions, evaluate, records


def main(parent, previous, out):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 512 * 1024**2
    oldpub = ROOT / "results/response-subspace-293"
    pub = ROOT / "results/response-change-294"
    for local, public, names in [
        (parent, oldpub, ("design.json", "status.json", "checks.csv")),
        (previous, pub, ("design.json", "status.json", "audit.json", "metrics.csv")),
    ]:
        for name in names:
            assert sha(local / name) == sha(public / name)
    old = read(parent / "design.json")
    st = read(parent / "status.json")
    prev = read(previous / "status.json")
    pd = read(previous / "design.json")
    assert read(previous / "audit.json")["status"] == "passed" and len(old["cases"]) == 8
    assert pd["parent_entries"] == st["cases"] and old["cases"] == pd["cases"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {n: sha(ROOT / n) for n in SOURCES}
    for n, h in hashes.items():
        assert (
            hashlib.sha256(
                subprocess.check_output(["git", "show", commit + ":" + n], cwd=ROOT)
            ).hexdigest()
            == h
        )
    folder, checkpoint = locate(ROOT)
    assert checkpoint == pd["checkpoint"]
    out.mkdir()
    (out / "cases").mkdir()
    (out / "sources").mkdir()
    for n in SOURCES:
        shutil.copy2(ROOT / n, out / "sources" / Path(n).name)
    save_json(
        out / "design.json",
        dict(
            experiment=295,
            retrospective=True,
            conditional_P_inputs=True,
            source_commit=commit,
            source_hashes=hashes,
            cases=old["cases"],
            parent_entries=st["cases"],
            previous_entries=prev["cases"],
            reference_hashes={
                str(public.relative_to(ROOT) / name): sha(local / name)
                for local, public, names in [
                    (parent, oldpub, ("design.json", "status.json", "checks.csv")),
                    (previous, pub, ("design.json", "status.json", "audit.json", "metrics.csv")),
                ]
                for name in names
            },
            checkpoint=checkpoint,
            arms=list(ARMS),
            primary=PRIMARY,
            bounds=BOUNDS,
            sites=[0, 14, 27],
            guards_MiB=[512, 384, 32],
            model_forwards=0,
            native_true_calls=0,
            stopping_rule="Close this local line after this directed endpoint/path diagnostic; no automatic full correction/rank/node sweep.",
            boundary="SuppliedPgate/up endpoints; target only for scoring and labelled oracle. Known-function integral is calibration, not a learned circuit or native forecast.",
        ),
    )
    save_json(
        out / "runtime.json",
        dict(
            python=platform.python_version(),
            numpy=np.__version__,
            torch=str(torch.__version__),
            arithmetic="CPUFloat64",
            model_instantiations=0,
            model_forwards=0,
            checkpoint_tensor_access="Three down weights only",
            speedup_claimed=False,
        ),
    )
    torch.set_num_threads(2)
    packs = {c["key"]: {} for c in old["cases"]}
    meta = {c["key"]: [] for c in old["cases"]}
    checks = []
    metrics = []
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:
        for site in [0, 14, 27]:
            assert shutil.disk_usage(out).free >= 384 * 1024**2
            wd = (
                handle.get_tensor(f"model.layers.{site}.mlp.down_proj.weight")
                .to(torch.float32)
                .numpy()
                .astype(float)
            )
            for c, e, pe in zip(old["cases"], st["cases"], prev["cases"], strict=True):
                key = c["key"]
                p = parent / "cases" / (key + ".npz")
                q = previous / "cases" / (key + ".npz")
                assert (
                    e["key"] == pe["key"] == key
                    and sha(p) == e["arrays_sha256"]
                    and sha(p.with_suffix(".json")) == e["metadata_sha256"]
                    and sha(q) == pe["arrays_sha256"]
                    and sha(q.with_suffix(".json")) == pe["metadata_sha256"]
                )
                with np.load(p) as z:
                    probe = {
                        n: z[f"P_probe{site}_" + n]
                        for n in [
                            "g",
                            "u",
                            "phi",
                            "slope",
                            "nonlinear",
                            "write",
                            "False_write",
                            "linear",
                        ]
                    }
                    probe.update(
                        {"False_" + n: z[f"False{site}_" + n] for n in ["g", "u", "phi", "slope"]}
                    )
                with np.load(q) as z:
                    prior = z[f"probe{site}_CHANGE_RANGE_projected"]
                data = evaluate(wd, probe, prior)
                rr = qualify(wd, probe, data, prior)
                checks.extend(dict(key=key, layer=site + 1, **r) for r in rr)
                mm = records(key, site, data)
                metrics.extend(mm)
                meta[key].extend(mm)
                packs[key].update({f"probe{site}_" + n: v for n, v in data.items()})
                print(
                    "295 probe",
                    site + 1,
                    key[:8],
                    "numeric failures",
                    sum(not r["passes"] for r in rr),
                    flush=True,
                )
            del wd
    entries = []
    total = 0
    for c in old["cases"]:
        key = c["key"]
        p = out / "cases" / (key + ".npz")
        np.savez_compressed(p, **packs[key])
        save_json(
            p.with_suffix(".json"),
            dict(key=key, group=c["group"], arrays_sha256=sha(p), metrics=meta[key]),
        )
        e = dict(
            key=key,
            arrays_sha256=sha(p),
            metadata_sha256=sha(p.with_suffix(".json")),
            bytes=p.stat().st_size,
        )
        entries.append(e)
        total += e["bytes"]
        assert total <= 32 * 1024**2
    save_json(
        out / "status.json",
        dict(
            status="complete", cases=entries, raw_bytes=total, model_forwards=0, native_true_calls=0
        ),
    )
    table(out / "checks.csv", checks)
    failed = [r for r in checks if not r["passes"]]
    audit = dict(
        experiment=295,
        status="failed" if failed else "passed",
        design_sha256=sha(out / "design.json"),
        status_sha256=sha(out / "status.json"),
        required_checks=len(checks),
        failed_checks=len(failed),
        scientific_outcomes_withheld=bool(failed),
        general_question_B_solved=False,
        learned_circuit_identified=False,
        model_forwards=0,
        native_true_calls=0,
    )
    if not failed:
        table(out / "metrics.csv", metrics)
        pop = decisions(metrics)
        audit.update(
            populations=pop,
            primary_pass=pop["ENDPOINT_DRIVE"]["population_pass"],
            oracle_direction_pass=pop["ORACLE_DIRECTION"]["population_pass"],
        )
    audit["csv_sha256"] = {p.name: sha(p) for p in out.glob("*.csv")}
    save_json(out / "audit.json", audit)
    print(audit, flush=True)
    assert not failed, "Numerical gate failed; scientific tables withheld"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ["parent", "previous", "out"]:
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    main(a.parent.resolve(), a.previous.resolve(), a.out.resolve())
