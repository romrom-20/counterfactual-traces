"""Read checkpoint tensors and saved probes only; no model forward or native call."""

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
from .audit294 import qualify
from .cache272 import installed_sources, save_json
from .change294 import ARMS, BOUNDS, PRIMARY, ROOT, SOURCES, decisions, evaluate, records
from .checkpoint238 import locate, weights


def parent_probe(z, site):
    prefix = f"P_probe{site}_"
    names = (
        "nonlinear",
        "current_a",
        "current_b",
        "current_omega",
        "current_Q",
        "False_a",
        "False_b",
        "False_omega",
        "False_Q",
    )
    return {n: z[prefix + n] for n in names}


def main(parent, qualified, out):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 512 * 1024**2
    public = ROOT / "results/response-subspace-293"
    for name in ("design.json", "status.json", "checks.csv"):
        assert sha(parent / name) == sha(public / name)
    assert (
        sha(qualified / "audit.json") == sha(public / "audit.json")
        and read(qualified / "audit.json")["status"] == "passed"
    )
    d = read(parent / "design.json")
    st = read(parent / "status.json")
    assert len(d["cases"]) == 8
    assert installed_sources() == d["installed_sources"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {n: sha(ROOT / n) for n in SOURCES}
    for n, h in hashes.items():
        assert (
            hashlib.sha256(
                subprocess.check_output(["git", "show", commit + ":" + n], cwd=ROOT)
            ).hexdigest()
            == h
        )
    out.mkdir()
    (out / "sources").mkdir()
    (out / "cases").mkdir()
    for n in SOURCES:
        shutil.copy2(ROOT / n, out / "sources" / Path(n).name)
    folder, checkpoint = locate(ROOT)
    config = read(folder / "config.json")
    assert config["hidden_size"] == 1536 and config["num_hidden_layers"] == 28
    torch.set_num_threads(2)
    save_json(
        out / "design.json",
        dict(
            experiment=294,
            retrospective=True,
            conditional_P_inputs=True,
            source_commit=commit,
            source_hashes=hashes,
            checkpoint=checkpoint,
            installed_sources=installed_sources(),
            cases=d["cases"],
            parent_entries=st["cases"],
            reference_hashes={
                n: sha(parent / n) for n in ("design.json", "status.json", "checks.csv")
            },
            qualified_reference_sha256=sha(qualified / "audit.json"),
            arms=list(ARMS),
            primary=PRIMARY,
            bounds=BOUNDS,
            sites=[0, 14, 27],
            guards_MiB=[512, 384, 192],
            model_forwards=0,
            native_true_calls=0,
            boundary="48 saved P probe points; same inputs and32-dimensional construction budget. Not native prediction, causal repair, efficiency or learned-circuit evidence.",
        ),
    )
    save_json(
        out / "runtime.json",
        dict(
            python=platform.python_version(),
            numpy=np.__version__,
            torch=str(torch.__version__),
            arithmetic="CPUFloat64",
            device="cpu",
            checkpoint_tensor_access="safetensors local read; no model instantiated",
            model_forwards=0,
        ),
    )
    packs = {c["key"]: {} for c in d["cases"]}
    metas = {c["key"]: [] for c in d["cases"]}
    allchecks = []
    metrics = []
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:
        for site in (0, 14, 27):
            assert shutil.disk_usage(out).free >= 384 * 1024**2
            w = weights(handle, config, site, "mlp")
            for c, e in zip(d["cases"], st["cases"], strict=True):
                key = c["key"]
                p = parent / "cases" / (key + ".npz")
                assert (
                    sha(p) == e["arrays_sha256"]
                    and sha(p.with_suffix(".json")) == e["metadata_sha256"]
                )
                with np.load(p) as z:
                    probe = parent_probe(z, site)
                pack, checks = evaluate(w, probe)
                checks += qualify(w, probe, pack)
                allchecks.extend(dict(key=key, layer=site + 1, **r) for r in checks)
                rows = records(key, site, pack)
                metrics.extend(rows)
                metas[key].extend(rows)
                packs[key].update({f"probe{site}_" + n: v for n, v in pack.items()})
                print(
                    "294 probe",
                    site + 1,
                    key[:8],
                    "numeric failures",
                    sum(not r["passes"] for r in checks),
                    flush=True,
                )
            del w
    entries = []
    total = 0
    for c in d["cases"]:
        key = c["key"]
        p = out / "cases" / (key + ".npz")
        np.savez_compressed(p, **packs[key])
        save_json(
            p.with_suffix(".json"),
            dict(key=key, group=c["group"], arrays_sha256=sha(p), metrics=metas[key]),
        )
        e = dict(
            key=key,
            arrays_sha256=sha(p),
            metadata_sha256=sha(p.with_suffix(".json")),
            bytes=p.stat().st_size,
        )
        entries.append(e)
        total += e["bytes"]
        assert total <= 192 * 1024**2
    table(out / "checks.csv", allchecks)
    failed = [r for r in allchecks if not r["passes"]]
    status = dict(
        status="complete", cases=entries, raw_bytes=total, model_forwards=0, native_true_calls=0
    )
    save_json(out / "status.json", status)
    report = dict(
        experiment=294,
        status="failed" if failed else "passed",
        design_sha256=sha(out / "design.json"),
        status_sha256=sha(out / "status.json"),
        required_checks=len(allchecks),
        failed_checks=len(failed),
        model_forwards=0,
        general_question_B_solved=False,
        scientific_outcomes_withheld=bool(failed),
    )
    if not failed:
        dd, pop = decisions(metrics)
        table(out / "metrics.csv", metrics)
        table(out / "decisions.csv", dd)
        report.update(populations=pop, primary_pass=pop["H1_improvement"]["population_pass"])
    report["csv_sha256"] = {p.name: sha(p) for p in out.glob("*.csv")}
    save_json(out / "audit.json", report)
    print(report, flush=True)
    assert not failed, "Numerical gate failed; scientific tables withheld"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("parent", "qualified", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    try:
        main(a.parent.resolve(), a.qualified.resolve(), a.out.resolve())
    except Exception as exc:
        if a.out.exists() and not (a.out / "failure.json").exists():
            save_json(a.out / "failure.json", dict(error=repr(exc)))
        raise
