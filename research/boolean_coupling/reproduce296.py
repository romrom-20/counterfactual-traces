"""Numerical repetition on already-public296 inputs; not a fresh sealed prediction."""

import argparse
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read, table
from .audit296 import audit_bundle, native_sources, populations, scientific
from .cache272 import save_json
from .core225b import load_model
from .query296 import ROOT, execute


def repeat(model, cases, out):
    assert (
        not out.exists()
        and out.parent.exists()
        and shutil.disk_usage(out.parent).free >= 512 * 1024**2
    )
    out.mkdir()
    (out / "cases").mkdir()
    checks = []
    raw = []
    decisions = []
    vectors = []
    counts = Counter()
    entries = []
    w = dict(
        norm=model.model.norm.weight.detach().cpu().numpy().astype(float),
        eps=np.array(model.config.rms_norm_eps),
        head=model.lm_head.weight.detach().cpu().numpy()[[895, 830]].astype(float),
    )
    for i, c in enumerate(cases, 1):
        assert shutil.disk_usage(out).free >= 512 * 1024**2
        f, fw, fc, fcount, fm = execute(model, c, "forecast")
        fm["checks"] = fc
        counts.update(fcount)
        t, tw, tc, tcount, tm = execute(model, c, "native", f, fm)
        tm["checks"] = tc
        counts.update(tcount)
        audit_bundle(c, f, fw, fm, w, "forecast", checks, raw)
        audit_bundle(c, t, tw, tm, w, "native", checks, raw)
        native_sources(c, fm, tm, checks)
        rr, vv = scientific(c, f, t)
        decisions.extend(rr)
        vectors.extend(vv)
        p = out / "cases" / (c["key"] + ".npz")
        np.savez_compressed(p, **f, **t, **fw, **tw)
        save_json(
            p.with_suffix(".json"), dict(key=c["key"], arrays_sha256=sha(p), forecast=fm, native=tm)
        )
        entries.append(
            dict(
                key=c["key"],
                arrays_sha256=sha(p),
                metadata_sha256=sha(p.with_suffix(".json")),
                bytes=p.stat().st_size,
            )
        )
        save_json(out / f"progress-{i:02d}.json", dict(cases=i, counts=dict(counts)))
        print("296 repetition", i, "/", len(cases), flush=True)
    failed = [r for r in checks if not r["passes"]]
    table(out / "checks.csv", checks)
    table(out / "raw-logits.csv", raw)
    a = dict(
        numerical_repetition=True,
        globally_pre_native=False,
        synthetic=model.config.hidden_size != 1536,
        general_question_B_solved=False,
        status="failed" if failed else "passed",
        required_checks=len(checks),
        failed_checks=len(failed),
        raw_logit_failed=sum(not r["passes"] for r in raw),
        raw_logit_checks=len(raw),
        counts=dict(counts),
        cases=entries,
        scientific_outcomes_withheld=bool(failed),
    )
    if not failed:
        table(out / "native.csv", decisions)
        save_json(out / "vector-records.json", dict(statistics=vectors))
        a["populations"] = populations(decisions)
    save_json(out / "audit.json", a)
    assert not failed, "Numerical gate failed; repetition outcomes withheld"
    return a


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--offline", action="store_true")
    a = p.parse_args()
    d = read(ROOT / "results/query-geometry-296/pre-native/design.json")
    for n, h in d["source_hashes"].items():
        assert sha(ROOT / n) == h, n
    result = repeat(load_model(a.offline), d["cases"], a.out.resolve())
    print(result["status"], result["populations"], flush=True)
