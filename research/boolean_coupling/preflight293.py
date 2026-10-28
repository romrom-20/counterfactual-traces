"""Tiny CPU complete diagnostic, unequal inputs and coherent rejection fixtures."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from . import audit293, run293
from .audit229 import sha
from .audit272 import read
from .cache272 import save_json
from .joint292 import Engine
from .preflight285 import cases
from .preflight286 import tiny


def main(out):
    assert not out.exists()
    out.mkdir()
    model = tiny()
    cc = run293.selected(dict(cases=cases()))
    assert len(cc) == 8
    refs = {}
    for c in cc:
        e = Engine(model, c, allow_native=True)
        f, fm = e.run("F")
        p, pm = e.run("P")
        t, _ = e.run("T")
        refs[c["key"]] = (
            {**{"F_" + k: v for k, v in f.items()}, **{"P_" + k: v for k, v in p.items()}},
            t,
            dict(arms=dict(F=fm, P=pm)),
        )
    run293.main(out / "complete", model=model, cases=cc, refs=refs)
    rejected = []
    for name in ("bounds", "basis"):
        dest = out / ("rejected-" + name)
        shutil.copytree(
            out / "complete", dest, ignore=shutil.ignore_patterns("audit.json", "*.csv")
        )
        if name == "bounds":
            p = dest / "design.json"
            d = read(p)
            d["bounds"]["range"] = 0.9
            p.write_text(json.dumps(d, indent=2) + "\n")
        else:
            st = read(dest / "status.json")
            e = st["cases"][0]
            p = dest / "cases" / (e["key"] + ".npz")
            with np.load(p) as z:
                a = {n: z[n] for n in z.files}
            k = next(n for n in a if n.startswith("E_probe") and n.endswith("current_Q"))
            # Permute residual coordinates: outer hashes and orthonormality survive.
            a[k] = a[k][..., ::-1].copy()
            np.savez_compressed(p, **a)
            m = read(p.with_suffix(".json"))
            m["arrays_sha256"] = sha(p)
            p.with_suffix(".json").write_text(json.dumps(m, indent=2) + "\n")
            e.update(
                arrays_sha256=sha(p),
                metadata_sha256=sha(p.with_suffix(".json")),
                bytes=p.stat().st_size,
            )
            st["raw_bytes"] = sum(e["bytes"] for e in st["cases"])
            (dest / "status.json").write_text(json.dumps(st, indent=2) + "\n")
        try:
            audit293.main(dest, model=model)
        except AssertionError:
            assert (
                not (dest / "audit.json").exists()
                or read(dest / "audit.json")["status"] == "failed"
            )
            rejected.append(name)
        else:
            raise AssertionError("Corruption accepted: " + name)
    save_json(
        out / "preflight-summary.json",
        dict(
            passed=True,
            synthetic=True,
            checkpoint_loaded=False,
            cases=8,
            unequal_lengths=sorted({len(c["false_ids"]) for c in cc}),
            required_checks=read(out / "complete/audit.json")["required_checks"],
            rejected=rejected,
            scientific_outcomes_not_required=True,
        ),
    )
    print(read(out / "preflight-summary.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args().out.resolve())
