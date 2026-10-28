"""Resource-only AST qualification, all-case exact replay and corruption refusal."""

import argparse
from pathlib import Path

import numpy as np

from .audit272 import read
from .cache272 import save_json
from .joint292 import Engine
from .preflight285 import cases
from .preflight286 import tiny
from .resource293b import main as continued
from .resource293b import modules, replay
from .run293 import selected


def main(original, out):
    assert not out.exists()
    out.mkdir()
    modules()
    model = tiny()
    cc = selected(dict(cases=cases()))
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
    continued(original, out / "complete", model=model, cases=cc, refs=refs)
    key = cc[0]["key"]
    p = original / "cases" / (key + ".npz")
    with np.load(p) as z:
        a = {n: z[n] for n in z.files}
    m = read(p.with_suffix(".json"))
    meta = {k: v for k, v in m.items() if k not in ("key", "arrays_sha256")}
    a["E_final"].flat[0] += 0.01
    try:
        replay(original, key, a, meta)
    except AssertionError:
        rejected = True
    else:
        raise AssertionError("Coherent replay mismatch accepted")
    proof = read(out / "complete/resource-qualification.json")
    assert (
        len(proof["replayed_arrays"]) == 8
        and proof["qualified_total_cap_MiB"] == 384
        and proof["initial_and_percase_free_MiB"] == [512, 384]
    )
    save_json(
        out / "preflight-summary.json",
        dict(
            passed=True,
            synthetic=True,
            checkpoint_loaded=False,
            resource_only_literals=4,
            replayed_cases=8,
            replayed_arrays=proof["replayed_arrays"],
            coherent_mismatch_rejected=rejected,
            required_checks=read(out / "complete/audit.json")["required_checks"],
        ),
    )
    print(read(out / "preflight-summary.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.original.resolve(), a.out.resolve())
