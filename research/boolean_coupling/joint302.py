"""Posthoc signed joint ledger on already measured 302 corners; no new model calls."""

import argparse
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read, table
from .cache272 import save_json
from .geometry302 import ROOT, delta, statistics


def main(run, out):
    assert read(run / "audit.json")["status"] == "passed"
    assert not out.exists()
    out.mkdir(parents=True)
    rows, closure = [], []
    for case in read(run / "design.json")["cases"]:
        key, data = case["key"], {}
        for folder in ("calibration", "interventions"):
            p = run / folder / (key + ".npz")
            assert sha(p) == read(p.with_suffix(".json"))["sha256"]
            with np.load(p) as z:
                data.update({n: z[n] for n in z.files})
        for alias in "AB":
            sign = 1 if alias == "A" else -1
            target = sign * (delta(data, "KV", alias, "source") - delta(data, "N", alias, "source"))
            for name in ("source", "write", "mlp"):
                base = delta(data, "N", alias, name)
                k = delta(data, "K", alias, name) - base
                v = delta(data, "V", alias, name) - base
                joint = delta(data, "KV", alias, name) - base - k - v
                component_target = delta(data, "KV", alias, name) - base
                closure.append(float(np.max(abs(k + v + joint - component_target))))
                ct = statistics(joint, component_target)
                rows.append(
                    dict(
                        key=key,
                        alias=alias,
                        component=name,
                        joint_component_norm_ratio=float(
                            np.linalg.norm(joint) / max(np.linalg.norm(component_target), 1e-30)
                        ),
                        joint_component_projection=ct["signed_target_fraction"],
                        **statistics(sign * joint, target),
                    )
                )
    table(out / "joint.csv", rows)
    save_json(
        out / "provenance.json",
        dict(
            posthoc=True,
            threshold_or_primary_changed=False,
            model_calls=0,
            script_sha256=sha(ROOT / "research/boolean_coupling/joint302.py"),
            signed_identity_max_error=max(closure),
            boundary="Defined finite corner interaction, not a learned feature or a direct interaction-removal intervention.",
            output_sha256=sha(out / "joint.csv"),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.run.resolve(), a.out.resolve())
