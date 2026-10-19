"""Descriptive comparison of native-input remainders and own-input corrections in238."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main(run):
    audit = json.loads((run/"audit.json").read_text())
    assert audit["status"] in ("passed", "passed_declared_domain_with_raw_bridge_failures")
    assert not (run/"native-own-remainders.csv").exists()
    old = Path(json.loads((run/"design.json").read_text())["input228"])
    selected = json.loads((run/"selection.json").read_text())["selected"]
    rows = []
    for entry in json.loads((run/"status.json").read_text())["cases"]:
        key = entry["key"]
        meta = json.loads((run/"cases"/(key+".json")).read_text())
        ledger = dict(np.load(run/"ledger"/(key+".npz")))
        data = dict(np.load(run/"cases"/(key+".npz")))
        for n in selected:
            r = ledger[f"n{n}_remainder"]
            for arm in ("own4", "random4"):
                label = f"{arm}_n{n}"
                own = data[label+"_finite_output"].astype(float)-data[label+"_linear"]
                for scope in ("all", "verdict"):
                    pick = slice(None) if scope == "all" else -1
                    a, b = r[pick], own[pick]
                    na, nb = np.linalg.norm(a), np.linalg.norm(b)
                    rows.append(dict(key=key, **meta["group"], node=n, layer=n//2+1,
                        kind="attention" if n % 2 == 0 else "mlp", arm=arm, scope=scope,
                        native_norm=float(na), own_norm=float(nb), norm_ratio=float(nb/na),
                        relative_error=float(np.linalg.norm(b-a)/na),
                        cosine=float(np.vdot(a, b)/(na*nb)) if na*nb else 0.,
                        signed_share=float(np.vdot(b, a)/(na*na))))
    with (run/"native-own-remainders.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(descriptive_only=True, new_model_calls=0, unused_native_input_path=str(old),
        medians={str(n):{arm:{scope:{name:float(np.median([r[name] for r in rows if r["node"] == n and r["arm"] == arm
            and r["scope"] == scope])) for name in ("native_norm", "own_norm", "norm_ratio", "relative_error", "cosine", "signed_share")}
            for scope in ("all", "verdict")} for arm in ("own4", "random4")} for n in selected})
    (run/"native-own-summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run.resolve())
