"""Numerical repetition of published277 literals; NEVER a new fresh confirmation."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .audit229 import sha
from .audit272 import margin, read, table, vector
from .cache272 import digest, save_json
from .core225b import array, load_model, observe
from .run272 import stamp
from .run277 import validate
from .source277 import ARMS, First


def main(run):
    design = validate(run)
    status, qualification = read(run / "forecast-status.json"), read(run / "forecast-audit.json")
    assert status["status"] == "complete" and qualification["status"] == "passed"
    assert qualification["ordinary_true"] == 0 and qualification["status_sha256"] == sha(run / "forecast-status.json")
    assert not any((run / n).exists() for n in ("repetition-seal.json", "native-started.json", "observations", "repetition-observations"))
    for entry in status["cases"]:
        p = run / "forecasts" / (entry["key"] + ".npz")
        assert sha(p) == entry["arrays_sha256"] and sha(p.with_suffix(".json")) == entry["metadata_sha256"]
    save_json(run / "repetition-seal.json", dict(created=stamp(), fresh_confirmation=False,
        scope="Published-literal numerical repetition only; no new public forecast receipt or independent confirmation.",
        design_sha256=sha(run / "design.json"), forecast_status_sha256=sha(run / "forecast-status.json"),
        forecast_audit_sha256=sha(run / "forecast-audit.json"), cases=status["cases"]))
    (run / "repetition-observations").mkdir()
    model = load_model(True)
    bank, native = {}, []
    for case in design["cases"]:
        key, start = case["key"], case["position"]
        with First(model, case) as capture:
            states, _, scores = observe(model, torch.tensor([case["true_ids"]], device="mps"))
        with np.load(run / "forecasts" / (key + ".npz")) as f:
            for name in ("before", "norm", "q", "k", "v", "full_k", "full_v", "p", "mask", "scaling", "y", "attwrite", "mid"):
                assert np.array_equal(capture.data[name], f["prepared_" + name])
            assert [digest(array(s[0, :start])) for s in states] == read(run / "forecasts" / (key + ".json"))["false_prefix_hashes"]
            t = dict(source=array(states[0][0, start]).astype(float), final=array(states[-1][0, start:]).astype(float), margin=margin(array(scores)))
            own = {"T": t, **{a: dict(source=f[a + "_source"][0].astype(float), final=f[a + "_final"].astype(float), margin=margin(f[a + "_scores"])) for a in ("F", *ARMS)}}
            bank[key] = own
            for a in ARMS[:-1]:
                row = dict(key=key, **case["group"], arm=a)
                good = []
                for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                    v = vector((own[a]["final"] - own["F"]["final"])[pick], (t["final"] - own["F"]["final"])[pick])
                    row.update({scope + "_" + k: x for k, x in v.items()})
                    good.append(v["norm"] >= .001 and v["relative"] <= .05 and v["cosine"] >= .99)
                err = abs(own[a]["margin"] - t["margin"])
                row.update(margin_error=err, passes=bool(all(good) and err <= .1))
                native.append(row)
            p = run / "repetition-observations" / (key + ".npz")
            np.savez_compressed(p, T_source=np.stack([array(s[0, start]) for s in states]), T_final=t["final"].astype(np.float32), T_scores=array(scores))
        print("277 published-literal repetition", len(bank), "/32", flush=True)
    paired, combined = [], []
    cases = {c["key"]: c for c in design["cases"]}
    for key, own in bank.items():
        other = bank[design["partners"][key]]
        ts = (own["T"]["source"] - own["F"]["source"]) - (other["T"]["source"] - other["F"]["source"])
        tv = (own["T"]["final"][-1] - own["F"]["final"][-1]) - (other["T"]["final"][-1] - other["F"]["final"][-1])
        tm = (own["T"]["margin"] - own["F"]["margin"]) - (other["T"]["margin"] - other["F"]["margin"])
        for a in ARMS[:-1]:
            s = vector((own[a]["source"] - own["F"]["source"]) - (other[a]["source"] - other["F"]["source"]), ts)
            v = vector((own[a]["final"][-1] - own["F"]["final"][-1]) - (other[a]["final"][-1] - other["F"]["final"][-1]), tv)
            m = (own[a]["margin"] - own["F"]["margin"]) - (other[a]["margin"] - other["F"]["margin"])
            ok = all(x["norm"] >= .001 and x["relative"] <= .25 and x["cosine"] >= .95 for x in (s, v)) and abs(tm) >= .02 and abs(m - tm) <= .005 + .25 * abs(tm)
            row = dict(key=key, **cases[key]["group"], arm=a, **{"source_" + k: x for k, x in s.items()}, **{"verdict_" + k: x for k, x in v.items()}, target_margin=tm, margin_error=abs(m - tm), passes=bool(ok))
            paired.append(row)
            nr = next(r for r in native if r["key"] == key and r["arm"] == a)
            combined.append(dict(key=key, schema=row["schema"], reverse=row["reverse"], arm=a, passes=bool(nr["passes"] and ok)))
    for name, records in dict(native=native, paired=paired, combined=combined).items():
        table(run / ("repetition-" + name + ".csv"), records)
    nc = {a: sum(r["passes"] for r in native if r["arm"] == a) for a in ARMS[:-1]}
    cc = {a: sum(r["passes"] for r in combined if r["arm"] == a) for a in ARMS[:-1]}
    save_json(run / "repetition-audit.json", dict(status="passed", experiment=277, fresh_confirmation=False,
        ordinary_true=32, native_counts=nc, combined_counts=cc, seal_sha256=sha(run / "repetition-seal.json"),
        csv_sha256={p.name: sha(p) for p in run.glob("repetition-*.csv")},
        boundary="Repetition of already published literals; no public pre-native receipt, new independent support, full stronger-gain recertification or old-failure repair."))
    print(dict(published_literal_repetition=True, fresh_confirmation=False, native=nc, combined=cc), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    main(args.run.resolve())
