"""Separate NumPy audit of the saved computation and frozen225 action rules."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def read_arrays(path):
    record = json.loads(path.with_suffix(".json").read_text())
    assert sha(path) == record["arrays_sha256"]
    assert all(c["error"] <= c["bound"] for c in record["checks"])
    return dict(np.load(path)), record


def margin(scores):
    return float(np.float64(scores[1]) - np.float64(scores[0]))


def validate_values(values, prefix, site, weights):
    get = lambda name: values[prefix+name].astype(np.float64)
    product = get("product")
    down = weights[str(site)+"_down"].astype(np.float64)
    projection = float(np.max(abs(product @ down.T - get("write"))))
    assert projection <= 1e-4
    final = get("final")
    normalized = final[-1] / np.sqrt(np.mean(final[-1]**2) + float(weights["eps"]))
    scores = (normalized * weights["norm"].astype(np.float64)) @ weights["head"].astype(np.float64).T
    head = float(np.max(abs(scores-get("scores"))))
    assert head <= .005 and np.max(abs(final-get("trajectory")[-1])) == 0
    return dict(down=projection, head=head)


def qualification(out):
    target = out / "qualification-audit.json"
    assert not target.exists() and not (out / "qualification-seal.json").exists()
    design = json.loads((out / "design.json").read_text())
    assert all(sha(ROOT/p) == h for p, h in design["source_hashes"].items())
    status = json.loads((out / "qualification-status.json").read_text())
    assert status["status"] == "complete" and status["counts"] == design["expected_qualify"]
    weights = dict(np.load(out / "weights.npz"))
    errors, controls, maximum_ad, maximum_fd = [], 0, 0., 0.
    files = {name: sha(out/name) for name in ("design.json", "runtime.json", "weights.npz", "qualification-status.json")}
    for key in design["contexts"]:
        path = out / "qualification" / (key+".npz")
        q, record = read_arrays(path)
        controls += len(record["checks"])
        files[str(path.relative_to(out))] = sha(path)
        files[str(path.with_suffix(".json").relative_to(out))] = sha(path.with_suffix(".json"))
        start = record["position"] + 1
        for job in record["jobs"]:
            site, prefix = job["site"], job["prefix"]
            c = {k[len(f"c{site}_"):]: v.astype(np.float64) for k, v in q.items() if k.startswith(f"c{site}_")}
            get = lambda name: q[prefix+name].astype(np.float64)
            slope = c["sigma"] * (1+c["gate"]*(1-c["sigma"]))
            dg = get("gate")-c["gate"]
            affine = c["phi"]+slope*dg
            if job["silu"]:
                expected_phi = get("gate") * np.exp(-np.logaddexp(0., -get("gate")))
                assert np.max(abs(expected_phi-get("phi"))) <= 1e-4
            else:
                assert np.max(abs(affine.astype(np.float32)-get("phi"))) == 0
            analytic = dict(silu=slope*dg,
                            product=(get("phi")-c["phi"])*c["up"]+c["phi"]*(get("up")-c["up"]))
            for node in ("silu", "product"):
                assert np.max(abs(analytic[node]-get(node+"_analytic"))) <= 1e-9
                ad = float(np.max(abs(analytic[node]-get(node+"_dual"))))
                fd = float(np.max(abs(analytic[node]-get(node+"_fd"))))
                assert ad <= 1e-4 and fd <= 1e-4
                maximum_ad, maximum_fd = max(maximum_ad, ad), max(maximum_fd, fd)
            if job["product"]:
                expected_product = get("phi").astype(np.float32)*get("up").astype(np.float32)
                assert np.max(abs(expected_product-get("product"))) <= 1e-4
            else:
                expected_product = c["product"]+analytic["product"]
                assert np.max(abs(expected_product.astype(np.float32)-get("product"))) == 0
            errors.append(validate_values(q, prefix, site, weights))
            if job["name"] == "s11":
                assert np.max(abs(get("trajectory")-q["native_tail"][site-22:, start:])) <= 1e-4
                assert np.max(abs(get("scores")-q["native_scores"])) <= .005
    assert len(errors) == 256
    result = dict(qualified=True, contexts=32, controls=controls, counts=status["counts"],
                  maximum_ad=maximum_ad, maximum_fd=maximum_fd,
                  maxima={k: max(e[k] for e in errors) for k in errors[0]},
                  retrospective=True, old_wholeblock_jacobian_qualified=False,
                  source_sha256=sha(Path(__file__)))
    write(target, result)
    files[target.name] = sha(target)
    write(out/"qualification-seal.json", dict(files=files, intervention_calls_before_seal=0))
    print(json.dumps(result, indent=2))


def interventions(out):
    target = out / "audit.json"
    assert not target.exists()
    design = json.loads((out / "design.json").read_text())
    assert all(sha(ROOT/p) == h for p, h in design["source_hashes"].items())
    seal = json.loads((out/"qualification-seal.json").read_text())
    assert seal["intervention_calls_before_seal"] == 0 and all(sha(out/p) == h for p, h in seal["files"].items())
    status = json.loads((out/"status.json").read_text())
    assert status["status"] == "complete" and status["counts"] == design["expected_intervene"]
    weights = dict(np.load(out/"weights.npz"))
    rows, errors, controls = [], [], 0
    for key in design["contexts"]:
        qp = out/"qualification"/(key+".npz")
        q, _ = read_arrays(qp)
        z, record = read_arrays(out/"interventions"/qp.name)
        assert sha(qp) == record["qualification_sha256"]
        controls += len(record["checks"])
        for site in (22, 23):
            pq = lambda name: q[f"s{site}_{name}_product"].astype(np.float64)
            p0 = pq("b0")
            a, b = pq("s00")-p0, pq("b1")-p0
            k = pq("s11")-pq("b1")-pq("s00")+p0
            random = {name: z[f"s{site}_random{name}"] for name in ("B", "K")}
            for name, v in (("A", a), ("B", b), ("K", k)):
                assert np.max(abs(z[f"s{site}_{name}"]-v)) <= 1e-9
            for name, v in (("B", b), ("K", k)):
                seed = int(hashlib.sha256((key+str(site)+name+"225").encode()).hexdigest()[:16], 16)
                unit = np.random.default_rng(seed).standard_normal(v.shape)
                expected_random = unit/np.linalg.norm(unit)*np.linalg.norm(v)
                assert np.max(abs(random[name]-expected_random)) <= 1e-9
                assert abs(np.linalg.norm(random[name])-np.linalg.norm(v)) <= 1e-9
            recipes = dict(B_only=p0+b+a, K_only=p0+a+k,
                           random_K_add=p0+a+random["K"], random_K_remove=pq("s11")-random["K"],
                           random_B_base=p0+random["B"], random_B_0=p0+a+random["B"],
                           random_B_1=p0+a+k+random["B"])
            for name, proposed in recipes.items():
                prefix = f"s{site}_{name}_"
                assert np.max(abs(z[prefix+"proposed"]-proposed)) <= 1e-9
                assert np.max(abs(z[prefix+"product"]-proposed.astype(np.float32))) == 0
                errors.append(validate_values(z, prefix, site, weights))
            m = lambda data, name: margin(data[f"s{site}_{name}_scores"])
            e00 = m(q, "s00")-m(q, "b0")
            e11 = m(q, "s11")-m(q, "b1")
            e10 = m(z, "B_only")-m(q, "b1")
            e01 = m(z, "K_only")-m(q, "b0")
            rk0 = m(z, "random_K_add")-m(q, "b0")-e00
            rk1 = m(z, "random_K_remove")-m(q, "b1")-e11
            rb0 = m(z, "random_B_0")-m(z, "random_B_base")-e00
            rb1 = m(z, "random_B_1")-m(z, "random_B_base")-e01
            guarded = e00*e11 < 0 and min(abs(e00), abs(e11)) > .02
            k_match = max(abs(e01-e11), abs(e10-e00)) <= .1
            b_match = max(abs(e10-e11), abs(e01-e00)) <= .1
            k_specific = abs(e01-e00)>abs(rk0) and abs(e11-e10)>abs(rk1)
            b_specific = abs(e10-e00)>abs(rb0) and abs(e11-e01)>abs(rb1)
            k_sign = e01*e11>0 and e10*e00>0
            b_sign = e10*e11>0 and e01*e00>0
            rows.append(dict(context=key, group=record["group"], site=site,
                             E00=e00, E10=e10, E01=e01, E11=e11,
                             random_K0=rk0, random_K1=rk1, random_B0=rb0, random_B1=rb1,
                             guarded=bool(guarded), K_match=bool(k_match), B_match=bool(b_match),
                             K_specific=bool(k_specific), B_specific=bool(b_specific),
                             K_account=bool(guarded and k_match and k_specific and k_sign),
                             B_account=bool(guarded and b_match and b_specific and b_sign),
                             K_only_reverses=bool(e01*e00<0), B_only_reverses=bool(e10*e00<0)))
    assert len(rows) == 64 and len(errors) == 448
    gates = []
    for account in ("K_account", "B_account"):
        primary = [r for r in rows if r["site"] == 23]
        strata = []
        for schema, order in itertools.product(("generic", "descriptive"), (0, 1)):
            rs = [r for r in primary if r["group"]["schema"] == schema and r["group"]["reverse"] == order]
            assert len(rs) == 8
            strata.append(dict(schema=schema, order=order, passed=sum(r[account] for r in rs)))
        n = sum(r[account] for r in primary)
        gates.append(dict(account=account, count=n, strata=strata,
                          passed=bool(n>=24 and all(s["passed"]>=6 for s in strata))))
    result = dict(qualified=True, contexts=32, rows=rows, gates=gates, controls=controls,
                  counts=status["counts"], maxima={k: max(e[k] for e in errors) for k in errors[0]},
                  retrospective=True, prospective=False, learned_circuit_identified=False,
                  full_question_B_solved=False, source_sha256=sha(Path(__file__)))
    write(target, result)
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("qualify", "interventions"), required=True)
    parser.add_argument("--out", type=Path, default=ROOT/"runs/baseline-interaction-225b")
    args = parser.parse_args()
    qualification(args.out.resolve()) if args.phase == "qualify" else interventions(args.out.resolve())


if __name__ == "__main__":
    main()
