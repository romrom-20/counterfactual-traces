"""Post-observation composite comparison and normalization/product identity; no model."""

import argparse
import json
from pathlib import Path

import numpy as np

from .audit229 import build


def main(run):
    assert not (run/"composite-description.json").exists()
    design = json.loads((run/"design.json").read_text())
    previous = Path(design["input_root"])
    status = json.loads((run/"status.json").read_text())
    weights = {k:v.astype(float) for k, v in dict(np.load(previous/"weights.npz")).items()}
    scores = {f"{i:04b}":[] for i in range(16)}
    rows, maximum = [], 0.
    for ent in status["cases"]:
        data = dict(np.load(run/"cases"/(ent["key"]+".npz")))
        old = {k:v.astype(float) for k, v in dict(np.load(previous/"cases"/(ent["key"]+".npz"))).items()}
        meta = json.loads((run/"cases"/(ent["key"]+".json")).read_text())
        start = meta["start"]
        target = data["old_R"][-1]
        for bits in scores:
            value = (data["public_"+bits].astype(float)-data["public_0000"].astype(float))[-1]
            rel = float(np.linalg.norm(value-target)/np.linalg.norm(target))
            cos = float(np.vdot(value, target)/(np.linalg.norm(value)*np.linalg.norm(target))) if np.linalg.norm(value) else 0.
            scores[bits].append(dict(relative=rel, cosine=cos, meets_old_bounds=rel <= .25 and cos >= .95))
        original = build(weights, old, old["x"])
        dx = old["receiver"]-old["x"]
        dz = weights["attn_norm"]*(dx/original["rms"]-old["x"]*np.mean(old["x"]*dx, -1, keepdims=True)/original["rms"]**3)
        eta = np.zeros_like(dz)
        eta[start:] = data["norm_curvature"]
        width, groups = int(weights["width"]), int(weights["groups"])

        def changes(z):
            def projection(name):
                v = (z@weights[name+"_weight"].T).reshape(len(z), -1, width).transpose(1, 0, 2)
                if name != "v":
                    a, b = np.split(v, 2, -1)
                    v = v*old["rope_cos"][0]+np.concatenate((-b, a), -1)*old["rope_sin"][0]
                return np.repeat(v, groups, 0) if name != "q" else v
            q, k, v = (projection(n) for n in ("q", "k", "v"))
            dl = (q@original["k"].transpose(0, 2, 1)+original["q"]@k.transpose(0, 2, 1))*weights["scaling"]
            dp = original["p"]*(dl-np.sum(original["p"]*dl, -1, keepdims=True))
            return dp, v

        pa, va = changes(dz)
        pb, vb = changes(eta)
        contexts = dict(linear_weight_curved_value=pa@vb, curved_weight_linear_value=pb@va,
                        curved_weight_curved_value=pb@vb)
        vectors = {k:(v.transpose(1, 0, 2).reshape(dz.shape)@weights["o_weight"].T)[start:] for k, v in contexts.items()}
        measured = data["smooth_1001"]-data["smooth_1000"]-data["smooth_0001"]+data["smooth_0000"]
        error = float(np.max(abs(sum(vectors.values())-measured)))
        assert error <= 1e-9, error
        maximum = max(maximum, error)
        missing = data["old_R"]-(data["smooth_1000"]-data["smooth_0000"])
        rows.append(dict(key=ent["key"], group=meta["group"], identity_error=error,
            mixed_term_norm=float(np.linalg.norm(measured[-1])),
            mixed_signed_share_of_norm_only_error=float(np.vdot(measured[-1], missing[-1])/np.vdot(missing[-1], missing[-1])),
            mixed_remaining_error_ratio=float(np.linalg.norm((missing-measured)[-1])/np.linalg.norm(missing[-1])),
            **{k+"_signed_share_of_M":float(np.vdot(v[-1], measured[-1])/np.vdot(measured[-1], measured[-1])) for k, v in vectors.items()}))
    result = dict(explicitly_post_observation=True, model_calls=0,
        note="Composite bounds are descriptive reuse of old bounds, not registered single-primitive successes. "
             "The three-term product split is an algebraic identity, not three independently established mechanisms.",
        composites={k:dict(count=sum(r["meets_old_bounds"] for r in v),
            median_relative=float(np.median([r["relative"] for r in v])),
            maximum_relative=max(r["relative"] for r in v), median_cosine=float(np.median([r["cosine"] for r in v]))) for k, v in scores.items()},
        mixed_identity_maximum=maximum, cases=rows,
        mixed_medians={k:float(np.median([r[k] for r in rows])) for k in rows[0] if k not in ("key", "group")})
    (run/"composite-description.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    main(parser.parse_args().run.resolve())
