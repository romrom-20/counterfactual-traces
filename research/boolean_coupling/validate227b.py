"""Post-run numerical qualification of finite norm/gate/up values, not derivatives."""

import argparse
import hashlib
import json
from pathlib import Path
from time import time

import numpy as np
from safetensors import safe_open


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(out, checkpoint):
    seal = out / "finite-projection-validation-seal.json"
    target = out / "finite-projection-validation.json"
    assert not seal.exists() and not target.exists()
    settings = json.loads((Path(__file__).parent / "settings.json").read_text())
    assert checkpoint.parent.name == settings["model"]["revision"]
    config = json.loads((checkpoint.parent / "config.json").read_text())
    files = sorted((out / "cases").glob("*.npz"))
    assert len(files)==32
    names = dict(norm="model.layers.23.post_attention_layernorm.weight",
                 gate="model.layers.23.mlp.gate_proj.weight",
                 up="model.layers.23.mlp.up_proj.weight")
    with safe_open(checkpoint, framework="pt", device="cpu") as weights:
        arrays = {k:weights.get_tensor(v).float().numpy().astype(float) for k,v in names.items()}
    metadata = dict(created_unix=time(), retrospective_validation=True,
        source_sha256=sha(__file__), checkpoint=str(checkpoint), revision=settings["model"]["revision"],
        tensor_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()},
        inputs={str(p):sha(p) for p in files}, coordinate_bound=1e-4,
        derivative_claim=False)
    seal.write_text(json.dumps(metadata,indent=2)+"\n")
    maximum, checks = dict(norm=0.,gate=0.,up=0.), dict(norm=0,gate=0,up=0)
    for p in files:
        q = dict(np.load(p))
        for name in ("zero", "b", "d", "both", "add"):
            x = q[name+"_mid"].astype(float)
            normalized = x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+config["rms_norm_eps"])*arrays["norm"]
            error = float(np.max(abs(normalized-q[name+"_norm"])))
            assert error<=1e-4, (p.name,name,"norm",error)
            maximum["norm"] = max(maximum["norm"],error)
            checks["norm"] += 1
        for name in ("zero", "b", "d", "both", "add", "X", "Y"):
            normalized = q[name+"_norm"].astype(float)
            for field in ("gate", "up"):
                reconstructed = normalized @ arrays[field].T
                error = float(np.max(abs(reconstructed-q[name+"_"+field])))
                assert error<=1e-4, (p.name,name,field,error)
                maximum[field] = max(maximum[field],error)
                checks[field] += 1
    assert sha(__file__)==metadata["source_sha256"]
    assert all(sha(p)==h for p,h in metadata["inputs"].items())
    result = dict(status="passed", checks=checks, maximum_errors=maximum,
        bound=1e-4, model_calls=0, derivative_claim=False, seal_sha256=sha(seal))
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    args = parser.parse_args()
    main(args.out.resolve(),args.checkpoint.absolute())
