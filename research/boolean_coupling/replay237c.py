"""Same-backend full-sequence finite-MLP replay; does not repair raw NumPy failures."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .core225b import array, load_model, observe, sha

MODES = ("legacy", "1001", "1101", "1011", "1111", "public")


def main(run):
    assert not (run/"public-mlp-replay.json").exists()
    design, status = (json.loads((run/n).read_text()) for n in ("design.json", "status.json"))
    assert (run/"failure-audit.json").exists() and status["status"] == "complete"
    old = Path(design["input228"])
    root = Path(__file__).resolve().parents[2]
    model = load_model(True)
    checks, counts = [], dict(ordinary_false=0, ordinary_true=0, source_attention=0, finite_mlps=0)
    with torch.no_grad():
        for number, case in enumerate(json.loads((root/"research/boolean_coupling/cases228.json").read_text()), 1):
            key, start = case["key"], case["position"]+1
            raw = run/"cases"/(key+".npz")
            assert sha(raw) == json.loads(raw.with_suffix(".json").read_text())["arrays_sha256"]
            f, data = (dict(np.load(p)) for p in (old/"forecasts"/raw.name, raw))
            states, context, _ = observe(model, torch.tensor([case["false_ids"]], device="mps"))
            counts["ordinary_false"] += 1
            parents = {}
            for i in range(19, 24):
                source = states[i-1].clone()
                source[:, start-1] = torch.tensor(f["source_rows"][i-1][None], device="mps")
                layer = model.model.layers[i]
                parents[i] = source+layer.self_attn(layer.input_layernorm(source), attention_mask=context["mask"],
                    position_embeddings=context["rope"])[0]
                counts["source_attention"] += 1
            for regime in ("forecast", "oracle"):
                for mode in MODES:
                    arm = regime+"_"+mode
                    for i in range(19, 24):
                        parent = parents[i].clone()
                        parent[:, start:] = torch.tensor(data[f"{arm}_m{i}_parent"][None], device="mps", dtype=torch.float32)
                        layer = model.model.layers[i]
                        result = parent+layer.mlp(layer.post_attention_layernorm(parent))
                        error = float(np.max(abs(array(result[0, start:]).astype(float)-data[f"{arm}_m{i}_output"])))
                        checks.append(dict(case=key, arm=arm, layer=i+1, error=error, bound=1e-4, passed=error <= 1e-4))
                        assert np.isfinite(error), checks[-1]
                        counts["finite_mlps"] += 1
            print(f"public MLP replay {number}/32", flush=True)
    failed = [c for c in checks if not c["passed"]]
    out = dict(status="passed" if not failed else "failed", same_backend_replay=True, independent_mathematical_reconstruction=False,
        raw_numpy_failures_reclassified=False, source_sha256=sha(__file__), protocol_sha256=sha(root/"docs/experiments/237c-complete-numerical-diagnostic.md"),
        original_design_sha256=sha(run/"design.json"), revision=model.config._commit_hash, checks=checks, counts=counts,
        failed_checks=failed, bit_exact=sum(c["error"] == 0 for c in checks))
    (run/"public-mlp-replay.json").write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps({k:v for k, v in out.items() if k not in ("checks", "failed_checks")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run.resolve())
