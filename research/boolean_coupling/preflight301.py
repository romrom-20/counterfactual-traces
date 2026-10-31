"""Unequal-length equivalent-alias routing fixtures and coherent corruption tests."""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from .alias301 import ROOT, SOURCES, execute, final_scores, sites
from .audit229 import sha, softmax
from .audit301 import bundle, native_sources, populations, scientific
from .cache272 import save_json


def algebra():
    """Known causal response tables test both reference backgrounds and signs."""
    outcomes = []
    for account in ("source", "remaining", "joint"):
        f, n = {}, {}
        d = {"AA": np.ones((3, 5)), "BB": np.full((3, 5), 2.0)}
        c = {"AA": 1.0, "BB": 2.0}
        for h in ("AB", "BA"):
            target = (
                ("BB" if (h == "AB") == (account == "source") else "AA")
                if account != "joint"
                else "AA"
            )
            d[h] = d[target]
            c[h] = c[target]
        d["RA"] = d["AA"]
        c["RA"] = c["AA"]
        for alias, label, center in [("A", "AA", 0.7), ("B", "BB", 1.3)]:
            f[alias + "_F_final"] = np.full((3, 5), center)
            n[alias + "_T_final"] = f[alias + "_F_final"] + d[label]
            f[alias + "_F_scores"] = np.array([0.0, center])
            n[alias + "_T_scores"] = np.array([0.0, center + c[label]])
        for i, h in enumerate(("AB", "BA", "RA")):
            center = 2.0 + i
            f[h + "_F_final"] = np.full((3, 5), center)
            f[h + "_T_final"] = f[h + "_F_final"] + d[h]
            f[h + "_F_scores"] = np.array([0.0, center])
            f[h + "_T_scores"] = np.array([0.0, center + c[h]])
        for alias, h in [("A", "BA"), ("B", "AB")]:
            f["P_" + alias + "_final"] = f[alias + "_F_final"] + d[h]
            f["P_" + alias + "_scores"] = np.array([0.0, f[alias + "_F_scores"][1] + c[h]])
        cohort = []
        for i in range(32):
            case = dict(key=str(i), group=dict(block=str(i % 4), mention_order=(i // 4) % 2))
            rows, _, _, _ = scientific(case, f, n)
            cohort.extend(rows)
            for r in rows:
                assert r["passes"] == (r["account"] == account), (account, r)
        pop = populations(cohort)
        assert pop["source"]["passes"] == (32 if account == "source" else 0)
        assert pop["remaining"]["passes"] == (32 if account == "remaining" else 0)
        outcomes.append(account)
    return outcomes


def main(out):
    assert not out.exists()
    out.mkdir()
    arithmetic = algebra()
    torch.manual_seed(301)
    torch.set_num_threads(2)
    model = (
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=1024,
                hidden_size=16,
                intermediate_size=23,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                tie_word_embeddings=False,
                attention_dropout=0.0,
                _attn_implementation="eager",
            )
        )
        .float()
        .eval()
    )
    model.requires_grad_(False)
    w = dict(
        norm=model.model.norm.weight.detach().numpy().astype(float),
        eps=np.array(model.config.rms_norm_eps),
        head=model.lm_head.weight.detach().numpy()[[895, 830]].astype(float),
    )
    maps = {
        str(j): model.model.layers[j].self_attn.o_proj.weight.detach().numpy() for j in sites(model)
    }
    rng = np.random.default_rng(301)
    checks, raw, rejected = [], [], []
    total = 0
    counts = []
    for i, length in enumerate((12, 15, 13, 17)):
        a = rng.integers(5, 400, length).tolist()
        pos = length - 5
        begin = pos - 3
        a[pos] = 1
        b = a.copy()
        b[begin : begin + 2] = [401, 402]
        at = a.copy()
        bt = b.copy()
        at[pos] = bt[pos] = 2
        case = dict(
            key="tiny301-" + str(i),
            position=pos,
            alias_start=begin,
            alias_changed=[begin, begin + 1],
            aliases=dict(A=dict(false_ids=a, true_ids=at), B=dict(false_ids=b, true_ids=bt)),
            group=dict(block=str(i), mention_order=i % 2, reverse=0, target=0),
        )
        f, fw, fc, used, fm = execute(model, case, "forecast")
        fm["checks"] = fc
        n, nw, nc, nused, nm = execute(model, case, "native", f, fm)
        nm["checks"] = nc
        total += used["full_forwards"] + nused["full_forwards"]
        counts.append(dict(forecast=used, native=nused))
        bundle(case, f, fw, fm, w, maps, "forecast", checks, raw, f)
        bundle(case, n, nw, nm, w, maps, "native", checks, raw, f)
        native_sources(case, f, n, fm, nm, checks)
        rr, vv, nn, cc = scientific(case, f, n)
        assert len(rr) == 3 and len(nn) == 2 and len(vv) == 16 and len(cc) == 5
        np.savez_compressed(
            out / f"fixture-{i}.npz",
            **{"forecast_" + k: v for k, v in f.items()},
            **{"native_" + k: v for k, v in n.items()},
        )
        # A coherent wrong source-prefix donor must fail the route identity,
        # even when its resulting probability, PV and projection are consistent.
        bad = {k: v.copy() for k, v in fw.items()}
        prefix = "AB_F_1_"
        for kind in ("k", "v"):
            bad[prefix + "used_" + kind][:, begin:pos] = f[f"bank_A_{kind}1"]
        q = bad[prefix + "query"].astype(float)
        k = np.repeat(bad[prefix + "used_k"].astype(float), int(bad[prefix + "groups"]), axis=0)
        v = np.repeat(bad[prefix + "used_v"].astype(float), int(bad[prefix + "groups"]), axis=0)
        r = q @ k.swapaxes(-1, -2) * float(bad[prefix + "scaling"])
        p = softmax(r + bad[prefix + "mask"])
        y = p @ v
        bad[prefix + "raw"] = r
        bad[prefix + "prob"] = p
        bad[prefix + "head_output"] = y
        bad[prefix + "source_write"] = (y.transpose(1, 0, 2).reshape(1, -1) @ maps["1"].T)[0]
        ck = []
        bundle(case, f, bad, fm, w, maps, "forecast", ck, [], f)
        assert any(not r["passes"] for r in ck), "Coherent wrong prefix donor accepted"
        rejected.append("coherent_wrong_donor-" + str(i))
        bad = {k: v.copy() for k, v in f.items()}
        bad["P_B_final"] = (
            f["A_F_final"].astype(float)
            + f["AB_T_final"].astype(float)
            - f["AB_F_final"].astype(float)
        ).astype(np.float32)
        bad["P_B_scores"] = final_scores(model, bad["P_B_final"], length, pos)
        ck = []
        bundle(case, bad, fw, fm, w, maps, "forecast", ck, [], f)
        assert any(not r["passes"] for r in ck), "Coherent wrong False baseline accepted"
        rejected.append("coherent_wrong_baseline-" + str(i))
        wrong = dict(
            nm,
            source_hashes={
                **nm["source_hashes"],
                "B_T": ["corrupt"] + nm["source_hashes"]["B_T"][1:],
            },
        )
        ck = []
        native_sources(case, f, n, fm, wrong, ck)
        assert any(not r["passes"] for r in ck)
        rejected.append("source_signature-" + str(i))
    save_json(out / "checks.json", dict(checks=checks, raw=raw))
    failed = [r for r in checks if not r["passes"]]
    assert not failed, failed[:20]
    save_json(out / "progress-0001.json", dict(cases=1))
    try:
        save_json(out / "progress-0001.json", dict(cases=999))
    except AssertionError:
        rejected.append("immutable_progress")
    else:
        raise AssertionError("Immutable writer accepted overwrite")
    save_json(
        out / "qualification.json",
        dict(
            passed=True,
            synthetic=True,
            checkpoint_loaded=False,
            required_checks=len(checks),
            failed_checks=0,
            raw_checks=len(raw),
            raw_failed=sum(not r["passes"] for r in raw),
            full_forwards=total,
            counts=counts,
            unequal_lengths=[12, 15, 13, 17],
            rejected=rejected,
            wrong_baseline_extra_heads=4,
            known_causal_tables=arithmetic,
            source_hashes={n: sha(ROOT / n) for n in SOURCES},
        ),
    )
    print(dict(passed=True, checks=len(checks), tiny_forwards=total, rejected=len(rejected)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out.resolve())
