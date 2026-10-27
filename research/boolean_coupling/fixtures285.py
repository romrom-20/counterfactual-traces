"""Tokenization-only structural prefixes and a forecast-only historical donor index."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LAYOUTS = ("table", "user_rule", "dialogue", "xml")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def render(layout, system, task, assistant):
    def message(role, text):
        return "<|im_start|>" + role + "\n" + text + "<|im_end|>\n"

    if layout == "table":
        passage = task["passage"].split(". ")
        passage = [fact.rstrip(".") + "." for fact in passage]
        user = (
            "Evidence table (each row states one fact):\n"
            + "\n".join(f"row {i + 1} | {fact}" for i, fact in enumerate(passage))
            + "\nQuestion | "
            + task["question"]
            + "\nProposed answer | "
            + task["proposed_answer"]
        )
        prefix = message("system", system) + message("user", user)
    elif layout == "user_rule":
        user = (
            "Verification procedure:\n"
            + system
            + "\n\nApply that procedure to this record:\n"
            + json.dumps(task, sort_keys=True)
        )
        prefix = message("system", "Follow the user's verification procedure.") + message(
            "user", user
        )
    elif layout == "dialogue":
        prefix = (
            message("system", system)
            + message("user", "I will provide one evidence record. Wait for it before judging.")
            + message("assistant", "Ready for the evidence record.")
            + message("user", json.dumps(task, sort_keys=True))
        )
    elif layout == "xml":
        user = (
            "<record>\n<question>" + task["question"] + "</question>\n"
            "<proposed_answer>" + task["proposed_answer"] + "</proposed_answer>\n"
            "<passage>" + task["passage"] + "</passage>\n</record>"
        )
        prefix = message("system", system) + message("user", user)
    else:
        raise ValueError(layout)
    return prefix + "<|im_start|>assistant\n" + assistant


def main(reference):
    outputs = [HERE / name for name in ("cases285.json", "history285.json", "bank285.json")]
    assert all(not path.exists() for path in outputs)
    specification = json.loads((HERE / "settings.json").read_text())["model"]
    tok = AutoTokenizer.from_pretrained(
        specification["model"], revision=specification["revision"], local_files_only=True
    )
    catalog, provenance = set(), []
    for path in sorted(HERE.glob("historical-input-hashes-*.json")):
        catalog.update(json.loads(path.read_text())["hashes"])
        provenance.append(dict(path=path.name, sha256=sha(path)))
    for path in sorted(HERE.glob("cases*.json")):
        value = json.loads(path.read_text())
        if isinstance(value, list):
            for case in value:
                for field in ("false_ids", "true_ids"):
                    if field in case:
                        catalog.add(input_hash(case[field]))
        provenance.append(dict(path=path.name, sha256=sha(path)))
    original = json.loads((HERE / "cases279.json").read_text())
    base = [
        c for c in original if c["group"]["block"] == "apatite" and c["group"]["assignment"] == 0
    ]
    assert len(base) == 8
    jobs, seen = [], set()
    for layout in LAYOUTS:
        for case in base:
            decoded = tok.decode(case["false_ids"])
            system = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", decoded, re.S).group(1)
            user = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", decoded, re.S).group(1)
            task = json.loads(user)
            assistant = decoded.split("<|im_start|>assistant\n")[1]
            assert assistant.count(": false,") == 1
            false_text = render(layout, system, task, assistant)
            true_text = render(layout, system, task, assistant.replace(": false,", ": true,"))
            f, t = (tok.encode(text, add_special_tokens=False) for text in (false_text, true_text))
            assert len(f) == len(t)
            changed = [i for i, (a, b) in enumerate(zip(f, t, strict=True)) if a != b]
            assert len(changed) == 1 and f[changed[0]] == 895 and t[changed[0]] == 830
            hashes = [input_hash(ids) for ids in (f, t)]
            assert not set(hashes) & (seen | catalog)
            seen.update(hashes)
            group = {**case["group"], "block": layout}
            key = hashlib.sha256(json.dumps(group, sort_keys=True).encode()).hexdigest()
            jobs.append(
                dict(
                    key=key,
                    group=group,
                    position=changed[0],
                    false_ids=f,
                    true_ids=t,
                    input_hashes=hashes,
                    task=task,
                    expected_supported=bool(group["target"]),
                )
            )
    counts = Counter(c["group"]["block"] for c in jobs)
    assert len(jobs) == 32 and len(seen) == 64 and counts == Counter(dict.fromkeys(LAYOUTS, 8))
    names = ("design.json", "forecast-status.json", "forecast-audit.json", "forecast-seal.json")
    reference_hashes = {name: sha(reference / name) for name in names}
    status = json.loads((reference / "forecast-status.json").read_text())
    assert status["status"] == "complete" and status["ordinary_true"] == 0
    assert json.loads((reference / "forecast-audit.json").read_text())["status"] == "passed"
    entries = {entry["key"]: entry for entry in status["cases"]}
    groups = {}
    for c in original:
        if c["group"]["assignment"] != 0:
            continue
        group = {name: c["group"][name] for name in ("schema", "reverse", "target", "assignment")}
        key = json.dumps(group, sort_keys=True)
        e = entries[c["key"]]
        path = reference / "forecasts" / (c["key"] + ".npz")
        assert (
            sha(path) == e["arrays_sha256"]
            and sha(path.with_suffix(".json")) == e["metadata_sha256"]
        )
        groups.setdefault(key, dict(group=group, donors=[]))["donors"].append(e)
    assert len(groups) == 8 and all(len(g["donors"]) == 2 for g in groups.values())
    bank = dict(
        experiment=285,
        reference_experiment=279,
        match_fields=["schema", "reverse", "target", "assignment"],
        reference_hashes=reference_hashes,
        groups=[groups[k] for k in sorted(groups)],
        boundary="Only sealed279 False/C forecast writes; no native observations or target norm. Equal mean of both literal blocks, assignment0 fixed before new forecasts.",
    )
    for path, value in zip(
        outputs,
        (
            jobs,
            dict(
                hashes=sorted(catalog),
                provenance=provenance,
                scope="Available historical input hashes; new prefix layout, same facts/checkpoint. No model forwards.",
            ),
            bank,
        ),
        strict=True,
    ):
        path.write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                cases=32,
                new_input_hashes=64,
                catalog=len(catalog),
                layouts=dict(counts),
                source_positions=sorted({c["position"] for c in jobs}),
                model_forwards=0,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-279", type=Path, required=True)
    main(parser.parse_args().reference_279.resolve())
