"""Tokenization-only32 cases in four extra evidence encodings, no model forward."""

import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
LAYOUTS = ("jsonlines", "ini", "sexpr", "memo")


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def render(layout, system, task, assistant):
    facts = [x.strip() + "." for x in task["passage"].split(".") if x.strip()]
    if layout == "jsonlines":
        user = (
            "Each line below is one record. Use only the fact records as evidence.\n"
            + "\n".join(json.dumps({"fact": x}) for x in facts)
            + "\n"
            + json.dumps({"question": task["question"]})
            + "\n"
            + json.dumps({"proposed_answer": task["proposed_answer"]})
        )
    elif layout == "ini":
        user = (
            "Interpret the following evidence and test sections literally.\n[evidence]\n"
            + "\n".join(f"statement_{i + 1} = {json.dumps(x)}" for i, x in enumerate(facts))
            + "\n[test]\nquestion = "
            + json.dumps(task["question"])
            + "\nproposed_answer = "
            + json.dumps(task["proposed_answer"])
        )
    elif layout == "sexpr":
        user = (
            "The following form is data, not code.\n(verification-input\n (evidence "
            + " ".join(json.dumps(x) for x in facts)
            + ")\n (question "
            + json.dumps(task["question"])
            + ")\n (proposed-answer "
            + json.dumps(task["proposed_answer"])
            + "))"
        )
    else:
        assert layout == "memo"
        user = (
            "BEGIN EVIDENCE MEMO\n"
            + "\n".join("Recorded statement: " + x for x in facts)
            + "\nEND EVIDENCE MEMO\nBEGIN REVIEW REQUEST\nQuestion to answer: "
            + task["question"]
            + "\nAnswer to check: "
            + task["proposed_answer"]
            + "\nEND REVIEW REQUEST"
        )
    return (
        "<|im_start|>system\n"
        + system
        + "<|im_end|>\n<|im_start|>user\n"
        + user
        + "<|im_end|>\n<|im_start|>assistant\n"
        + assistant
    )


def main():
    assert not (HERE / "cases296.json").exists() and not (HERE / "history296.json").exists()
    spec = json.loads((HERE / "settings.json").read_text())["model"]
    tok = AutoTokenizer.from_pretrained(
        spec["model"], revision=spec["revision"], local_files_only=True
    )
    catalog = set()
    provenance = []
    for p in sorted(HERE.glob("*.json")):
        value = json.loads(p.read_text())
        if p.name.startswith(("history", "historical-input")) and isinstance(value, dict):
            catalog.update(value.get("hashes", []))
        if p.name.startswith("cases") and isinstance(value, list):
            for c in value:
                for field in ["false_ids", "true_ids"]:
                    if field in c:
                        catalog.add(digest(c[field]))
        if p.name.startswith(("history", "historical-input", "cases")):
            provenance.append(dict(path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    base = [
        c
        for c in json.loads((HERE / "cases292.json").read_text())
        if c["group"]["block"] == "table"
    ]
    assert len(base) == 8
    jobs = []
    seen = set()
    for layout in LAYOUTS:
        for c in base:
            decoded = tok.decode(c["false_ids"])
            system = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", decoded, re.S).group(1)
            task = {k: v for k, v in c["task"].items()}
            replacements = {
                "rhodonite conservatory": "selenite observatory",
                "carnelian planetarium": "garnet pavilion",
                "Tuesday": "Wednesday",
                "Saturday": "Monday",
            }
            for k, v in task.items():
                for a, b in replacements.items():
                    v = v.replace(a, b)
                task[k] = v
            assistant = decoded.split("<|im_start|>assistant\n")[1]
            assert assistant.count(": false,") == 1
            f = tok.encode(render(layout, system, task, assistant), add_special_tokens=False)
            t = tok.encode(
                render(layout, system, task, assistant.replace(": false,", ": true,")),
                add_special_tokens=False,
            )
            changed = [j for j, (a, b) in enumerate(zip(f, t, strict=True)) if a != b]
            assert len(changed) == 1 and f[changed[0]] == 895 and t[changed[0]] == 830
            hs = [digest(f), digest(t)]
            assert not set(hs) & (catalog | seen)
            seen.update(hs)
            group = {**c["group"], "block": layout}
            key = hashlib.sha256(
                json.dumps(
                    dict(experiment=296, group=group, input_hashes=hs), sort_keys=True
                ).encode()
            ).hexdigest()
            jobs.append(
                dict(
                    key=key,
                    group=group,
                    position=changed[0],
                    false_ids=f,
                    true_ids=t,
                    input_hashes=hs,
                    task=task,
                    expected_supported=c["expected_supported"],
                )
            )
    assert len(jobs) == 32 and len(seen) == 64
    for name, value in [
        ("cases296.json", jobs),
        (
            "history296.json",
            dict(
                hashes=sorted(catalog),
                provenance=provenance,
                scope="Available historical full-input hashes only; dependent synthetic evidence encodings, tokenization without model forwards.",
            ),
        ),
    ]:
        (HERE / name).write_text(json.dumps(value, indent=2) + "\n")
    print(
        dict(
            cases=32,
            new_input_hashes=64,
            historical_hashes=len(catalog),
            lengths=sorted({len(c["false_ids"]) for c in jobs}),
            model_forwards=0,
        )
    )


if __name__ == "__main__":
    main()
