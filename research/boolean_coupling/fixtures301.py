"""Tokenization-only equivalent aliases; matched positions and one Boolean edit."""

import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from .fixtures296 import LAYOUTS, digest, render

HERE = Path(__file__).resolve().parent


def main():
    assert not any(
        (HERE / n).exists() for n in ("cases301.json", "cases301-flat.json", "history301.json")
    )
    spec = json.loads((HERE / "settings.json").read_text())["model"]
    tok = AutoTokenizer.from_pretrained(
        spec["model"], revision=spec["revision"], local_files_only=True
    )
    catalog, provenance = set(), []
    for p in sorted(HERE.glob("*.json")):
        value = json.loads(p.read_text())
        if p.name.startswith(("history", "historical-input")) and isinstance(value, dict):
            catalog.update(value.get("hashes", []))
        if p.name.startswith("cases") and isinstance(value, list):
            for c in value:
                for field in ("false_ids", "true_ids"):
                    if field in c:
                        catalog.add(digest(c[field]))
        if p.name.startswith(("history", "historical-input", "cases")):
            provenance.append(dict(path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    base = [
        c
        for c in json.loads((HERE / "cases299.json").read_text())
        if c["group"]["block"] == "jsonlines" and c["group"]["schema"] == "generic"
    ]
    assert len(base) == 4
    jobs, flat, seen = [], [], set()
    for layout in LAYOUTS:
        for mention_order in (0, 1):
            aliases = (
                '"first_value" or "record_flag"'
                if mention_order == 0
                else '"record_flag" or "first_value"'
            )
            system = (
                "Use only the passage to judge whether proposed_answer is directly supported as an answer to question. "
                "An occurrence of the proposed answer in an unrelated fact is not sufficient. Do not use outside knowledge. "
                "Return only a JSON object with two boolean fields: one arbitrary metadata field named either "
                + aliases
                + ', followed by "supported". Those two metadata names are interchangeable aliases; use exactly one of them. '
                "The metadata value has no meaning about the passage, question or answer correctness. Either true or false is permitted. "
                'For "supported", true means the proposed answer is supported and false means it is unsupported.'
            )
            for old in base:
                task = old["task"].copy()
                for field, value in task.items():
                    for a, b in {
                        "azurite observatory": "obsidian archive",
                        "serpentine museum": "agate pavilion",
                        "Wednesday": "Thursday",
                        "Monday": "Sunday",
                    }.items():
                        value = value.replace(a, b)
                    task[field] = value
                pair = {}
                for alias, name in (("A", "first_value"), ("B", "record_flag")):
                    f = tok.encode(
                        render(layout, system, task, '{"' + name + '": false, "supported":'),
                        add_special_tokens=False,
                    )
                    t = tok.encode(
                        render(layout, system, task, '{"' + name + '": true, "supported":'),
                        add_special_tokens=False,
                    )
                    changed = [j for j, (a, b) in enumerate(zip(f, t, strict=True)) if a != b]
                    assert len(changed) == 1 and f[changed[0]] == 895 and t[changed[0]] == 830
                    hs = [digest(f), digest(t)]
                    assert not set(hs) & (catalog | seen)
                    seen.update(hs)
                    pair[alias] = dict(name=name, false_ids=f, true_ids=t, input_hashes=hs)
                names = [
                    i
                    for i, (a, b) in enumerate(
                        zip(pair["A"]["false_ids"], pair["B"]["false_ids"], strict=True)
                    )
                    if a != b
                ]
                s = changed[0]
                assert len(names) == 2 and max(names) < s
                assert pair["A"]["false_ids"][s:] == pair["B"]["false_ids"][s:]
                group = dict(
                    block=layout,
                    mention_order=mention_order,
                    reverse=old["group"]["reverse"],
                    target=old["group"]["target"],
                )
                key = hashlib.sha256(
                    json.dumps(
                        dict(experiment=301, group=group, aliases=pair), sort_keys=True
                    ).encode()
                ).hexdigest()
                item = dict(
                    key=key,
                    group=group,
                    position=s,
                    alias_start=min(names),
                    alias_changed=names,
                    aliases=pair,
                    task=task,
                    expected_supported=old["expected_supported"],
                )
                jobs.append(item)
                for alias, c in pair.items():
                    flat.append(
                        dict(
                            key=key + "-" + alias,
                            group={**group, "alias": alias},
                            position=s,
                            false_ids=c["false_ids"],
                            true_ids=c["true_ids"],
                        )
                    )
    assert len(jobs) == 32 and len(flat) == 64 and len(seen) == 128
    for n, value in [
        ("cases301.json", jobs),
        ("cases301-flat.json", flat),
        (
            "history301.json",
            dict(
                hashes=sorted(catalog),
                provenance=provenance,
                scope="Available prior full-input hashes; cases301-flat exposes all128 new hashes to future catalogs. No model forwards.",
            ),
        ),
    ]:
        (HERE / n).write_text(json.dumps(value, indent=2) + "\n")
    print(dict(pairs=32, full_inputs=128, historical_hashes=len(catalog), model_forwards=0))


if __name__ == "__main__":
    main()
