"""Tokenization-only fresh literals for the native query-route test."""

import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

from .fixtures296 import LAYOUTS, digest, render

HERE = Path(__file__).resolve().parent


def main():
    assert not (HERE / "cases297.json").exists() and not (HERE / "history297.json").exists()
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
    jobs, seen = [], set()
    for layout in LAYOUTS:
        for c in base:
            decoded = tok.decode(c["false_ids"])
            system = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", decoded, re.S).group(1)
            task = c["task"].copy()
            for k, value in task.items():
                for old, new in {
                    "rhodonite conservatory": "jasper archive",
                    "carnelian planetarium": "fluorite gallery",
                    "Tuesday": "Friday",
                    "Saturday": "Thursday",
                }.items():
                    value = value.replace(old, new)
                task[k] = value
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
                    dict(experiment=297, group=group, input_hashes=hs), sort_keys=True
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
        ("cases297.json", jobs),
        (
            "history297.json",
            dict(
                hashes=sorted(catalog),
                provenance=provenance,
                scope="Available complete input hashes, related synthetic templates; no model forward.",
            ),
        ),
    ]:
        (HERE / name).write_text(json.dumps(value, indent=2) + "\n")
    print(dict(cases=32, fresh_hashes=64, historical_hashes=len(catalog), model_forwards=0))


if __name__ == "__main__":
    main()
