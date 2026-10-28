"""Tokenization-only new literals in all four existing structural layouts."""

import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

from .fixtures285 import LAYOUTS, input_hash, render, sha

HERE = Path(__file__).resolve().parent


def main():
    assert not any((HERE / n).exists() for n in ("cases292.json", "history292.json"))
    spec = json.loads((HERE / "settings.json").read_text())["model"]
    tok = AutoTokenizer.from_pretrained(
        spec["model"], revision=spec["revision"], local_files_only=True
    )
    catalog, provenance = set(), []
    for path in sorted(HERE.glob("*.json")):
        value = json.loads(path.read_text())
        if path.name.startswith(("history", "historical-input")) and isinstance(value, dict):
            catalog.update(value.get("hashes", []))
        if path.name.startswith("cases") and isinstance(value, list):
            for c in value:
                for field in ("false_ids", "true_ids"):
                    if field in c:
                        catalog.add(input_hash(c[field]))
        if path.name.startswith(("cases", "history", "historical-input")):
            provenance.append(dict(path=path.name, sha256=sha(path)))
    base = [
        c
        for c in json.loads((HERE / "cases279.json").read_text())
        if c["group"]["block"] == "apatite" and c["group"]["assignment"] == 0
    ]
    assert len(base) == 8
    jobs, seen = [], set()
    for layout in LAYOUTS:
        for case in base:
            decoded = tok.decode(case["false_ids"])
            system = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", decoded, re.S).group(1)
            task = json.loads(
                re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", decoded, re.S).group(1)
            )
            replacements = {
                "apatite greenhouse": "rhodonite conservatory",
                "elm auditorium": "carnelian planetarium",
                "Sunday": "Tuesday",
                "Friday": "Saturday",
            }
            for field, value in task.items():
                for a, b in replacements.items():
                    value = value.replace(a, b)
                task[field] = value
            assistant = decoded.split("<|im_start|>assistant\n")[1]
            assert assistant.count(": false,") == 1
            false = render(layout, system, task, assistant)
            true = render(layout, system, task, assistant.replace(": false,", ": true,"))
            f, t = (tok.encode(s, add_special_tokens=False) for s in (false, true))
            changed = [j for j, (a, b) in enumerate(zip(f, t, strict=True)) if a != b]
            assert (
                len(f) == len(t)
                and len(changed) == 1
                and f[changed[0]] == 895
                and t[changed[0]] == 830
            )
            hs = [input_hash(f), input_hash(t)]
            assert not set(hs) & (catalog | seen)
            seen.update(hs)
            group = {**case["group"], "block": layout}
            key = hashlib.sha256(
                json.dumps(
                    dict(experiment=292, group=group, input_hashes=hs), sort_keys=True
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
                    expected_supported=bool(group["target"]),
                )
            )
    assert len(jobs) == 32 and len(seen) == 64
    for name, value in [
        ("cases292.json", jobs),
        (
            "history292.json",
            dict(
                hashes=sorted(catalog),
                provenance=provenance,
                scope="Available historical input catalog; unseen full token sequences in dependent templates. Tokenization only, no model forwards.",
            ),
        ),
    ]:
        (HERE / name).write_text(json.dumps(value, indent=2) + "\n")
    print(
        dict(
            cases=32,
            new_input_hashes=64,
            historical_hashes=len(catalog),
            model_forwards=0,
            lengths=sorted({len(c["false_ids"]) for c in jobs}),
            tail_lengths=sorted({len(c["false_ids"]) - c["position"] for c in jobs}),
        )
    )


if __name__ == "__main__":
    main()
