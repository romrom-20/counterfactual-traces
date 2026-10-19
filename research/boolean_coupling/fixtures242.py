"""Tokenization-only fresh pairs for the uniform primitive-family test."""

import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
FAMILIES = (("indigo", "indigo boathouse", "linen arcade", "Saturday", "Tuesday"),
    ("umber", "umber conservatory", "pewter station", "Thursday", "Monday"))


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def main():
    destination = HERE / "cases242.json"
    catalog_path = HERE / "historical-input-hashes-242.json"
    assert not destination.exists() and not catalog_path.exists()
    spec = json.loads((HERE / "settings.json").read_text())["model"]
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"], local_files_only=True)
    catalog, provenance = set(), []
    for p in sorted(HERE.glob("historical-input-hashes-*.json")):
        value = json.loads(p.read_text())
        catalog.update(value["hashes"])
        provenance.append(dict(path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    for p in sorted(HERE.glob("cases*.json")):
        value = json.loads(p.read_text())
        if not isinstance(value, list):
            continue
        for c in value:
            for key in ("false_ids", "true_ids"):
                if key in c:
                    catalog.add(digest(c[key]))
        provenance.append(dict(path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    jobs, seen = [], set()
    for family, a, b, day0, day1 in FAMILIES:
        replacements = dict(marzipan=a, nougat=b, Monday=day0, Wednesday=day1)
        def replace(value):
            return re.sub(r"\b(marzipan|nougat|Monday|Wednesday)\b", lambda m: replacements[m.group()], value)
        for c in json.loads((HERE / "cases.json").read_text()):
            if c["group"]["block"] != "marzipan":
                continue
            ids = {bit: tokenizer.encode(replace(tokenizer.decode(c[bit + "_ids"])), add_special_tokens=False)
                for bit in ("false", "true")}
            assert len(ids["false"]) == len(ids["true"])
            positions = [i for i, (x, y) in enumerate(zip(ids["false"], ids["true"], strict=True)) if x != y]
            assert len(positions) == 1
            position = positions[0]
            assert ids["false"][position] == 895 and ids["true"][position] == 830
            hashes = [digest(ids[bit]) for bit in ("false", "true")]
            assert not set(hashes) & (catalog | seen)
            seen.update(hashes)
            group = {**c["group"], "block": family}
            key = hashlib.sha256(json.dumps(group, sort_keys=True).encode()).hexdigest()
            jobs.append(dict(key=key, group=group, position=position, false_ids=ids["false"], true_ids=ids["true"],
                input_hashes=hashes, messages=[{**m, "content": replace(m["content"])} for m in c["messages"]]))
    assert len(jobs) == 32 and len(seen) == 64
    destination.write_text(json.dumps(jobs, indent=2) + "\n")
    catalog_path.write_text(json.dumps(dict(hashes=sorted(catalog), provenance=provenance,
        scope="Union of published historical catalogs and all available cases*.json token sequences."), indent=2) + "\n")
    print("32 fresh literal pairs, 64 input hashes absent from", len(catalog), "recorded sequences; zero model forwards.")


if __name__ == "__main__":
    main()
