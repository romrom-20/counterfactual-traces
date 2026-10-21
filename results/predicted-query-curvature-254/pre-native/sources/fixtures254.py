"""Tokenization-only new literals for the fixed-attention hypothesis; no forwards."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
FAMILIES = (("mauve", "mauve atrium", "tin observatory", "Monday", "Friday"),
    ("ochre", "ochre courtyard", "glass terminal", "Sunday", "Tuesday"))


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def main():
    dest, catalog_path = HERE/"cases254.json", HERE/"historical-input-hashes-254.json"
    assert not dest.exists() and not catalog_path.exists()
    spec = json.loads((HERE/"settings.json").read_text())["model"]
    tok = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"], local_files_only=True)
    catalog, provenance = set(), []
    for path in sorted(HERE.glob("historical-input-hashes-*.json")):
        catalog.update(json.loads(path.read_text())["hashes"])
        provenance.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for path in sorted(HERE.glob("cases*.json")):
        value = json.loads(path.read_text())
        if isinstance(value, list):
            for case in value:
                for key in ("false_ids", "true_ids"):
                    if key in case:
                        catalog.add(digest(case[key]))
        provenance.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    jobs, seen, strata = [], set(), Counter()
    for family, a, b, day0, day1 in FAMILIES:
        replacements = dict(marzipan=a, nougat=b, Monday=day0, Wednesday=day1)

        def replace(value):
            return re.sub(r"\b(marzipan|nougat|Monday|Wednesday)\b", lambda m: replacements[m.group()], value)

        for case in json.loads((HERE/"cases.json").read_text()):
            if case["group"]["block"] != "marzipan":
                continue
            ids = {bit:tok.encode(replace(tok.decode(case[bit+"_ids"])), add_special_tokens=False) for bit in ("false", "true")}
            assert len(ids["false"]) == len(ids["true"])
            positions = [i for i, (x, y) in enumerate(zip(ids["false"], ids["true"], strict=True)) if x != y]
            assert len(positions) == 1 and ids["false"][positions[0]] == 895 and ids["true"][positions[0]] == 830
            hashes = [digest(ids[bit]) for bit in ("false", "true")]
            assert not set(hashes) & (seen | catalog)
            seen.update(hashes)
            group = {**case["group"], "block":family}
            strata[(group["schema"], group["reverse"])] += 1
            key = hashlib.sha256(json.dumps(group, sort_keys=True).encode()).hexdigest()
            jobs.append(dict(key=key, group=group, position=positions[0], false_ids=ids["false"], true_ids=ids["true"], input_hashes=hashes))
    assert len(jobs) == 32 and len(seen) == 64 and len(strata) == 4 and set(strata.values()) == {8}
    dest.write_text(json.dumps(jobs, indent=2)+"\n")
    catalog_path.write_text(json.dumps(dict(hashes=sorted(catalog), provenance=provenance,
        scope="Published historical catalogs and every available cases*.json token sequence; fresh literals, same dependent templates."), indent=2)+"\n")
    print(f"32 new literal pairs;64 hashes absent from {len(catalog)} cataloged sequences; zero model forwards.")


if __name__ == "__main__":
    main()
