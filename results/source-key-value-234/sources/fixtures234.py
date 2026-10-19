"""Fresh tokenization-only source-key/value forecast cases; no model weights."""

import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
FAMILIES = (("jade", "jade pavilion", "copper rotunda", "Monday", "Thursday"),
            ("beige", "beige observatory", "ginger foundry", "Tuesday", "Friday"))


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def main():
    dest, prior = HERE/"cases234.json", HERE/"historical-input-hashes-234.json"
    assert not dest.exists() and not prior.exists()
    spec = json.loads((HERE/"settings.json").read_text())["model"]
    tok = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"], local_files_only=True)
    catalog = set(json.loads((HERE/"historical-input-hashes-232.json").read_text())["hashes"])
    for c in json.loads((HERE/"cases232.json").read_text()):
        catalog.update(digest(c[b+"_ids"]) for b in ("false", "true"))
    jobs, seen = [], set()
    for family, a, b, day0, day1 in FAMILIES:
        replacements = dict(marzipan=a, nougat=b, Monday=day0, Wednesday=day1)
        def replace(value):
            return re.sub(r"\b(marzipan|nougat|Monday|Wednesday)\b", lambda m:replacements[m.group()], value)
        for c in json.loads((HERE/"cases.json").read_text()):
            if c["group"]["block"] != "marzipan":
                continue
            ids = {bit:tok.encode(replace(tok.decode(c[bit+"_ids"])), add_special_tokens=False) for bit in ("false", "true")}
            positions = [i for i, (x, y) in enumerate(zip(ids["false"], ids["true"])) if x != y]
            assert len(ids["false"]) == len(ids["true"]) and len(positions) == 1
            position = positions[0]
            assert ids["false"][position] == 895 and ids["true"][position] == 830
            hashes = [digest(ids[bit]) for bit in ("false", "true")]
            assert not set(hashes) & (catalog | seen)
            seen.update(hashes)
            group = {**c["group"], "block":family}
            key = hashlib.sha256(json.dumps(group, sort_keys=True).encode()).hexdigest()
            jobs.append(dict(key=key, group=group, position=position,
                false_ids=ids["false"], true_ids=ids["true"], input_hashes=hashes,
                messages=[{**m, "content":replace(m["content"])} for m in c["messages"]]))
    assert len(jobs) == 32 and len(seen) == 64
    dest.write_text(json.dumps(jobs, indent=2)+"\n")
    prior.write_text(json.dumps(dict(hashes=sorted(catalog),
        scope="Bounded 232 catalog plus every 232 token sequence; 233 reuses 232."), indent=2)+"\n")
    print(f"Built 32 pairs, 64 hashes absent from {len(catalog)} prior sequences; zero model calls.")


if __name__ == "__main__":
    main()
