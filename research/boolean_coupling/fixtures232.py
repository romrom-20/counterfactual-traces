"""Two fresh tokenized families for a source-context local forecast; no model."""

import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
FAMILIES = (("indigo", "indigo theater", "vermilion studio", "Thursday", "Saturday"),
            ("quartz", "quartz conservatory", "velvet arcade", "Wednesday", "Friday"))


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def main():
    destination, catalog_path = HERE/"cases232.json", HERE/"historical-input-hashes-232.json"
    assert not destination.exists() and not catalog_path.exists(), "Never overwrite inputs."
    spec = json.loads((HERE/"settings.json").read_text())["model"]
    tok = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"], local_files_only=True)
    catalog = set(json.loads((HERE/"historical-input-hashes-231.json").read_text())["hashes"])
    for case in json.loads((HERE/"cases231.json").read_text()):
        catalog.update(digest(case[k+"_ids"]) for k in ("false", "true"))
    jobs, fresh = [], set()
    for family, a, b, day0, day1 in FAMILIES:
        replacements = dict(marzipan=a, nougat=b, Monday=day0, Wednesday=day1)
        def replace(value):
            return re.sub(r"\b(marzipan|nougat|Monday|Wednesday)\b", lambda m:replacements[m.group()], value)
        for case in json.loads((HERE/"cases.json").read_text()):
            if case["group"]["block"] != "marzipan":
                continue
            ids = {k:tok.encode(replace(tok.decode(case[k+"_ids"])), add_special_tokens=False) for k in ("false", "true")}
            changed = [i for i, (x, y) in enumerate(zip(ids["false"], ids["true"])) if x != y]
            assert len(ids["false"]) == len(ids["true"]) and len(changed) == 1
            position = changed[0]
            assert ids["false"][position] == 895 and ids["true"][position] == 830
            hashes = [digest(ids[k]) for k in ("false", "true")]
            assert not (set(hashes) & (catalog | fresh))
            fresh.update(hashes)
            group = {**case["group"], "block":family}
            key = hashlib.sha256(json.dumps(group, sort_keys=True).encode()).hexdigest()
            jobs.append(dict(key=key, group=group, position=position, false_ids=ids["false"], true_ids=ids["true"],
                input_hashes=hashes, messages=[{**m, "content":replace(m["content"])} for m in case["messages"]]))
    assert len(jobs) == 32 and len(fresh) == 64
    destination.write_text(json.dumps(jobs, indent=2)+"\n")
    catalog_path.write_text(json.dumps(dict(hashes=sorted(catalog),
        scope="Recorded 231 catalog plus all 231 token sequences; inherited bounded historical coverage."), indent=2)+"\n")
    print(f"Built 32 pairs; 64 hashes absent from {len(catalog)} recorded prior sequences.")


if __name__ == "__main__":
    main()
