"""Tokenization-only fresh local-forecast fixture. Never loads a model."""

import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def main():
    destination, catalog_path = HERE/"cases231.json", HERE/"historical-input-hashes-231.json"
    assert not destination.exists() and not catalog_path.exists(), "Never overwrite inputs."
    spec = json.loads((HERE/"settings.json").read_text())["model"]
    tok = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"], local_files_only=True)
    catalog = set(json.loads((HERE/"historical-input-hashes-228.json").read_text())["hashes"])
    for case in json.loads((HERE/"cases228.json").read_text()):
        catalog.update(digest(case[k+"_ids"]) for k in ("false", "true"))
    replacement = dict(marzipan="coral sculpture", nougat="silver tapestry", Monday="Saturday", Wednesday="Tuesday")
    def replace(value):
        return re.sub(r"\b(marzipan|nougat|Monday|Wednesday)\b", lambda m:replacement[m.group()], value)
    jobs, fresh = [], set()
    for case in json.loads((HERE/"cases.json").read_text()):
        if case["group"]["block"] != "marzipan":
            continue
        ids = {k:tok.encode(replace(tok.decode(case[k+"_ids"])), add_special_tokens=False) for k in ("false", "true")}
        changed = [i for i, (a, b) in enumerate(zip(ids["false"], ids["true"])) if a != b]
        assert len(ids["false"]) == len(ids["true"]) and len(changed) == 1
        position = changed[0]
        assert ids["false"][position] == 895 and ids["true"][position] == 830
        hashes = [digest(ids[k]) for k in ("false", "true")]
        assert not (set(hashes) & (catalog | fresh))
        fresh.update(hashes)
        group = {**case["group"], "block":"coral"}
        key = hashlib.sha256(json.dumps(group, sort_keys=True).encode()).hexdigest()
        jobs.append(dict(key=key, group=group, position=position, false_ids=ids["false"], true_ids=ids["true"],
            input_hashes=hashes, messages=[{**m, "content":replace(m["content"])} for m in case["messages"]]))
    assert len(jobs) == 16 and len(fresh) == 32
    destination.write_text(json.dumps(jobs, indent=2)+"\n")
    catalog_path.write_text(json.dumps(dict(hashes=sorted(catalog), scope="Recorded 228 catalog plus all 228 pairs; "
        "229 and 230 reuse those pairs. This is not a catalog of every possible historical artifact."), indent=2)+"\n")
    print(f"Built 16 pairs; 32 hashes absent from {len(catalog)} historical token sequences.")


if __name__ == "__main__":
    main()
