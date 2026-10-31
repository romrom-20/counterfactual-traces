"""Freeze existing297 witness bindings and deterministic token-position groups."""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from .messages298 import ROOT, roles
from .publish285 import read, sha, write


def main(parent):
    dest = ROOT / "research/boolean_coupling/parents298.json"
    assert not dest.exists()
    design, status = read(parent / "design.json"), read(parent / "native-status.json")
    public = ROOT / "results/query-routes-297"
    assert sha(parent / "design.json") == sha(public / "pre-native/design.json")
    assert sha(parent / "native-status.json") == sha(public / "native-phase/native-status.json")
    spec = read(ROOT / "research/boolean_coupling/settings.json")["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model"], revision=spec["revision"], local_files_only=True
    )
    eom = tokenizer.convert_tokens_to_ids("<|im_end|>")
    boolean_ids = []
    for word in ["true", "false", " true", " false"]:
        ids = tokenizer.encode(word, add_special_tokens=False)
        assert len(ids) == 1
        boolean_ids.extend(ids)
    entries = {e["key"]: e for e in status["cases"]}
    cases = []
    for c in design["cases"]:
        if c["group"]["reverse"] or c["group"]["target"]:
            continue
        e = entries[c["key"]]
        assert e["witness_sha256"]
        native, witness = (
            parent / "observations" / (c["key"] + ".npz"),
            parent / "witnesses-native" / (c["key"] + ".npz"),
        )
        assert sha(native) == e["arrays_sha256"] and sha(witness) == e["witness_sha256"]
        assert (
            sha(native.with_suffix(".json"))
            == e["metadata_sha256"]
            == sha(public / "native-phase/cases" / native.with_suffix(".json").name)
        )
        cases.append(
            dict(
                key=c["key"],
                group=c["group"],
                position=c["position"],
                length=len(c["false_ids"]),
                input_hashes=c["input_hashes"],
                ids=c["false_ids"],
                positions=roles(
                    c["false_ids"], c["position"], c["group"]["schema"], eom, set(boolean_ids)
                ),
                parent_hashes=dict(
                    native=e["arrays_sha256"],
                    witness=e["witness_sha256"],
                    metadata=e["metadata_sha256"],
                ),
            )
        )
    assert len(cases) == 8
    write(
        dest,
        dict(
            experiment=298,
            retrospective=True,
            source_experiment=297,
            source_design_sha256=sha(parent / "design.json"),
            source_native_status_sha256=sha(parent / "native-status.json"),
            source_native_audit_sha256=sha(parent / "audit.json"),
            original_global_seal_commit="3c031b37c872520d61063f0395bad0c4af2a238a",
            eom_id=eom,
            boolean_ids=sorted(set(boolean_ids)),
            cases=cases,
            boundary="Eight fixed old native witnesses, group hypothesis informed by posthoc readouts; no fresh input or prospective prediction.",
        ),
    )
    print(dict(cases=8, eom_id=eom, boolean_ids=sorted(set(boolean_ids)), model_forwards=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    main(parser.parse_args().parent.resolve())
