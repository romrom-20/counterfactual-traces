"""Fresh284 phase boundaries and lossless records; update itself is unchanged283."""

import hashlib
import json
from pathlib import Path

from .audit229 import sha
from .audit272 import read
from .carried283 import SOURCES as SOURCE283
from .carried283 import restore

ROOT = Path(__file__).resolve().parents[2]
FORECAST = ("F", "B", "K", "P", "H", "J", "G0", "G1", "G2", "FZ")
NATIVE = ("C1", "U", "N", "T")
SOURCES = (
    *SOURCE283,
    "research/boolean_coupling/carried284.py",
    "research/boolean_coupling/run284.py",
    "research/boolean_coupling/audit284.py",
    "research/boolean_coupling/preflight284.py",
    "research/boolean_coupling/fixtures284.py",
    "research/boolean_coupling/cases284.json",
    "research/boolean_coupling/historical-input-hashes-284.json",
    "docs/experiments/284-sealed-carried-response.md",
)


class Memory:
    def __init__(self, values=None):
        self.values = {} if values is None else values
        self.files = list(self.values)

    def __getitem__(self, name):
        return self.values[name]


def input_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def validate_inputs(cases, history):
    seen, groups = set(), {}
    assert len(cases) == 32
    for c in cases:
        f, t = c["false_ids"], c["true_ids"]
        assert len(f) == len(t) and [
            i for i, (a, b) in enumerate(zip(f, t, strict=True)) if a != b
        ] == [c["position"]]
        assert f[c["position"]] == 895 and t[c["position"]] == 830
        hs = [input_hash(ids) for ids in (f, t)]
        assert hs == c["input_hashes"] and not set(hs) & (set(history) | seen)
        seen.update(hs)
        g = c["group"]
        index = tuple(g[n] for n in ("assignment", "block", "reverse", "target"))
        assert g["schema"] not in groups.setdefault(index, {})
        groups[index][g["schema"]] = c["key"]
    assert (
        len(seen) == 64
        and len(groups) == 16
        and all(set(x) == {"generic", "descriptive"} for x in groups.values())
    )
    for schema in ("generic", "descriptive"):
        for reverse in (0, 1):
            assert (
                sum(
                    c["group"]["schema"] == schema and c["group"]["reverse"] == reverse
                    for c in cases
                )
                == 8
            )


def load_forecast(run, entry):
    p = run / "forecasts" / (entry["key"] + ".npz")
    assert (
        sha(p) == entry["arrays_sha256"] and sha(p.with_suffix(".json")) == entry["metadata_sha256"]
    )
    m = read(p.with_suffix(".json"))
    assert all(set(a) == {"local"} for a in m["array_aliases"].values())
    return restore(p, m, Memory()), m


def load_native(run, entry, forecast):
    p = run / "native" / (entry["key"] + ".npz")
    assert (
        sha(p) == entry["arrays_sha256"] and sha(p.with_suffix(".json")) == entry["metadata_sha256"]
    )
    m = read(p.with_suffix(".json"))
    return restore(p, m, Memory(forecast)), m


def bindings(run):
    d = read(run / "design.json")
    assert d["experiment"] == 284 and d["globally_pre_native"]
    for n, h in d["source_hashes"].items():
        assert sha(ROOT / n) == sha(run / "sources" / Path(n).name) == h
    validate_inputs(d["cases"], d["historical_input_hashes"])
    return d
