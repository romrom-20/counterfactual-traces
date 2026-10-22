"""Clamp forecast-derived amplitudes and receiver queries after backbone ablation."""

import shutil
from pathlib import Path

from .audit229 import sha
from .backbone270 import SOURCES as PREVIOUS_SOURCES
from .backbone270 import policy as backbone_policy
from .fields267 import ROOT

ARMS = ("F", "Z", "L", "B", "J", "S", "Q", "C", "T")
SOURCES = (*PREVIOUS_SOURCES, "research/boolean_coupling/mediation271.py",
           "research/boolean_coupling/rollout271.py", "research/boolean_coupling/audit271.py",
           "docs/experiments/271-amplitude-query-mediation.md")


def bind(out):
    (out / "sources").mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT / name, out / "sources" / Path(name).name)
    return {name: sha(ROOT / name) for name in SOURCES}


def policy(arm, parts, controller, original_z, star, matrices, length, start, key, layer):
    # Only the two correction coefficients change source; the common backbone,
    # its tangent norm, and its finite gate/product diagnostic keep OWN inputs.
    selected = parts
    if arm in ("S", "C"):
        selected = dict(parts, base=controller["base"], amplitude=controller["amplitude"])
    target = "J" if arm in ("S", "Q", "C") else arm
    write, nodes, mechanism = backbone_policy(target, selected, original_z, star,
                                             matrices, length, start, key, layer)
    if arm in ("J", "S", "Q", "C"):
        mechanism.update(controller_p=selected["base"], controller_k=selected["amplitude"])
    return write, nodes, mechanism
