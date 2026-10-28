"""Finite receiving-MLP coefficient interventions; no backward replacement."""

import itertools
from contextlib import contextmanager

import numpy as np
import torch

from .core225b import array
from .interface289 import Interface
from .receiver290 import SOURCES as PREVIOUS
from .receiver290 import Receiver

CORNERS = tuple("".join(x) for x in itertools.product("01", repeat=3))
ARMS = (*CORNERS, "R0", "R1", "R2")
FACTORS = ["gate_response", "up_multiplier", "up_response_coefficient"]
SOURCES = (
    *PREVIOUS,
    "research/boolean_coupling/coeff291.py",
    "research/boolean_coupling/run291.py",
    "research/boolean_coupling/audit291.py",
    "research/boolean_coupling/preflight291.py",
    "docs/experiments/291-receiving-coefficients.md",
)
PRIMARY = dict(
    hypotheses={
        "GATE": "100",
        "UP_MULTIPLIER": "010",
        "UP_COEFFICIENT": "001",
        "GATE_AND_UP": "110",
        "GATE_AND_COEFFICIENT": "101",
        "UP_AND_COEFFICIENT": "011",
    },
    norm_min=0.001,
    relative_max=0.25,
    cosine_min=0.95,
    finite_absolute=0.005,
    finite_relative=0.25,
    material_min=0.02,
    population_min=24,
    per_layout_min=6,
)


def slope(g):
    s = torch.sigmoid(g)
    return s * (1 + g * (1 - s))


def permutation(key, j, arm, width):
    seed = (int(key[:16], 16) + 291_000_003 * (j + 1) + int(arm[1:])) % (2**64)
    return np.random.default_rng(seed).permutation(width)


class Coefficients(Receiver):
    def __init__(self, model, case, write):
        self.operands = []
        self.case_key = case["key"]
        super().__init__(model, case, write)
        assert len(self.operands) == 2
        self.counts.update(
            coefficient_maps=0,
            mixed_base_down=0,
            coefficient_projections=0,
            coefficient_probe_VJPs=0,
            coefficient_probe_down=0,
            coefficient_probe_finite_down=0,
        )

    def native(self, base):
        if not self.collect_anchors:
            return super().native(base)
        saved, handles = {}, []
        for j, layer in enumerate(self.model.model.layers[1:]):
            saved[j] = {}
            for module, name in [
                (layer.mlp.gate_proj, "g"),
                (layer.mlp.up_proj, "u"),
                (layer.mlp.act_fn, "phi"),
            ]:

                def capture(module, args, output, j=j, name=name):
                    saved[j][name] = output.detach().clone()

                handles.append(module.register_forward_hook(capture))
        try:
            result = super().native(base)
        finally:
            for h in handles:
                h.remove()
        self.operands.append(saved)
        return result

    def coefficients(self, j, arm):
        f, e = self.operands[0][j], self.operands[1][j]
        da = e["u"] * slope(e["g"]) - f["u"] * slope(f["g"])
        db = e["phi"] - f["phi"]
        ix = torch.tensor(
            permutation(self.case_key, j, arm, da.shape[-1]), device=da.device, dtype=torch.long
        )
        return da.index_select(-1, ix), db.index_select(-1, ix)

    def product(self, j, arm, dg, du):
        """Actual finite primitive function for the saved-parent VJP probe."""
        mlp = self.model.model.layers[j + 1].mlp
        f = self.operands[0][j]
        if arm.startswith("R"):
            da, db = self.coefficients(j, arm)
            return mlp.act_fn(f["g"] + dg) * (f["u"] + du) + da * dg + db * du
        a, b, c = map(int, arm)
        gate, up, coefficient = (self.operands[x][j] for x in (a, b, c))
        phi = mlp.act_fn(gate["g"] + dg)
        return phi * (up["u"] + du) + (coefficient["phi"] - gate["phi"]) * du

    @contextmanager
    def receiving(self, j, arm):
        layer = self.model.model.layers[j + 1]
        mlp = layer.mlp
        assert mlp.down_proj.bias is None
        f = self.receiving_anchors[0][j]
        prior_attribute = mlp.__dict__.get("forward")
        original = mlp.forward
        if arm.startswith("R"):
            da, db = self.coefficients(j, arm)

            def finite(value):
                dg = mlp.gate_proj(value) - self.operands[0][j]["g"]
                du = mlp.up_proj(value) - self.operands[0][j]["u"]
                self.counts["coefficient_maps"] += 1
                self.counts["coefficient_projections"] += 4
                return (
                    f["write"] + (original(value) - f["write"]) + mlp.down_proj(da * dg + db * du)
                )
        else:
            a, b, c = map(int, arm)
            gate, up, coefficient = (self.operands[x][j] for x in (a, b, c))
            with torch.no_grad():
                base = (
                    self.receiving_anchors[a][j]["write"]
                    if a == b
                    else mlp.down_proj(gate["phi"] * up["u"])
                )
            if a != b:
                self.counts["mixed_base_down"] += 1

            def finite(value):
                t = value - f["post_norm"]
                zg = self.receiving_anchors[a][j]["post_norm"] + t
                zu = self.receiving_anchors[b][j]["post_norm"] + t
                phi = mlp.act_fn(mlp.gate_proj(zg))
                current_up = mlp.up_proj(zu)
                product = phi * current_up
                if c != a:
                    product = product + (coefficient["phi"] - gate["phi"]) * (current_up - up["u"])
                self.counts["coefficient_maps"] += 1
                self.counts["coefficient_projections"] += 2
                return f["write"] + (mlp.down_proj(product) - base)

        with Interface.receiving(self, j, "111"):
            mlp.forward = finite
            try:
                yield
            finally:
                if prior_attribute is None:
                    del mlp.__dict__["forward"]
                else:
                    mlp.forward = prior_attribute


def primitive_probes(engine):
    """One seeded layer, actual cached operand shapes, all intermediate coordinates."""
    j = int(engine.case_key[16:24], 16) % (len(engine.tapes[0]) - 1)
    mlp = engine.model.model.layers[j + 1].mlp
    eta = float(np.float32(1 / np.sqrt(engine.model.config.hidden_size)))
    q = torch.full((mlp.down_proj.weight.shape[0],), eta, device=engine.model.device)
    cot = q @ mlp.down_proj.weight
    cpu = np.full(len(q), eta) @ array(mlp.down_proj.weight).astype(float)
    data = {"probe_layer": np.asarray(j), "probe_down_cot": array(cot), "probe_down_CPU64": cpu}
    for a in (0, 1):
        for n in ("g", "u", "phi"):
            data[f"probe_{a}_{n}"] = array(engine.operands[a][j][n][0, -1])
    for arm in ARMS:
        with torch.enable_grad():
            g = torch.zeros_like(engine.operands[0][j]["g"], requires_grad=True)
            u = torch.zeros_like(g, requires_grad=True)
            out = mlp.down_proj(engine.product(j, arm, g, u))
            gg, gu = torch.autograd.grad(out[0, -1].sum() * eta, (g, u))
        data[arm + "_probe_gate_AD"] = array(gg[0, -1])
        data[arm + "_probe_up_AD"] = array(gu[0, -1])
        other = torch.cat((gg[:, :-1].flatten(), gu[:, :-1].flatten()))
        data[arm + "_probe_other_max"] = np.asarray(float(other.abs().max().cpu()))
        engine.counts["coefficient_probe_VJPs"] += 1
        engine.counts["coefficient_probe_down"] += 1
        coords = np.random.default_rng(int(engine.case_key[:16], 16) + j).choice(
            g.shape[-1], 2, replace=False
        )
        for kind, coordinate in zip(("gate", "up"), coords, strict=True):
            data[f"{arm}_probe_{kind}_coordinate"] = np.asarray(coordinate)
            for step in (0.05, 0.025):
                values = []
                for sign in (1, -1):
                    dg, du = torch.zeros_like(g), torch.zeros_like(u)
                    (dg if kind == "gate" else du)[0, -1, coordinate] = sign * step
                    with torch.no_grad():
                        out = mlp.down_proj(engine.product(j, arm, dg, du))
                        values.append(float((out[0, -1].sum() * eta).cpu()))
                    engine.counts["coefficient_probe_finite_down"] += 1
                data[f"{arm}_probe_{kind}_scores_{step}"] = np.asarray(values)
    return data


def primitive_records(data, key):
    def sp(g):
        s = 1 / (1 + np.exp(-g))
        return s * (1 + g * (1 - s))

    rows = []
    f = {n: data["probe_0_" + n].astype(float) for n in ("g", "u", "phi")}
    e = {n: data["probe_1_" + n].astype(float) for n in ("g", "u", "phi")}
    cot = data["probe_down_CPU64"]

    def check(arm, node, actual, expected, bound=1e-4):
        err = float(np.max(abs(np.asarray(actual, float) - expected)))
        rows.append(
            dict(
                arm=arm,
                node=node,
                layer=int(data["probe_layer"]) + 2,
                error=err,
                bound=bound,
                passes=bool(np.isfinite(err) and err <= bound),
            )
        )

    check("shared", "down_projection", data["probe_down_cot"], cot)
    for a, v in enumerate((f, e)):
        check(str(a), "SiLU_primal", v["phi"], v["g"] / (1 + np.exp(-v["g"])))
    for arm in ARMS:
        if arm.startswith("R"):
            ix = permutation(key, int(data["probe_layer"]), arm, len(cot))
            cg = f["u"] * sp(f["g"]) + (e["u"] * sp(e["g"]) - f["u"] * sp(f["g"]))[ix]
            cu = f["phi"] + (e["phi"] - f["phi"])[ix]
        else:
            a, b, c = map(int, arm)
            cg = (f, e)[b]["u"] * sp((f, e)[a]["g"])
            cu = (f, e)[c]["phi"]
        check(arm, "gate_VJP", data[arm + "_probe_gate_AD"], cg * cot)
        check(arm, "up_VJP", data[arm + "_probe_up_AD"], cu * cot)
        check(arm, "unselected_zero", data[arm + "_probe_other_max"], 0.0, 0.0)
        for kind in ("gate", "up"):
            c = int(data[f"{arm}_probe_{kind}_coordinate"])
            ad = float(data[f"{arm}_probe_{kind}_AD"][c])
            estimates = []
            for step in (0.05, 0.025):
                values = data[f"{arm}_probe_{kind}_scores_{step}"]
                fd = float((values[0] - values[1]) / (2 * step))
                check(arm, f"{kind}_FD_{step}", fd, ad, 0.001 + 0.02 * abs(ad))
                estimates.append(fd)
            check(arm, kind + "_halving", estimates[0], estimates[1], 0.002 + 0.03 * abs(ad))
    return rows
