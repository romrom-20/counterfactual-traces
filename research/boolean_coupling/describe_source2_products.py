"""All-case source2 product predictions and intervention effects; no model calls."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'results/source2-products-281d'
FIG = ROOT / 'docs/figures/source2-products'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows(n):
    with (DEST / (n + '.csv')).open() as stream:
        return list(csv.DictReader(stream))


def main():
    assert not FIG.exists()
    FIG.mkdir(parents=True)
    native, local, effect, selection = (rows(n) for n in ('native', 'local', 'effect', 'selection'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9))
    arms = ['B', 'P', 'H', 'R0', 'R1', 'R2', 'G0', 'G1', 'G2', 'C1']
    colors = ['#818b9d', '#287b99', '#d7a13e', '#c1c8d0', '#c1c8d0', '#c1c8d0', '#ae7d8a', '#ae7d8a', '#ae7d8a', '#5a9a7a']
    summary = {}
    for data, field, ax, title, bound in [(local, 'relative', axes[0, 0], 'Predict the missing MLP2 write', 25),
            (native, 'all_relative', axes[0, 1], 'Keep the original native response', 5),
            (effect, 'all_relative', axes[1, 0], 'Actual continued effect of the missing write', 25)]:
        actual = [[float(r[field]) * 100 for r in data if r['arm'] == a] for a in arms]
        values = [[max(v, .1) for v in arm_values] for arm_values in actual]
        boxes = ax.boxplot(values, patch_artist=True, tick_labels=arms, showfliers=False)
        for patch, color in zip(boxes['boxes'], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(.7)
        for i, val in enumerate(values):
            ax.scatter(np.linspace(i + .86, i + 1.14, len(val)), val, s=8, alpha=.6, color=colors[i], zorder=3)
        ax.axhline(bound, color='#555', linestyle='--', linewidth=1)
        ax.set_yscale('log')
        ax.set_ylabel('Relative error (%) · log scale')
        ax.set_title(title, loc='left', weight='bold')
        summary[title] = {a: float(np.median(v)) / 100 for a, v in zip(arms, actual, strict=True)}
    ax = axes[1, 1]
    mask_arms = ['P', 'H', 'R0', 'R1', 'R2']
    x = np.arange(len(mask_arms))
    energy = [np.median([float(r['individual_energy_fraction']) for r in selection if r['arm'] == a]) for a in mask_arms]
    projection = [np.median([float(r['signed_local_projection']) for r in selection if r['arm'] == a]) for a in mask_arms]
    ax.bar(x - .18, energy, .36, color='#287b99', label='Individual energy before cancellation')
    ax.bar(x + .18, projection, .36, color='#c68f37', label='Signed local-vector projection')
    ax.set_xticks(x, mask_arms)
    ax.axhline(1, color='#777', linewidth=.8)
    ax.set_ylabel('Separate medians; neither is a causal share')
    ax.set_title('Which products does each mask capture?', loc='left', weight='bold')
    ax.legend(fontsize=8, loc='best')
    fig.suptitle('Source MLP2: can a target-blind mask recover the products?', fontsize=17, weight='bold', x=.07, ha='left')
    fig.text(.07, .927, '32 old contexts / 16 matched pairs · 90 selected coordinates · all later operations finite', fontsize=10, color='#444')
    fig.text(.07, .025, 'P: rectifier-residual selector · H: fourth-power selector · R: random masks · G: norm-matched random writes.\nAll cases retained; dashed lines show vector bounds only. Joint criteria also require cosine and actual margins. Exact zeros plotted at 0.1%.', fontsize=8.5, color='#444')
    fig.subplots_adjust(top=.865, bottom=.13, hspace=.47, wspace=.27)
    for ext in ('png', 'svg'):
        fig.savefig(FIG / ('source2-products.' + ext), dpi=170, bbox_inches='tight', metadata={'Creator': 'Counterfactual Traces'})
    p = FIG / 'source2-products.svg'
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines()) + '\n')
    (FIG / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (FIG / 'provenance.json').write_text(json.dumps(dict(posthoc=True, model_calls=0, predictions_modified=False,
        script_sha256=sha(__file__), source_csv_sha256={n + '.csv': sha(DEST / (n + '.csv')) for n in ('native', 'local', 'effect', 'selection')},
        figure_sha256={p.name: sha(p) for p in FIG.iterdir() if p.suffix in ('.png', '.svg')}), indent=2) + '\n')


if __name__ == '__main__':
    main()
