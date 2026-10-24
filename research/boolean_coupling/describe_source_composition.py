"""Zero-model all-case280 figures and summary from the frozen audit CSVs."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .audit229 import sha

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'results/source-composition-280b'
FIG = ROOT / 'docs/figures/source-composition'


def rows(name):
    with (DEST / (name + '.csv')).open() as stream:
        return list(csv.DictReader(stream))


def main():
    assert not FIG.exists()
    FIG.mkdir(parents=True)
    native, trajectory, ledger, local = (rows(n) for n in ('native', 'trajectory', 'ledger', 'local'))
    arms = ['L', 'Q', 'C', 'C1', 'R', 'G0', 'G1', 'G2']
    colors = ['#929aa8', '#d19b38', '#267c9a', '#66a185', '#ac5c69', '#c1c7cf', '#c1c7cf', '#c1c7cf']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))
    summary = {}
    for data, field, ax, title in [(native, 'all_relative', axes[0, 0], 'Original native receiving displacement'),
                                  (trajectory, 'source_relative', axes[0, 1], 'Own-input source trajectory across 28 layers')]:
        vals = [[max(float(r[field]) * 100, .0001) for r in data if r['arm'] == a] for a in arms]
        box = ax.boxplot(vals, patch_artist=True, tick_labels=arms, showfliers=False)
        for patch, col in zip(box['boxes'], colors, strict=True):
            patch.set_facecolor(col)
            patch.set_alpha(.7)
        for i, v in enumerate(vals):
            ax.scatter(np.linspace(i + .83, i + 1.17, len(v)), v, s=9, color=colors[i], alpha=.65, zorder=3)
        ax.set_yscale('log')
        ax.axhline(5 if field == 'all_relative' else 10, color='#555', linestyle='--', linewidth=1)
        ax.set_ylabel('Relative error (%) · log scale')
        ax.set_title(title, loc='left', weight='bold')
        summary[field] = {a: float(np.median(v)) / 100 for a, v in zip(arms, vals, strict=True)}
    ax = axes[1, 0]
    for field, label, color in [('error_norm', 'Total source error', '#303d54'), ('inherited_norm', 'Inherited input effect', '#c38127'), ('new_norm', 'New local approximation error', '#287a99')]:
        values = [np.median([float(r[field]) for r in ledger if int(r['layer']) == i]) for i in range(1, 29)]
        ax.plot(range(1, 29), np.maximum(values, 1e-8), label=label, color=color, linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Source block')
    ax.set_ylabel('Median vector norm · log scale')
    ax.set_title('Exact local/inherited split; norms do not add', loc='left', weight='bold')
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    for incoming, label, color in [('T', 'Native incoming state', '#66a185'), ('C', 'Cubic forecast incoming state', '#287a99')]:
        values = [np.median([float(r['relative']) * 100 for r in local if r['incoming'] == incoming and int(r['layer']) == i]) for i in range(1, 29)]
        ax.plot(range(1, 29), np.maximum(values, .0001), label=label, color=color, linewidth=2)
    ax.set_yscale('log')
    ax.axhline(10, color='#555', linestyle='--', linewidth=1)
    ax.set_xlabel('Source MLP')
    ax.set_ylabel('Median local write-change error (%) · log scale')
    ax.set_title('Same polynomial, different incoming gates', loc='left', weight='bold')
    ax.legend(fontsize=9)
    fig.suptitle('Does a first-source activation rule compose?', fontsize=19, weight='bold', x=.065, ha='left')
    fig.text(.065, .925, '32 old contexts / 16 matched pairs · all receiving operations finite · no fresh confirmation', color='#444')
    fig.text(.065, .024, 'All cases retained. R restores one selected source activation; G0–G2 use norm-matched random writes.\nDashed lines show vector-error limits only; joint criteria also require cosine, source V and actual margin. Zero norms shown at 1e-8.', fontsize=9, color='#444')
    fig.subplots_adjust(top=.87, bottom=.13, hspace=.45, wspace=.28)
    for ext in ('png', 'svg'):
        fig.savefig(FIG / ('source-composition.' + ext), dpi=170, bbox_inches='tight', metadata={'Creator': 'Counterfactual Traces'})
    p = FIG / 'source-composition.svg'
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines()) + '\n')
    (FIG / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (FIG / 'provenance.json').write_text(json.dumps(dict(posthoc=True, model_calls=0, predictions_modified=False,
        script_sha256=sha(__file__), source_csv_sha256={n + '.csv': sha(DEST / (n + '.csv')) for n in ('native', 'trajectory', 'ledger', 'local')},
        figure_sha256={p.name: sha(p) for p in FIG.iterdir() if p.suffix in ('.png', '.svg')}), indent=2) + '\n')


if __name__ == '__main__':
    main()
