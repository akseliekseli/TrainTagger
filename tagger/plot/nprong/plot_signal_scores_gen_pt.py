import json
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from tagger.plot import style
from .plot_turn_on_pt_by_class import selected_signal_indices


def plot_signal_scores_gen_pt(
    plot_data, *, signal_classes, study_mass=30, window=2.0,
    gen_pt_bins=(0, 100, 150, 200, 300, 400, 500), score_bins=None,
    score_cuts=(0.490, 0.161), l1_pt_min=None, output_directory=None,
):
    score_bins = np.asarray(score_bins if score_bins is not None else np.linspace(0, 1, 51))
    output = Path(output_directory or Path(plot_data['output_directory']) /
                  'plots/signal_diagnostics')
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use(hep.style.CMS)
    signal = plot_data['signal']
    indices = selected_signal_indices(signal, study_mass, window, signal_classes)
    if l1_pt_min is not None:
        indices = indices[signal['pt'][indices] > l1_pt_min]
    groups = {'inclusive': indices}
    groups.update({name: indices[signal['class_label'][indices] == signal['source_labels'][name]]
                   for name in signal_classes})
    result = dict(study_mass=study_mass, mass_field='higgs_mass', window=window,
                  signal_classes=list(signal_classes), gen_pt_bins=list(gen_pt_bins),
                  score_bins=score_bins.tolist(), score_cuts=list(score_cuts),
                  l1_pt_min=l1_pt_min, groups={})
    for name, selected in groups.items():
        fig, ax = plt.subplots(figsize=style.FIGURE_SIZE)
        records = []
        for low, high in zip(gen_pt_bins[:-1], gen_pt_bins[1:]):
            mask = (signal['gen_pt'][selected] >= low) & (signal['gen_pt'][selected] < high)
            scores = signal['tagger'][selected][mask]
            counts, _ = np.histogram(scores, bins=score_bins)
            if len(scores):
                ax.stairs(counts / len(scores), score_bins,
                          label=rf'${low:g}\leq p_T^{{gen}}<{high:g}$ GeV ($N={len(scores)}$)')
            records.append(dict(gen_pt_low=low, gen_pt_high=high, count=len(scores),
                                score_counts=counts.tolist(),
                                passing_counts=[int(np.count_nonzero(scores > cut)) for cut in score_cuts]))
        for cut, color in zip(score_cuts, ('#5790fc', '#f89c20', '#e42536', '#964a8b')):
            ax.axvline(cut, linestyle='--', color=color, label=f'Score cut {cut:.3f}')
        title = rf'{name}, matched Higgs mass ${study_mass:g}\pm{window:g}$ GeV'
        if l1_pt_min is not None:
            title += rf', $p_T^{{L1}}>{l1_pt_min:g}$ GeV'
        ax.text(.03, .96, title, transform=ax.transAxes, va='top', fontsize=style.SMALL_SIZE)
        ax.set(xlabel='Tagger score', ylabel='Fraction of selected jets per score bin',
               xlim=(score_bins[0], score_bins[-1]))
        ax.grid(alpha=.2)
        ax.legend(fontsize=style.SMALL_SIZE, loc='best')
        hep.cms.label(ax=ax, llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT,
                      fontsize=style.CMSHEADER_SIZE)
        fig.tight_layout()
        stem = f'scores_gen_pt_{name}_MH{study_mass:g}'
        if l1_pt_min is not None:
            stem += f'_L1pt{l1_pt_min:g}'
        for suffix in ('png', 'pdf'):
            fig.savefig(output / f'{stem}.{suffix}')
        plt.close(fig)
        result['groups'][name] = records
    suffix = '' if l1_pt_min is None else f'_L1pt{l1_pt_min:g}'
    with (output / f'scores_gen_pt_MH{study_mass:g}{suffix}.json').open('w') as stream:
        json.dump(result, stream, indent=2)
    return result

