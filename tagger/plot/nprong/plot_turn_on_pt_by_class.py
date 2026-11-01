import json
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from tagger.plot import style
from .plot_turn_on_curve_mass import _leading_indices, _score_pass
from .plot_turn_on_curve_pT import _efficiency_errors


def selected_signal_indices(signal, study_mass, window, signal_classes):
    labels = [signal['source_labels'][name] for name in signal_classes]
    selection = np.isin(signal['class_label'], labels)
    selection &= np.abs(signal['higgs_mass'] - study_mass) < window
    return _leading_indices(signal, selection)


def plot_turn_on_pt_by_class(
    plot_data, *, working_points, signal_classes, study_mass=30,
    window=2.0, pt_bins=None, output_directory=None,
):
    pt_bins = np.asarray(pt_bins if pt_bins is not None else
                         np.concatenate(([0, 40], np.arange(60, 501, 20))), dtype=float)
    output = Path(output_directory or Path(plot_data['output_directory']) /
                  'plots/signal_diagnostics')
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use(hep.style.CMS)
    signal = plot_data['signal']
    indices = selected_signal_indices(signal, study_mass, window, signal_classes)
    centers = (pt_bins[:-1] + pt_bins[1:]) / 2
    colors = ('#5790fc', '#f89c20', '#e42536', '#964a8b')
    result = dict(study_mass=study_mass, mass_field='higgs_mass', window=window,
                  pt_bins=pt_bins.tolist(), signal_classes=list(signal_classes),
                  selection='Leading selected signal jet per event before decay-class split',
                  working_points=working_points, classes={})

    for name in signal_classes:
        selected = indices[signal['class_label'][indices] == signal['source_labels'][name]]
        gen_pt = signal['gen_pt'][selected]
        den, _ = np.histogram(gen_pt, bins=pt_bins)
        curves = []
        fig, ax = plt.subplots(figsize=style.FIGURE_SIZE)
        for i, point in enumerate(working_points):
            threshold, cut = point['pt_threshold'], point['score_cut']
            passes = (signal['pt'][selected] > threshold)
            passes &= _score_pass(signal['tagger'][selected], cut)
            num, _ = np.histogram(gen_pt[passes], bins=pt_bins)
            valid = den > 0
            efficiency = np.full(len(den), np.nan)
            if valid.any():
                values, errors = _efficiency_errors(num[valid], den[valid])
                efficiency[valid] = values
                label = rf'$p_T^{{L1}}>{threshold:g}$ GeV, '
                label += 'No score cut' if cut is None else f'Score > {cut:.3f}'
                if 'achieved_rate_hz' in point:
                    label += f" ({point['achieved_rate_hz']/1000:.2f} kHz)"
                ax.errorbar(centers[valid], values, yerr=errors,
                            xerr=np.diff(pt_bins)[valid]/2, fmt='o',
                            color=colors[i % len(colors)], label=label)
                ax.plot(centers, efficiency, '--', color=colors[i % len(colors)], alpha=.6)
            curves.append(dict(num_counts=num.tolist()))
        result['classes'][name] = dict(selected_events=len(selected), den_counts=den.tolist(), curves=curves)
        ax.set(xlabel=r'Gen jet $p_T$ matched to leading merged L1 jet [GeV]',
               ylabel='Conditional L1 efficiency [%]', xlim=(pt_bins[0], pt_bins[-1]), ylim=(0, 105))
        ax.text(.04, .93, rf'{name}, matched Higgs mass ${study_mass:g}\pm{window:g}$ GeV',
                transform=ax.transAxes, fontsize=style.SMALL_SIZE)
        ax.axhline(100, color='gray', linestyle=':', alpha=.6)
        ax.grid(alpha=.2)
        if den.any():
            ax.legend(fontsize=style.SMALL_SIZE, loc='lower right')
        hep.cms.label(ax=ax, llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT,
                      fontsize=style.CMSHEADER_SIZE)
        fig.tight_layout()
        for suffix in ('png', 'pdf'):
            fig.savefig(output / f'turn_on_{name}_MH{study_mass:g}.{suffix}')
        plt.close(fig)
    with (output / f'turn_on_by_class_MH{study_mass:g}.json').open('w') as stream:
        json.dump(result, stream, indent=2)
    return result

