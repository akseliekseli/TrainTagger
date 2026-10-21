import json
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np


def plot_auc_mass(
    plot_data, *, mass_field="mass",
    signal_classes=("H_bb", "H_cc", "H_qq", "H_gg"),
    background_classes=("QCD*",),
    mass_bins=(0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200),
    mass_targets=(), mass_window=2.0, output_dir=None,
):
    import matplotlib.pyplot as plt
    import mplhep as hep
    from tagger.plot import style

    signal = plot_data["signal"]
    background = plot_data["background"]
    signal_ids = [signal["source_labels"][name] for name in signal_classes]
    background_names = [
        name for name in background["source_labels"]
        if any(fnmatchcase(name, pattern) for pattern in background_classes)
    ]
    background_ids = [background["source_labels"][name] for name in background_names]
    signal_mask = (
        np.isin(signal["class_label"], signal_ids)
        & np.isfinite(signal["higgs_mass"])
        & np.isfinite(signal[mass_field])
    )
    background_mask = np.isin(background["class_label"], background_ids)
    if mass_field == "mass":
        background_mask &= np.isfinite(background["mass"])
    sorted_background = np.sort(background["tagger"][background_mask])
    mass_bins = np.asarray(mass_bins, dtype=float)
    mass_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])

    if mass_field == "mass":
        background_scores = [
            np.sort(background["tagger"][
                background_mask & (background["mass"] >= low) & (background["mass"] < high)
            ])
            for low, high in zip(mass_bins[:-1], mass_bins[1:])
        ]
        background_selection = "QCD jets in the same L1 jet mass bin as signal"
        xlabel = "L1 jet mass [GeV]"
        filename = "auc_jet_mass"
    else:
        background_scores = [sorted_background] * (len(mass_bins) - 1)
        background_selection = "Same selected QCD jets in every bin; no background mass cut"
        xlabel = "Matched AK8 gen jet mass [GeV]"
        filename = "auc_gen_mass"

    fig, ax = plt.subplots(figsize=style.FIGURE_SIZE)
    hep.cms.label(ax=ax, llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT,
                  fontsize=style.CMSHEADER_SIZE)
    curves = []
    for higgs_mass in (None, *mass_targets):
        selected = signal_mask.copy()
        label = "All matched Higgs masses"
        if higgs_mass is not None:
            selected &= (
                (signal["higgs_mass"] >= higgs_mass - mass_window)
                & (signal["higgs_mass"] < higgs_mass + mass_window)
            )
            label = rf"$m_H={higgs_mass:g}\pm{mass_window:g}$ GeV"
        values = []
        counts = []
        for low, high, bkg_scores in zip(mass_bins[:-1], mass_bins[1:], background_scores):
            in_bin = selected & (signal[mass_field] >= low) & (signal[mass_field] < high)
            sig_scores = signal["tagger"][in_bin]
            auc = None
            if len(sig_scores) and len(bkg_scores):
                lower = np.searchsorted(bkg_scores, sig_scores, side="left")
                upper = np.searchsorted(bkg_scores, sig_scores, side="right")
                auc = float(np.mean(0.5 * (lower + upper)) / len(bkg_scores))
            values.append(auc)
            counts.append(len(sig_scores))
            print(f"{filename}: {label}, jet mass {low:g}--{high:g}: "
                  f"{len(sig_scores)} signal, {len(bkg_scores)} background, AUC={auc}", flush=True)

        sig_scores = signal["tagger"][selected]
        unbinned_auc = None
        if len(sig_scores) and len(sorted_background):
            lower = np.searchsorted(sorted_background, sig_scores, side="left")
            upper = np.searchsorted(sorted_background, sig_scores, side="right")
            unbinned_auc = float(np.mean(0.5 * (lower + upper)) / len(sorted_background))
        legend_label = label
        if unbinned_auc is not None:
            legend_label += f" (unbinned: {unbinned_auc:.3f})"
        line, = ax.plot(
            mass_centers, [np.nan if value is None else value for value in values],
            "o-", label=legend_label, linewidth=style.LINEWIDTH,
        )
        if unbinned_auc is not None:
            ax.axhline(unbinned_auc, color=line.get_color(), linestyle="--",
                       linewidth=style.LINEWIDTH)
        curves.append({
            "higgs_mass": higgs_mass, "auc": values, "signal_counts": counts,
            "unbinned_auc": unbinned_auc, "unbinned_signal_count": len(sig_scores),
        })

    ax.set_xlabel(xlabel)
    ax.set_ylabel("ROC AUC")
    ax.set_xlim(mass_bins[0], mass_bins[-1])
    ax.set_ylim(0, 1.02)
    ax.axhline(0.5, color="gray", linestyle=":")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=style.SMALL_SIZE,
              title="Solid: jet-mass-binned; dashed: unbinned",
              title_fontsize=style.SMALL_SIZE)
    fig.tight_layout()

    output_dir = Path(output_dir or Path(plot_data["output_directory"]) / "plots/nprong/AUC_mass")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "mass_field": mass_field, "mass_bins": mass_bins.tolist(),
        "score_classes": list(plot_data["score_classes"]),
        "score_definition": "Sum of selected model probabilities; unchanged from plot_data",
        "signal_classes": list(signal_classes), "background_classes": background_names,
        "signal_selection": "All saved jets with requested truth labels and finite Higgs match and jet mass",
        "background_selection": background_selection,
        "background_counts": [len(scores) for scores in background_scores],
        "unbinned_background_count": len(sorted_background),
        "unbinned_mass_selection": "No jet-mass-bin bounds; includes jets outside the displayed range",
        "bin_convention": "lower inclusive, upper exclusive (including the final bin)",
        "mass_window": mass_window, "curves": curves,
        "signal_dir": signal["directory"], "background_dir": background["directory"],
        "max_chunks": plot_data["max_chunks"],
        "signal_chunks_read": signal["n_chunks"], "background_chunks_read": background["n_chunks"],
    }
    with (output_dir / f"{filename}.json").open("w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{filename}.{suffix}")
    plt.close(fig)
    return result

