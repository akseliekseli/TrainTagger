import json
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np


def _auc(signal_scores, sorted_background_scores, signal_events, background_events):
    lower = np.searchsorted(sorted_background_scores, signal_scores, side="left")
    upper = np.searchsorted(sorted_background_scores, signal_scores, side="right")
    signal_placements = 0.5 * (lower + upper) / len(sorted_background_scores)
    auc = float(np.mean(signal_placements))
    sorted_signal = np.sort(signal_scores)
    lower = np.searchsorted(sorted_signal, sorted_background_scores, side="left")
    upper = np.searchsorted(sorted_signal, sorted_background_scores, side="right")
    background_placements = 1 - 0.5 * (lower + upper) / len(signal_scores)

    variance = 0.0
    for placements, events in ((signal_placements, signal_events),
                               (background_placements, background_events)):
        event_counts = np.bincount(events)
        event_residuals = np.bincount(events, weights=placements - auc)[event_counts > 0]
        n_events = len(event_residuals)
        if n_events < 2:
            return auc, None
        variance += n_events / (n_events - 1) * np.sum(event_residuals**2) / len(placements)**2
    return auc, float(np.sqrt(variance))


def _pt_mask(pt, low, high, pt_field):
    if pt_field == "pt":
        return (pt > low) & (pt <= high)
    return (pt >= low) & (pt < high)


def _plot_auc_pt(
    plot_data, *, pt_field,
    signal_classes=("H_bb", "H_cc", "H_qq", "H_gg"),
    background_classes=("QCD_*",),
    pt_bins=(0, 50, 75, 100, 125, 150, 200, 300, 500, 800),
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
        & np.isfinite(signal[pt_field])
    )
    background_mask = np.isin(background["class_label"], background_ids)
    background_indices = np.flatnonzero(background_mask)
    background_indices = background_indices[np.argsort(background["tagger"][background_indices])]
    sorted_background = background["tagger"][background_indices]
    sorted_background_events = background["event_index"][background_indices]
    pt_bins = np.asarray(pt_bins, dtype=float)
    pt_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])

    if pt_field == "pt":
        background_bins = [
            background_indices[_pt_mask(background["pt"][background_indices], low, high, pt_field)]
            for low, high in zip(pt_bins[:-1], pt_bins[1:])
        ]
        background_selection = "QCD jets in the same reconstructed L1 pT bin as signal"
        xlabel = r"L1 jet $p_T$ [GeV]"
        ylabel = "ROC AUC"
        filename = "auc_jet_pt"
        bin_convention = "lower exclusive, upper inclusive"
    else:
        background_bins = [background_indices] * (len(pt_bins) - 1)
        background_selection = "Same selected QCD jets in every bin; no added background pT cut"
        xlabel = r"Matched GenJet AK8 $p_T$ [GeV]"
        ylabel = "ROC AUC"
        filename = "auc_gen_pt"
        bin_convention = "lower inclusive, upper exclusive"

    background_counts = [len(indices) for indices in background_bins]
    background_event_counts = [len(np.unique(background["event_index"][indices])) for indices in background_bins]
    fig, ax = plt.subplots(figsize=style.FIGURE_SIZE)
    hep.cms.label(ax=ax, llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT,
                  fontsize=style.CMSHEADER_SIZE)
    curves = []
    for mass in (None, *mass_targets):
        selected = signal_mask.copy()
        label = "All matched Higgs masses"
        if mass is not None:
            selected &= (
                (signal["higgs_mass"] >= mass - mass_window)
                & (signal["higgs_mass"] < mass + mass_window)
            )
            label = rf"$m_H={mass:g}\pm{mass_window:g}$ GeV"
        values = []
        errors = []
        counts = []
        event_counts = []
        for low, high, bkg_indices in zip(pt_bins[:-1], pt_bins[1:], background_bins):
            in_bin = selected & _pt_mask(signal[pt_field], low, high, pt_field)
            sig_scores = signal["tagger"][in_bin]
            bkg_scores = background["tagger"][bkg_indices]
            sig_events = signal["event_index"][in_bin]
            bkg_events = background["event_index"][bkg_indices]
            auc, error = (
                _auc(sig_scores, bkg_scores, sig_events, bkg_events)
                if len(sig_scores) and len(bkg_scores) else (None, None)
            )
            values.append(auc)
            errors.append(error)
            counts.append(len(sig_scores))
            event_counts.append(len(np.unique(sig_events)))
            print(f"{filename}: {label}, pT {low:g}--{high:g}: "
                  f"{len(sig_scores)} signal, {len(bkg_scores)} background, AUC={auc}, SE={error}", flush=True)
        sig_scores = signal["tagger"][selected]
        unbinned_auc, unbinned_error = (
            _auc(sig_scores, sorted_background, signal["event_index"][selected], sorted_background_events)
            if len(sig_scores) and len(sorted_background) else (None, None)
        )
        print(f"{filename}: {label}, unbinned: "
              f"{len(sig_scores)} signal, {len(sorted_background)} background, "
              f"AUC={unbinned_auc}", flush=True)
        legend_label = label
        if unbinned_auc is not None:
            legend_label += f" (unbinned: {unbinned_auc:.3f})"
        line, = ax.plot(
            pt_centers, [np.nan if value is None else value for value in values],
            "o-", label=legend_label, linewidth=style.LINEWIDTH,
        )
        valid = np.array([error is not None for error in errors])
        auc_values = np.array([np.nan if value is None else value for value in values])[valid]
        auc_errors = np.array([np.nan if error is None else error for error in errors])[valid]
        ax.errorbar(pt_centers[valid], auc_values,
                    yerr=np.vstack((np.minimum(auc_errors, auc_values),
                                    np.minimum(auc_errors, 1 - auc_values))),
                    fmt="none", ecolor=line.get_color(), capsize=3)
        if unbinned_auc is not None:
            ax.axhline(unbinned_auc, color=line.get_color(), linestyle="--",
                       linewidth=style.LINEWIDTH)
            if unbinned_error is not None:
                ax.axhspan(max(0, unbinned_auc - unbinned_error),
                           min(1, unbinned_auc + unbinned_error),
                           color=line.get_color(), alpha=0.08)
        curves.append({
            "higgs_mass": mass, "auc": values, "signal_counts": counts,
            "auc_se": errors, "signal_event_counts": event_counts,
            "unbinned_auc": unbinned_auc, "unbinned_signal_count": len(sig_scores),
            "unbinned_auc_se": unbinned_error,
        })

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(pt_bins[0], pt_bins[-1])
    ax.set_ylim(0, 1.02)
    ax.axhline(0.5, color="gray", linestyle=":")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=style.SMALL_SIZE,
              title="Solid: pT-binned; dashed: unbinned; errors: 1 SE",
              title_fontsize=style.SMALL_SIZE)
    fig.tight_layout()

    output_dir = Path(output_dir or Path(plot_data["output_directory"]) / "plots/nprong/AUC_pt")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "pt_field": pt_field, "score_classes": list(plot_data["score_classes"]),
        "score_definition": "Sum of requested model probabilities (P(2-prong) for one output)",
        "signal_classes": list(signal_classes), "background_classes": background_names,
        "background_selection": background_selection,
        "unbinned_background_selection": "All selected QCD jets; no added pT cut",
        "unbinned_background_count": len(sorted_background),
        "unbinned_pt_selection": "No pT-bin bounds; includes jets outside the displayed range",
        "background_counts": background_counts, "pt_bins": pt_bins.tolist(),
        "background_event_counts": background_event_counts,
        "uncertainty_method": "First-order event-clustered AUC influence variance; independent signal/background samples",
        "uncertainty_display": "1 statistical standard error, clipped to [0, 1]; null for fewer than 2 events in either sample",
        "uncertainty_caveats": "Asymptotic approximation; unreliable for sparse bins. Zero SE at perfect separation does not imply certainty. No training or systematic uncertainty; no between-bin covariance reported.",
        "bin_convention": bin_convention,
        "signal_selection": "Saved jets, requested truth labels, finite Higgs match and selected pT",
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


def plot_auc_jet_pt(plot_data, **kwargs):
    return _plot_auc_pt(plot_data, pt_field="pt", **kwargs)


def plot_auc_gen_pt(plot_data, **kwargs):
    return _plot_auc_pt(plot_data, pt_field="gen_pt", **kwargs)

