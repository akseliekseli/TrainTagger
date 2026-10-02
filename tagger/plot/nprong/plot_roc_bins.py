"""Class-vs-QCD ROCs from the compact arrays returned by load_plot_data()."""

import json
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np

FLAVOR_META = {
    "H_bb": {"label": r"$H\to b\bar b$"},
    "H_cc": {"label": r"$H\to c\bar c$"},
    "H_qq": {"label": r"$H\to q\bar q$"},
    "H_gg": {"label": r"$H\to gg$"},
}
PT_BINS = (
    (0, 100, r"$0<p_T\leq100$ GeV"),
    (100, 9999, r"$100<p_T$ GeV"),
)
BIN_LS = ("-", "--", ":", "-.")
MASS_LS = ("-", "--", ":", "-.")
SCORE_CUTS = np.linspace(0, 1, 200)


def _empty_counts(num_bins, num_classes):
    return {
        "signal_total": np.zeros((num_bins, num_classes), dtype=np.int64),
        "signal_pass": np.zeros((num_bins, num_classes, len(SCORE_CUTS)), dtype=np.int64),
        "background_total": np.zeros(num_bins, dtype=np.int64),
        "background_pass": np.zeros((num_bins, num_classes, len(SCORE_CUTS)), dtype=np.int64),
    }


def _passing(scores):
    return len(scores) - np.searchsorted(np.sort(scores), SCORE_CUTS, side="left")


def _count_signal(counts, bin_index, sig_scores, selection, flavor_decay_masks):
    for flavor_index, flavor in enumerate(flavor_decay_masks):
        sig_scores_sel = sig_scores[selection & flavor_decay_masks[flavor], flavor_index]
        counts["signal_total"][bin_index, flavor_index] += len(sig_scores_sel)
        counts["signal_pass"][bin_index, flavor_index] += _passing(sig_scores_sel)


def _count_background(counts, bin_index, bkg_scores, selection):
    bkg_scores_sel = bkg_scores[selection]
    counts["background_total"][bin_index] += len(bkg_scores_sel)
    for flavor_index in range(bkg_scores.shape[1]):
        counts["background_pass"][bin_index, flavor_index] += _passing(bkg_scores_sel[:, flavor_index])


def _accumulate_signal(data, l1_jets, flavor_decay_masks, mass_targets, mass_window):
    sig_pt_flat = l1_jets["pt"]
    sig_sc_flat = l1_jets["tagger"]
    sig_hmass_flat = l1_jets["higgs_mass"]

    for bin_index, (pt_lo, pt_hi, _) in enumerate(PT_BINS):
        sig_mask = (sig_pt_flat > pt_lo) & (sig_pt_flat <= pt_hi)
        _count_signal(data["ROC_pt_bins"], bin_index, sig_sc_flat, sig_mask, flavor_decay_masks)

    for mass_index, mass_target in enumerate(mass_targets):
        m_lo, m_hi = mass_target - mass_window, mass_target + mass_window
        mass_mask = (sig_hmass_flat >= m_lo) & (sig_hmass_flat < m_hi)
        selection = mass_mask & (sig_pt_flat > 0)
        _count_signal(data["ROC_mass_slices"], mass_index, sig_sc_flat, selection, flavor_decay_masks)

        fixed_mass_data = data[f"ROC_fixed_mass_pt_bins_m{mass_target:g}"]
        for bin_index, (pt_lo, pt_hi, _) in enumerate(PT_BINS):
            selection = mass_mask & (sig_pt_flat > pt_lo) & (sig_pt_flat <= pt_hi)
            _count_signal(fixed_mass_data, bin_index, sig_sc_flat, selection, flavor_decay_masks)


def _accumulate_background(data, l1_jets, mass_targets):
    bkg_pt_flat = l1_jets["pt"]
    bkg_sc_flat = l1_jets["tagger"]
    for bin_index, (pt_lo, pt_hi, _) in enumerate(PT_BINS):
        bkg_mask = (bkg_pt_flat > pt_lo) & (bkg_pt_flat <= pt_hi)
        _count_background(data["ROC_pt_bins"], bin_index, bkg_sc_flat, bkg_mask)

    # Background uses the same pT ranges, with no generator-mass selection.
    for mass_index, mass_target in enumerate(mass_targets):
        _count_background(data["ROC_mass_slices"], mass_index, bkg_sc_flat, bkg_pt_flat > 0)
        fixed_mass_data = data[f"ROC_fixed_mass_pt_bins_m{mass_target:g}"]
        for bin_index, (pt_lo, pt_hi, _) in enumerate(PT_BINS):
            bkg_mask = (bkg_pt_flat > pt_lo) & (bkg_pt_flat <= pt_hi)
            _count_background(fixed_mass_data, bin_index, bkg_sc_flat, bkg_mask)


def _make_figure():
    import matplotlib.pyplot as plt
    import mplhep as hep
    from tagger.plot import style

    fig, ax = plt.subplots(figsize=style.FIGURE_SIZE)
    hep.cms.label(ax=ax, llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT,
                  fontsize=style.CMSHEADER_SIZE)
    return fig, ax


def _plot_curve(ax, counts, bin_index, flavor_index, linestyle, description, min_jets, flavor):
    from tagger.plot import style

    n_sig = counts["signal_total"][bin_index, flavor_index]
    n_bkg = counts["background_total"][bin_index]
    if min(n_sig, n_bkg) < min_jets:
        print(f"Skipping {flavor}, {description}: {n_sig} signal, {n_bkg} background jets")
        return False
    tpr = counts["signal_pass"][bin_index, flavor_index] / n_sig
    fpr = counts["background_pass"][bin_index, flavor_index] / n_bkg
    ax.plot(tpr, np.ma.masked_equal(fpr, 0), color=f"C{flavor_index}",
            linestyle=linestyle, linewidth=style.LINEWIDTH)
    return True


def _save(fig, ax, output_path, bin_labels, linestyles, curves_drawn, score_classes, mass_label=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from tagger.plot import style

    if not curves_drawn:
        plt.close(fig)
        print(f"No populated curves for {output_path.stem}; no empty figure saved")
        return
    flavor_handles = [
        Line2D([], [], color=f"C{index}", label=FLAVOR_META.get(name, {"label": name})["label"],
               linewidth=style.LINEWIDTH)
        for index, name in enumerate(score_classes)
    ]
    bin_handles = [
        Line2D([], [], color="gray", label=label, linewidth=style.LINEWIDTH,
               linestyle=linestyles[index % len(linestyles)])
        for index, label in enumerate(bin_labels)
    ]
    handles = flavor_handles + bin_handles
    if mass_label is not None:
        handles.insert(0, Line2D([], [], color="none", label=mass_label))
    ax.legend(handles=handles, loc="lower right", frameon=False)
    ax.set_xlabel("Signal Efficiency")
    ax.set_ylabel("Background Efficiency")
    ax.set_yscale("log")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(f"{output_path}.{suffix}")
    plt.close(fig)


def plot_ROC_pt_bins(data, output_path="plots/ROC_pt_bins", min_jets=10, *, score_classes):
    fig, ax = _make_figure()
    curves_drawn = 0
    for bin_index, (pt_lo, pt_hi, _) in enumerate(PT_BINS):
        linestyle = BIN_LS[bin_index % len(BIN_LS)]
        for flavor_index, flavor in enumerate(score_classes):
            curves_drawn += _plot_curve(ax, data["ROC_pt_bins"], bin_index, flavor_index,
                                       linestyle, f"pT {pt_lo}--{pt_hi}", min_jets, flavor)
    bin_labels = [label for _, _, label in PT_BINS]
    _save(fig, ax, Path(output_path), bin_labels, BIN_LS, curves_drawn, score_classes)


def plot_ROC_mass_slices(data, output_path="plots/ROC_mass_slices",
                         mass_targets=(30, 50, 90), mass_window=2.0, min_jets=10, *, score_classes):
    fig, ax = _make_figure()
    curves_drawn = 0
    for mass_index, mass_target in enumerate(mass_targets):
        linestyle = MASS_LS[mass_index % len(MASS_LS)]
        for flavor_index, flavor in enumerate(score_classes):
            curves_drawn += _plot_curve(ax, data["ROC_mass_slices"], mass_index, flavor_index,
                                       linestyle, f"mass {mass_target}", min_jets, flavor)
    mass_labels = [rf"$m_H={mass:g}\pm{mass_window:g}$ GeV" for mass in mass_targets]
    _save(fig, ax, Path(output_path), mass_labels, MASS_LS, curves_drawn, score_classes)


def plot_ROC_fixed_mass_pt_bins(data, output_path="plots/ROC_fixed_mass_pt_bins",
                               mass_target=30, mass_window=2.0, min_jets=10, *, score_classes):
    fig, ax = _make_figure()
    curves_drawn = 0
    fixed_mass_data = data[f"ROC_fixed_mass_pt_bins_m{mass_target:g}"]
    for bin_index, (pt_lo, pt_hi, _) in enumerate(PT_BINS):
        linestyle = BIN_LS[bin_index % len(BIN_LS)]
        for flavor_index, flavor in enumerate(score_classes):
            curves_drawn += _plot_curve(ax, fixed_mass_data, bin_index, flavor_index,
                                       linestyle, f"mass {mass_target}, pT {pt_lo}--{pt_hi}", min_jets, flavor)
    bin_labels = [label for _, _, label in PT_BINS]
    mass_label = rf"$m_H={mass_target:g}\pm{mass_window:g}$ GeV"
    _save(fig, ax, Path(output_path), bin_labels, BIN_LS, curves_drawn, score_classes, mass_label)


def plot_ROC_bins(plot_data, *, background_classes, output_dir=None,
                  mass_targets=(), mass_window=2.0, min_jets=10):
    """Calculate and draw per-class ROCs without ROOT reads or model inference."""
    mass_targets = tuple(mass_targets)
    score_classes = tuple(plot_data["score_classes"])
    if not np.isfinite(mass_window) or mass_window <= 0 or min_jets < 1:
        raise ValueError("mass_window and min_jets must be positive")
    if len(set(mass_targets)) != len(mass_targets) or any(
        not np.isfinite(mass) or mass < 0 for mass in mass_targets
    ):
        raise ValueError("Mass targets must be unique, finite and nonnegative")
    if isinstance(background_classes, str) or not background_classes:
        raise ValueError("Pass background_classes as a nonempty tuple, e.g. ('QCD*',)")
    signal = plot_data["signal"]
    background = plot_data["background"]
    matched_classes = []
    for pattern in background_classes:
        matches = [name for name in background["source_labels"] if fnmatchcase(name, pattern)]
        if not matches:
            raise ValueError(f"No background labels match {pattern!r}; available: {list(background['source_labels'])}")
        matched_classes.extend(matches)
    matched_classes = list(dict.fromkeys(matched_classes))
    background_ids = [background["source_labels"][name] for name in matched_classes]
    print("ROC scores: individual class probabilities; background labels:", matched_classes, flush=True)

    counts = {"ROC_pt_bins": _empty_counts(len(PT_BINS), len(score_classes))}
    if mass_targets:
        counts["ROC_mass_slices"] = _empty_counts(len(mass_targets), len(score_classes))
    for mass in mass_targets:
        counts[f"ROC_fixed_mass_pt_bins_m{mass:g}"] = _empty_counts(len(PT_BINS), len(score_classes))

    # Slice only compact arrays; never copy the full dataset into each selection.
    for start in range(0, len(signal["pt"]), 100000):
        stop = start + 100000
        mass = signal["higgs_mass"][start:stop]
        labels = signal["class_label"][start:stop]
        flavor_masks = {
            name: np.isfinite(mass) & (labels == signal["source_labels"][name])
            for name in score_classes
        }
        jets = {"pt": signal["pt"][start:stop], "tagger": signal["scores"][start:stop],
                "higgs_mass": mass}
        _accumulate_signal(counts, jets, flavor_masks, mass_targets, mass_window)
    for start in range(0, len(background["pt"]), 100000):
        stop = start + 100000
        selected = np.isin(background["class_label"][start:stop], background_ids)
        jets = {"pt": background["pt"][start:stop][selected],
                "tagger": background["scores"][start:stop][selected]}
        _accumulate_background(counts, jets, mass_targets)

    output_dir = Path(output_dir or Path(plot_data["output_directory"]) / "plots/nprong/ROC_bins")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "score_classes": score_classes, "thresholds": SCORE_CUTS.tolist(),
        "score_definition": "Raw P(flavor) for signal and background; no sum or ratio",
        "background_classes": matched_classes, "background_labels": background["source_labels"],
        "signal_dir": signal["directory"], "background_dir": background["directory"],
        "flavors": score_classes, "mass_targets": mass_targets, "mass_window": mass_window,
        "source_labels": signal["source_labels"], "max_chunks": plot_data["max_chunks"],
        "signal_chunks_read": signal["n_chunks"], "background_chunks_read": background["n_chunks"],
        "truth_definition": "PDG25 spatial matching + saved per-jet class_label",
        "pt_bins": [[lo, hi] for lo, hi, _ in PT_BINS],
        "counts": {name: {key: value.tolist() for key, value in count.items()} for name, count in counts.items()},
    }
    with (output_dir / "ROC_bins_counts.json").open("w") as stream:
        json.dump(result, stream, indent=2)
    plot_ROC_pt_bins(counts, output_dir / "ROC_pt_bins", min_jets=min_jets, score_classes=score_classes)
    if mass_targets:
        plot_ROC_mass_slices(counts, output_dir / "ROC_mass_slices",
                            mass_targets=mass_targets, mass_window=mass_window, min_jets=min_jets,
                            score_classes=score_classes)
    for mass in mass_targets:
        plot_ROC_fixed_mass_pt_bins(counts, output_dir / f"ROC_fixed_mass_pt_bins_m{mass:g}",
                                   mass_target=mass, mass_window=mass_window, min_jets=min_jets,
                                   score_classes=score_classes)
    return result

