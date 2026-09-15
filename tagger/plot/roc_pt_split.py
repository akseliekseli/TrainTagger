# flake8: noqa
import os

import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from sklearn.metrics import auc, roc_curve

from tagger.plot import style

__all__ = [
    "ROC_pt_split",
    "ROC_binary_pt_split",
    "ROC_jets_pt_split",
    "ROC_taus_pt_split",
]

# Cycled through as pT bins are added to a plot (colors are used for
# class/category instead, same as the original single-pT-bin functions)
BIN_LINESTYLES = ["-", "--", "-.", ":"]


def _bin_label(lo, hi):
    """Human readable + filename-safe label for a pt bin."""
    if np.isinf(hi):
        human = rf"$p_T$ > {lo:g} GeV"
        fname = f"pt_gt_{lo:g}"
    elif lo <= 0:
        human = rf"$p_T$ < {hi:g} GeV"
        fname = f"pt_lt_{hi:g}"
    else:
        human = rf"{lo:g} < $p_T$ < {hi:g} GeV"
        fname = f"pt_{lo:g}_to_{hi:g}"
    return human, fname


def _pt_masks(pt_test, pt_bins):
    """Yield (mask, human_label, fname_label, linestyle) for each consecutive pair of pt_bins edges."""
    for i, (lo, hi) in enumerate(zip(pt_bins[:-1], pt_bins[1:])):
        mask = (pt_test >= lo) & (pt_test < hi)
        human, fname = _bin_label(lo, hi)
        linestyle = BIN_LINESTYLES[i % len(BIN_LINESTYLES)]
        yield mask, human, fname, linestyle


def _combined_fname(pt_bins):
    """Filename tag summarizing all bin edges, e.g. pt_0_100_inf."""
    parts = ["inf" if np.isinf(edge) else f"{edge:g}" for edge in pt_bins]
    return "pt_" + "_".join(parts)


def ROC_pt_split(
    y_pred, y_test, class_labels, pt_test, plot_dir, pt_bins=[0, 100, np.inf]
):
    """
    One-vs-rest ROC curve per class (like the `ROC` function), with every
    pT bin defined by `pt_bins` drawn on the same axes. Color = class
    (same colormap/order as `ROC`), linestyle = pT bin.

    Returns: dict {bin_fname_label: {class_label: auc}}
    """
    save_dir = os.path.join(plot_dir, "roc_pt_split")
    os.makedirs(save_dir, exist_ok=True)

    colormap = matplotlib.colormaps["Set1"].resampled(len(class_labels))

    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(
        llabel=style.CMSHEADER_LEFT,
        rlabel=style.CMSHEADER_RIGHT,
        ax=ax,
        fontsize=style.CMSHEADER_SIZE,
    )

    all_auc = {}
    auc_list = []

    for mask, human_label, fname_label, linestyle in _pt_masks(pt_test, pt_bins):
        if mask.sum() == 0:
            print(f"[ROC_pt_split] No events in bin {human_label}, skipping.")
            continue

        y_pred_bin = y_pred[mask]
        y_test_bin = y_test[mask]

        bin_auc = {}
        for i, class_label in enumerate(class_labels):
            y_true = y_test_bin[:, i]
            y_score = y_pred_bin[:, i]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            bin_auc[class_label] = roc_auc
            auc_list.append(roc_auc)

            ax.plot(
                tpr,
                fpr,
                label=f"{style.CLASS_LABEL_STYLE[class_label]}, {human_label} (AUC = {roc_auc:.2f})",
                color=colormap(i),
                linestyle=linestyle,
                linewidth=style.LINEWIDTH,
            )

        all_auc[fname_label] = bin_auc

    random_x = np.linspace(0, 1, 100)
    ax.plot(
        random_x,
        random_x,
        linestyle="--",
        color="gray",
        label="Random Classifier (AUC = 0.5)",
        linewidth=style.LINEWIDTH / 2,
    )
    auc_list.append(0.5)

    ax.grid(True)
    ax.set_ylabel("Mistag Rate", fontsize=32)
    ax.set_xlabel("Signal Efficiency", fontsize=32)

    handles, labels = ax.get_legend_handles_labels()
    order = np.argsort(auc_list)
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc="upper left",
        ncol=2,
        fontsize=style.SMALL_SIZE + 3,
    )

    ax.set_yscale("log")
    ax.set_ylim([1e-3, 1.1])

    save_path = os.path.join(save_dir, f"basic_ROC_{_combined_fname(pt_bins)}")
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_path}.png", bbox_inches="tight")
    plt.close(fig)

    return all_auc


def ROC_binary_pt_split(
    y_pred,
    y_test,
    class_labels,
    pt_test,
    plot_dir,
    class_pair,
    pt_bins=[0, 100, np.inf],
    signal_proc=None,
):
    """
    Binary ROC between two specific classes (like `ROC_binary`), with every
    pT bin drawn on the same axes (same color, different linestyle per bin).
    """
    assert class_pair[0] in class_labels and class_pair[1] in class_labels, (
        "Both class_pair labels must exist in class_labels"
    )

    save_dir = os.path.join(plot_dir, "roc_binary_pt_split")
    os.makedirs(save_dir, exist_ok=True)

    idx1, idx2 = class_labels[class_pair[0]], class_labels[class_pair[1]]

    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(
        llabel=style.CMSHEADER_LEFT,
        rlabel=style.CMSHEADER_RIGHT,
        ax=ax,
        fontsize=style.CMSHEADER_SIZE,
    )

    for mask, human_label, fname_label, linestyle in _pt_masks(pt_test, pt_bins):
        if mask.sum() == 0:
            print(f"[ROC_binary_pt_split] No events in bin {human_label}, skipping.")
            continue

        y_true1, y_true2 = y_test[mask, idx1], y_test[mask, idx2]
        y_score1, y_score2 = y_pred[mask, idx1], y_pred[mask, idx2]

        selection = (y_true1 == 1) | (y_true2 == 1)
        if selection.sum() == 0:
            print(
                f"[ROC_binary_pt_split] No {class_pair} events in bin {human_label}, skipping."
            )
            continue

        y_true_binary = y_true1[selection]
        y_score_binary = y_score1[selection] / (
            y_score1[selection] + y_score2[selection]
        )

        fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
        roc_auc = auc(fpr, tpr)

        ax.plot(
            tpr,
            fpr,
            label=f"{style.CLASS_LABEL_STYLE[class_pair[0]]} vs {style.CLASS_LABEL_STYLE[class_pair[1]]}, "
            f"{human_label} (AUC = {roc_auc:.2f})",
            color="blue",
            linestyle=linestyle,
            linewidth=5,
        )

    ax.grid(True)
    ax.set_ylabel("Mistag Rate")
    ax.set_xlabel("Signal Efficiency")
    leg = ax.legend(loc="lower right", fontsize=style.SMALL_SIZE + 3, title=signal_proc)
    leg._legend_box.align = "left"
    ax.set_yscale("log")
    ax.set_ylim([1e-3, 1.1])

    save_path = os.path.join(
        save_dir, f"ROC_{class_pair[0]}_vs_{class_pair[1]}_{_combined_fname(pt_bins)}"
    )
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_path}.png", bbox_inches="tight")
    plt.close(fig)


def ROC_jets_pt_split(
    y_pred,
    y_test,
    class_labels,
    pt_test,
    plot_dir,
    pt_bins=[0, 100, np.inf],
    process_label=None,
):
    """
    Combined ROC for light vs b, charm, gluon (like `ROC_jets`), with every
    pT bin drawn on the same axes. Color = background category (b/charm/
    gluon), linestyle = pT bin.
    """
    save_dir = os.path.join(plot_dir, "roc_jets_pt_split")
    os.makedirs(save_dir, exist_ok=True)

    light_idx = [class_labels["light"]]
    targets = {
        "b": [class_labels["b"]],
        "charm": [class_labels["charm"]],
        "gluon": [class_labels["gluon"]],
    }
    colormap = matplotlib.colormaps["Set1"].resampled(len(targets))

    def compute_roc_inputs(y_pred_bin, y_test_bin, signal_indices, background_indices):
        signal_mask = sum(y_test_bin[:, idx] for idx in signal_indices) > 0
        background_mask = sum(y_test_bin[:, idx] for idx in background_indices) > 0
        total_mask = signal_mask | background_mask

        signal_scores = sum(y_pred_bin[:, idx] for idx in signal_indices)
        background_scores = sum(y_pred_bin[:, idx] for idx in background_indices)
        total_scores = signal_scores + background_scores

        y_true = signal_mask[total_mask]
        y_score = (signal_scores / total_scores)[total_mask]
        return y_true, y_score

    plt.figure(figsize=style.FIGURE_SIZE)
    hep.cms.label(
        llabel=style.CMSHEADER_LEFT,
        rlabel=style.CMSHEADER_RIGHT,
        fontsize=style.CMSHEADER_SIZE,
    )

    for mask, human_label, fname_label, linestyle in _pt_masks(pt_test, pt_bins):
        if mask.sum() == 0:
            print(f"[ROC_jets_pt_split] No events in bin {human_label}, skipping.")
            continue

        y_pred_bin = y_pred[mask]
        y_test_bin = y_test[mask]

        for i, (label, bkg_idx) in enumerate(targets.items()):
            y_true, y_score = compute_roc_inputs(
                y_pred_bin, y_test_bin, light_idx, bkg_idx
            )
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            plt.plot(
                tpr,
                fpr,
                label=f"light vs {label}, {human_label} (AUC = {roc_auc:.2f})",
                color=colormap(i),
                linestyle=linestyle,
                linewidth=style.LINEWIDTH,
            )

    plt.grid(True)
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Mistag Rate")
    plt.yscale("log")
    plt.ylim(1e-3, 1.1)
    leg = plt.legend(
        loc="lower right", fontsize=style.SMALL_SIZE + 3, title=process_label
    )
    leg._legend_box.align = "left"

    save_path = os.path.join(
        save_dir, f"ROC_light_vs_all_jets_{_combined_fname(pt_bins)}"
    )
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_path}.png", bbox_inches="tight")
    plt.close()


def ROC_taus_pt_split(
    y_pred,
    y_test,
    class_labels,
    pt_test,
    plot_dir,
    pt_bins=[0, 100, np.inf],
    signal_proc=None,
):
    """
    Combined ROC for taus vs jets/muons/electrons (like `ROC_taus`), with
    every pT bin drawn on the same axes. Color = background category,
    linestyle = pT bin.
    """
    save_dir = os.path.join(plot_dir, "roc_taus_pt_split")
    os.makedirs(save_dir, exist_ok=True)

    tau_indices = [class_labels["taup"], class_labels["taum"]]
    jet_indices = [class_labels[key] for key in ["b", "charm", "light", "gluon"]]
    muon_indices = [class_labels["muon"]]
    electron_indices = [class_labels["electron"]]

    targets = [
        (r"$\tau_h^{\pm}$ vs Jets (b, c, light, gluon)", jet_indices),
        (r"$\tau_h^{\pm}$ vs Muons", muon_indices),
        (r"$\tau_h^{\pm}$ vs Electrons", electron_indices),
    ]
    colormap = matplotlib.colormaps["Set1"].resampled(len(targets))

    def compute_roc_inputs(y_pred_bin, y_test_bin, signal_indices, background_indices):
        signal_mask = sum(y_test_bin[:, idx] for idx in signal_indices) > 0
        total_mask = signal_mask | (
            sum(y_test_bin[:, idx] for idx in background_indices) > 0
        )

        signal_scores = sum(y_pred_bin[:, idx] for idx in signal_indices)
        background_scores = sum(y_pred_bin[:, idx] for idx in background_indices)
        total_scores = signal_scores + background_scores

        return signal_mask[total_mask], (signal_scores / total_scores)[total_mask]

    plt.figure(figsize=style.FIGURE_SIZE)
    hep.cms.label(
        llabel=style.CMSHEADER_LEFT,
        rlabel=style.CMSHEADER_RIGHT,
        fontsize=style.CMSHEADER_SIZE,
    )

    for mask, human_label, fname_label, linestyle in _pt_masks(pt_test, pt_bins):
        if mask.sum() == 0:
            print(f"[ROC_taus_pt_split] No events in bin {human_label}, skipping.")
            continue

        y_pred_bin = y_pred[mask]
        y_test_bin = y_test[mask]

        for i, (label, bkg_indices) in enumerate(targets):
            y_true, y_score = compute_roc_inputs(
                y_pred_bin, y_test_bin, tau_indices, bkg_indices
            )
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            plt.plot(
                tpr,
                fpr,
                label=f"{label}, {human_label} (AUC = {roc_auc:.2f})",
                color=colormap(i),
                linestyle=linestyle,
                linewidth=style.LINEWIDTH,
            )

    plt.grid(True)
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Mistag Rate")
    plt.yscale("log")
    plt.ylim(1e-3, 1.1)
    leg = plt.legend(loc="upper left", fontsize=style.SMALL_SIZE + 3, title=signal_proc)
    leg._legend_box.align = "left"

    save_path = os.path.join(save_dir, f"ROC_taus_combined_{_combined_fname(pt_bins)}")
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_path}.png", bbox_inches="tight")
    plt.close()
