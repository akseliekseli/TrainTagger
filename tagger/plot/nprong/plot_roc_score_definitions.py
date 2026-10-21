"""Compare three Higgs/QCD discriminants on the same truth-selected jets."""

import json
from fnmatch import fnmatchcase
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import auc, roc_curve

from tagger.plot import style


def _ratio(numerator, denominator):
    return np.divide(
        numerator, denominator,
        out=np.where(numerator > 0, np.inf, 0.0),
        where=denominator > 0,
    )


def _score_definitions(y_pred, class_labels, background_indices):
    higgs_indices = [class_labels[name] for name in ("H_bb", "H_cc", "H_gg", "H_qq")]
    h_2pr = y_pred[:, higgs_indices].sum(axis=1, dtype=np.float64)
    h_bb = np.asarray(y_pred[:, class_labels["H_bb"]], dtype=np.float64)
    qcd = y_pred[:, background_indices].sum(axis=1, dtype=np.float64)
    return {
        "X_2pr_QCD": _ratio(h_2pr, qcd),
        "X_bb_QCD": _ratio(h_bb, h_bb + qcd),
        "X_bb": h_bb,
    }


def plot_ROC_score_definitions(y_pred, y_test, class_labels, plot_dir, *,
                               signal_classes=("H_bb",), background_classes=("QCD*",),
                               output_name="ROC_score_definitions"):
    background_names = [
        name for name in class_labels
        if any(fnmatchcase(name, pattern) for pattern in background_classes)
    ]
    background_indices = [class_labels[name] for name in background_names]
    signal_indices = [class_labels[name] for name in signal_classes]
    signal_mask = np.any(y_test[:, signal_indices] == 1, axis=1)
    background_mask = np.any(y_test[:, background_indices] == 1, axis=1)
    mask = signal_mask | background_mask
    y_true = signal_mask[mask].astype(int)
    scores = _score_definitions(y_pred[mask], class_labels, background_indices)
    labels = {
        "X_2pr_QCD": r"$X_{2pr,QCD}=(H_{bb}+H_{cc}+H_{gg}+H_{qq})/QCD$",
        "X_bb_QCD": r"$X_{bb,QCD}=H_{bb}/(H_{bb}+QCD)$",
        "X_bb": r"$X_{bb}=H_{bb}$",
    }
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=style.FIGURE_SIZE)
    curves = {}
    result = {
        "signal_classes": list(signal_classes), "background_classes": background_names,
        "n_signal": int(signal_mask.sum()), "n_background": int(background_mask.sum()),
        "n_excluded": int((~mask).sum()),
        "score_definitions": {"X_2pr_QCD": "(H_bb + H_cc + H_gg + H_qq) / QCD",
                              "X_bb_QCD": "H_bb / (H_bb + QCD)", "X_bb": "H_bb"},
        "QCD": "Sum of the selected background output probabilities",
        "zero_denominators": "Positive/0 is +inf; 0/0 is assigned 0 by convention",
        "roc_input": "Tie-preserving score ranks; identical ordering, including +inf scores",
        "auc": {},
    }
    for (name, y_score), linestyle in zip(scores.items(), ("-", "--", ":")):
        fpr, tpr, _ = roc_curve(y_true, rankdata(y_score, method="average"))
        roc_auc = float(auc(fpr, tpr))
        result["auc"][name] = roc_auc
        curves[f"{name}_fpr"] = fpr
        curves[f"{name}_tpr"] = tpr
        ax.plot(tpr, fpr, linestyle=linestyle, linewidth=style.LINEWIDTH,
                label=labels[name] + f" (AUC = {roc_auc:.4f})")
        print(f"{name}: AUC={roc_auc:.6f}", flush=True)
    hep.cms.label(ax=ax, llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT,
                  fontsize=style.CMSHEADER_SIZE)
    ax.set(xlabel="Signal efficiency", ylabel="QCD mistag rate", xlim=(0, 1), ylim=(1e-4, 1.05))
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=style.SMALL_SIZE - 3,
              title="Truth: " + " + ".join(signal_classes) + " vs QCD",
              title_fontsize=style.SMALL_SIZE - 3)
    fig.tight_layout()
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = plot_dir / f"{output_name}.{extension}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}", flush=True)
    plt.close(fig)
    with (plot_dir / f"{output_name}.json").open("w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    np.savez_compressed(plot_dir / f"{output_name}.npz", **curves)
    return result

