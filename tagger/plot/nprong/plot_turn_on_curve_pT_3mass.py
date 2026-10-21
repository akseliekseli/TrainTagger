import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import mplhep as hep
import numpy as np

from tagger.plot import style
from .plot_turn_on_curve_mass import _working_point, _score_working_point, _score_pass
from .plot_turn_on_curve_pT import (
    _background_event_values, extract_mass_metrics, _efficiency_errors, _working_point_label,
)


def plot_turn_on_curve_pT_3mass(plot_data, *, target_mh_masses=(30, 50, 90),
                                 window=2.0, fixed_score_cut=0.95, target_rate_hz=50000,
                                 n_bunches=2760, revolution_frequency_hz=11246,
                                 fixed_pt_cut=None, signal_classes=("H_bb", "H_cc", "H_qq", "H_gg"),
                                 pt_bins=None, output_path=None):
    """Three masses at one working point, with leading-jet rate calibration."""
    target_mh_masses = tuple(target_mh_masses)
    pt_bins = np.asarray(pt_bins if pt_bins is not None else np.concatenate([
        np.arange(0, 41, 40),
        np.arange(60, 301, 20),
        np.arange(400, 501, 100)
    ]), dtype=float)
    total_mb_rate = n_bunches * revolution_frequency_hz
    bkg_pt, bkg_score = _background_event_values(plot_data["background"])
    if fixed_pt_cut is None:
        working_point = _working_point(bkg_pt, bkg_score, fixed_score_cut, target_rate_hz, total_mb_rate)
    else:
        working_point = _score_working_point(bkg_pt, bkg_score, fixed_pt_cut, target_rate_hz, total_mb_rate)
    score_cut = working_point["score_cut"]
    fixed_pt_threshold = working_point["pt_threshold"]
    del bkg_pt, bkg_score
    print(_working_point_label(working_point)
          + f"; {working_point['background_pass']}/{working_point['background_events']} MinBias events", flush=True)

    plt.style.use(hep.style.CMS)
    output_path = Path(output_path or Path(plot_data["output_directory"]) / "plots/turn_on_pt/cms_turn_on_curve_pT_3mass")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
    fig1, ax1 = plt.subplots(figsize=style.FIGURE_SIZE)
    ax2 = ax1.twinx()
    colors = ["#5790fc", "#f89c20", "#e42536"]
    max_count_height = 0
    result = {"score_classes": plot_data["score_classes"], "max_chunks": plot_data["max_chunks"],
              "signal_dir": plot_data["signal"]["directory"], "background_dir": plot_data["background"]["directory"],
              "target_rate_hz": target_rate_hz, "total_mb_rate_hz": total_mb_rate,
              "background_selection": "leading jet before score cut; all saved background events including zero jets",
              "signal_classes": list(signal_classes),
              "signal_selection": "leading merged, matched jet per event in each PDG25 mass window",
              "calibration": "Thresholds and achieved rates evaluated on the same MinBias sample",
              "fixed_pt_cut": fixed_pt_cut,
              "window": window, "pt_bins": pt_bins.tolist(), "working_point": working_point, "masses": {}}
    for idx, mh_val in enumerate(target_mh_masses):
        color_choice = colors[idx % len(colors)]
        flat_den_l1_pt, flat_den_score, flat_den_gen_pt = extract_mass_metrics(plot_data["signal"], mh_val, window, signal_classes)
        passing_mask = _score_pass(flat_den_score, score_cut) & (flat_den_l1_pt > fixed_pt_threshold)
        flat_num_gen_pt = flat_den_gen_pt[passing_mask]
        num_m, _ = np.histogram(flat_num_gen_pt, bins=pt_bins)
        den_m, _ = np.histogram(flat_den_gen_pt, bins=pt_bins)
        eff_m, asymmetric_errors = _efficiency_errors(num_m, den_m)
        max_count_height = max(max_count_height, int(den_m.max()), int(num_m.max()))
        hep.histplot(den_m, pt_bins, ax=ax2, color=color_choice, lw=1.5,
                     linestyle="--", histtype="fill", alpha=0.2, zorder=1)
        hep.histplot(num_m, pt_bins, ax=ax2, color=color_choice, lw=2.2,
                     linestyle="-", alpha=0.4, zorder=1)
        valid = den_m > 0
        if valid.any():
            ax1.errorbar(pt_centers[valid], eff_m[valid], yerr=asymmetric_errors[:, valid],
                         xerr=np.diff(pt_bins)[valid]/2, fmt="o", color=color_choice, capsize=0, zorder=3,
                         markersize=getattr(style, "MARKERSIZE_SMALL", style.MARKERSIZE/2),
                         linewidth=getattr(style, "LINEWIDTH_SMALL", style.LINEWIDTH/2))
            ax1.plot(pt_centers, eff_m, color=color_choice, linestyle="-", alpha=0.3, lw=1.5, zorder=2)
        result["masses"][f"{mh_val:g}"] = {"num_counts": num_m.tolist(), "den_counts": den_m.tolist(),
                                           "matched_events": len(flat_den_gen_pt)}
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
    hep.cms.label(ax=ax1, llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, fontsize=style.CMSHEADER_SIZE)
    ax1.set_xlabel(r"Gen jet $p_T$ matched to leading L1 jet [GeV]", labelpad=10)
    ax1.set_ylabel(f"Level-1 Efficiency @ {target_rate_hz/1000:g} kHz [%]")
    ax1.set_xlim(pt_bins[0], pt_bins[-1])
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax2.set_ylim(0, max(1, max_count_height)*1.4)
    ax2.tick_params(axis="y", which="both", right=True, labelright=True)
    ax2.set_ylabel("Events per bin")
    ax2.grid(False)
    legend_elements = [
        Line2D([], [], color=colors[i % len(colors)], lw=4, label=rf"$m_X={mass:g}\pm{window:g}$ GeV")
        for i, mass in enumerate(target_mh_masses)
    ] + [
        Line2D([], [], marker="o", color="gray", linestyle="None", label="Efficiency"),
        Line2D([], [], color="gray", lw=2.2, alpha=0.4, label=r"Gen $p_T$ (triggered)"),
        Patch(facecolor="gray", edgecolor="none", alpha=0.2, label=r"Gen $p_T$ (all)"),
    ]
    title = _working_point_label(working_point)
    ax1.legend(handles=legend_elements, title=title, loc="center right", bbox_to_anchor=(0.98, 0.55),
               fontsize=style.SMALL_SIZE, title_fontsize=style.SMALL_SIZE)
    fig1.tight_layout()
    for suffix in ("png", "pdf"):
        fig1.savefig(f"{output_path}.{suffix}")
    plt.close(fig1)
    with Path(f"{output_path}.json").open("w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


plot_turn_on_curve_pT_3masses = plot_turn_on_curve_pT_3mass

