import json
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from tagger.plot import style
from .plot_turn_on_curve_mass import (
    _leading_indices, _working_point, _score_working_point, _score_pass, _interval,
)


def _background_event_values(background):
    leading_jet_idx = _leading_indices(background)
    leading_pt = np.full(background["n_events"], -1.0)
    leading_score = np.full(background["n_events"], -1.0)
    event_index = background["event_index"][leading_jet_idx]
    leading_pt[event_index] = background["pt"][leading_jet_idx]
    leading_score[event_index] = background["tagger"][leading_jet_idx]
    return leading_pt, leading_score


def extract_mass_metrics(signal, study_mass, window, signal_classes):
    """Return one matched L1 leader per event, and the pT of ITS matched GenJet."""
    signal_indices = [signal["source_labels"][name] for name in signal_classes]
    merged = np.isin(signal["class_label"], signal_indices)
    current_mass_mask = merged & (np.abs(signal["higgs_mass"] - study_mass) < window)
    global_leading_idx = _leading_indices(signal, current_mass_mask)
    flat_den_l1_pt = signal["pt"][global_leading_idx]
    flat_den_gen_pt = signal["gen_pt"][global_leading_idx]
    flat_den_score = signal["tagger"][global_leading_idx]
    return flat_den_l1_pt, flat_den_score, flat_den_gen_pt


def _efficiency_errors(num_counts, den_counts):
    intervals = np.asarray([
        _interval(int(num), int(den)) for num, den in zip(num_counts, den_counts)
    ], dtype=float) * 100
    eff, cp_low, cp_high = intervals.T
    asymmetric_errors = np.vstack([eff-cp_low, cp_high-eff])
    return eff, asymmetric_errors


def _working_point_label(point):
    score_cut = point["score_cut"]
    score_label = "No score cut" if score_cut is None else f"Score > {score_cut:.3f}"
    return (rf"$p_T^{{L1}}>{point['pt_threshold']:.2f}$ GeV, " + score_label
            + f" ({point['achieved_rate_hz']/1000:.2f} kHz)")


def plot_turn_on_curve_pT(plot_data, *, study_masses=(30, 50), window=2.0,
                          fixed_pt_cuts=(90, 111), fixed_score_cuts=(None,),
                          signal_classes=("H_bb", "H_cc", "H_qq", "H_gg"),
                          target_rate_hz=50000, n_bunches=2760,
                          revolution_frequency_hz=11246, pt_bins=None, output_path=None):
    """Fixed-pT / fixed-score working points, calibrated with leading MinBias jets."""
    study_masses = tuple(study_masses)
    pt_bins = np.asarray(pt_bins if pt_bins is not None else np.concatenate([
        np.arange(0, 41, 40),
        np.arange(60, 301, 20),
        np.arange(400, 501, 100)
    ]), dtype=float)
    total_mb_rate = n_bunches * revolution_frequency_hz
    fixed_pt_cuts, fixed_score_cuts = tuple(fixed_pt_cuts), tuple(fixed_score_cuts)
    plt.style.use(hep.style.CMS)
    output_path = Path(output_path or Path(plot_data["output_directory"]) / "plots/turn_on_pt/cms_turn_on_curve_pT")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    leading_pt, leading_score = _background_event_values(plot_data["background"])
    working_points = []
    for pt_cut in fixed_pt_cuts:
        working_points.append(_score_working_point(leading_pt, leading_score, pt_cut, target_rate_hz, total_mb_rate))
    for score_cut in fixed_score_cuts:
        working_points.append(_working_point(leading_pt, leading_score, score_cut, target_rate_hz, total_mb_rate))
    del leading_pt, leading_score
    for point in working_points:
        print(_working_point_label(point)
              + f"; {point['background_pass']}/{point['background_events']} MinBias events", flush=True)

    pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
    result = {"score_classes": plot_data["score_classes"], "max_chunks": plot_data["max_chunks"],
              "signal_dir": plot_data["signal"]["directory"], "background_dir": plot_data["background"]["directory"],
              "target_rate_hz": target_rate_hz, "total_mb_rate_hz": total_mb_rate,
              "background_selection": "leading jet before score cut; all saved background events including zero jets",
              "signal_classes": list(signal_classes),
              "signal_selection": "leading merged, matched jet per event in each PDG25 mass window",
              "calibration": "Thresholds and achieved rates evaluated on the same MinBias sample",
              "pt_bins": pt_bins.tolist(), "window": window, "working_points": working_points, "masses": {}}
    for study_mass in study_masses:
        flat_den_l1_pt, flat_den_score, flat_den_gen_pt = extract_mass_metrics(plot_data["signal"], study_mass, window, signal_classes)
        den_counts, _ = np.histogram(flat_den_gen_pt, bins=pt_bins)
        fig1, ax1 = plt.subplots(figsize=style.FIGURE_SIZE)
        max_den = max(1, int(np.max(den_counts)))
        normalized_gen_shape = den_counts / max_den * 100
        ax1.bar(pt_centers, normalized_gen_shape, width=np.diff(pt_bins),
                color="gainsboro", alpha=0.6, edgecolor="none", zorder=1,
                label=rf"Gen jet $p_T$ shape (peak=100), $m_X={study_mass:g}\pm{window:g}$ GeV")
        mass_result = {"den_counts": den_counts.tolist(), "matched_events": len(flat_den_gen_pt), "curves": []}
        for point in working_points:
            pt_threshold, score_cut = point["pt_threshold"], point["score_cut"]
            passing_mask = _score_pass(flat_den_score, score_cut) & (flat_den_l1_pt > pt_threshold)
            flat_num_gen_pt = flat_den_gen_pt[passing_mask]
            num_counts, _ = np.histogram(flat_num_gen_pt, bins=pt_bins)
            eff, asymmetric_errors = _efficiency_errors(num_counts, den_counts)
            valid = den_counts > 0
            total_eff = float(num_counts.sum() / den_counts.sum() * 100) if den_counts.sum() else None
            label_string = _working_point_label(point)
            if valid.any():
                eb = ax1.errorbar(pt_centers[valid], eff[valid], yerr=asymmetric_errors[:, valid],
                    xerr=np.diff(pt_bins)[valid]/2, fmt="o", capsize=0, zorder=3, label=label_string,
                    markersize=getattr(style, "MARKERSIZE_SMALL", style.MARKERSIZE/2),
                    linewidth=getattr(style, "LINEWIDTH_SMALL", style.LINEWIDTH/2))
                ax1.plot(pt_centers, eff, color=eb[0].get_color(), linestyle="--", alpha=0.5, zorder=2)
            mass_result["curves"].append({"pt_threshold": pt_threshold, "score_cut": score_cut,
                                          "num_counts": num_counts.tolist(), "total_efficiency_percent_in_bins": total_eff})
            print(f"M_X={study_mass:g}: {label_string}; efficiency in plotted bins: {total_eff}", flush=True)
        hep.cms.label(ax=ax1, llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, fontsize=style.CMSHEADER_SIZE)
        ax1.axhline(100, color="gray", linestyle=":", alpha=0.5, zorder=0)
        ax1.set_xlabel(r"Gen jet $p_T$ matched to leading L1 jet [GeV]")
        ax1.set_ylabel(f"Level-1 Efficiency @ {target_rate_hz/1000:g} kHz [%]")
        ax1.set_xlim(pt_bins[0], pt_bins[-1])
        ax1.set_ylim(0, 105)
        ax1.grid(True, linestyle="--", alpha=0.3, color="gray")
        ax1.legend(loc="center right", frameon=False, fontsize=style.SMALL_SIZE)
        fig1.tight_layout()
        mass_output_path = f"{output_path}_MH{study_mass:g}"
        for suffix in ("png", "pdf"):
            fig1.savefig(f"{mass_output_path}.{suffix}")
        plt.close(fig1)
        result["masses"][f"{study_mass:g}"] = mass_result
    with Path(f"{output_path}.json").open("w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result

