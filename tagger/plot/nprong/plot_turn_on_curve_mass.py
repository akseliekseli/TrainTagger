"""Mass turn-on curves from the shared compact plotting data."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from .plot_data import SCORE_CLASSES


def _leading_indices(sample, selection=None):
    """First highest-pT selected jet per event, before applying any score cut."""
    selected = (np.arange(len(sample["pt"])) if selection is None
                else np.flatnonzero(selection))
    if not len(selected):
        return selected
    order = np.lexsort((selected, -sample["pt"][selected], sample["event_index"][selected]))
    selected = selected[order]
    event_indices = sample["event_index"][selected]
    first_in_event = np.r_[True, event_indices[1:] != event_indices[:-1]]
    return selected[first_in_event]


def _score_pass(scores, cut):
    # None is a genuinely score-free baseline; numeric 0 retains the reference's >0.
    return np.ones(scores.shape, dtype=bool) if cut is None else scores > cut


def _working_point(pt, scores, score_cut, target_rate_hz, total_rate_hz):
    """Reference order statistic with explicit small-sample/under-budget handling."""
    n = len(pt)  # Includes zero-jet events, whose stored pT and score are -1.
    if not n:
        raise ValueError("No MinBias events")
    eligible = pt[(pt > 0) & _score_pass(scores, score_cut)]
    budget = int(target_rate_hz / total_rate_hz * n)
    if len(eligible) <= budget:
        threshold = 0.0  # Already below target; do not produce a negative pT cut.
    elif budget == 0:
        threshold = float(np.max(eligible))  # Strict > gives zero passing events.
    else:
        # Same normal-case selection as util.find_pt_threshold: sorted_pts[-budget].
        threshold = float(np.partition(eligible, len(eligible) - budget)[-budget])
    passed = int(np.sum((pt > threshold) & _score_pass(scores, score_cut)))
    return {
        "score_cut": score_cut, "pt_threshold": threshold,
        "background_pass": passed, "background_events": n,
        "achieved_rate_hz": passed / n * total_rate_hz,
        "rate_step_hz": total_rate_hz / n,
    }


def _interval(k, n, confidence=0.6827):
    if n == 0:
        return None, None, None
    alpha = 1 - confidence
    low = beta.ppf(alpha / 2, k, n-k+1) if k else 0.0
    high = beta.isf(alpha / 2, k+1, n-k) if k < n else 1.0
    return k / n, float(low), float(high)


def plot_turn_on_curve_mass(plot_data, *,
                            mass_centers=(20, 30, 40, 50, 60, 70, 80, 90),
                            mass_window=2.0, score_cuts=(0.0, 0.85, 0.95, 0.96),
                            target_rate_hz=50000, n_bunches=2760,
                            revolution_frequency_hz=11246, output_dir=None):
    """No ROOT reads or inference. Score is the sum of the four Higgs outputs."""
    mass_centers, score_cuts = tuple(mass_centers), tuple(score_cuts)
    if tuple(plot_data["score_classes"]) != SCORE_CLASSES:
        raise ValueError("Unexpected score columns")
    if not mass_centers or len(set(mass_centers)) != len(mass_centers):
        raise ValueError("Specify unique mass centers")
    if any(not np.isfinite(m) or m < 0 for m in mass_centers) or not np.isfinite(mass_window) or mass_window <= 0:
        raise ValueError("Mass centers must be finite/nonnegative and window finite/positive")
    if not score_cuts or any(c is not None and (not np.isfinite(c) or not 0 <= c <= 1) for c in score_cuts):
        raise ValueError("Score cuts must be None or finite numbers between 0 and 1")
    total_rate = n_bunches * revolution_frequency_hz
    if not np.isfinite(total_rate) or total_rate <= 0 or not 0 < target_rate_hz < total_rate:
        raise ValueError("Invalid rate normalization or target rate")

    signal, background = plot_data["signal"], plot_data["background"]
    print("Turn-on score: " + " + ".join(SCORE_CLASSES), flush=True)
    print("Signal: leading matched jet per mass window, all decay labels. "
          "MinBias: leading jet, all saved events including zero jets.", flush=True)

    # Do not use the ROC's QCD-label filter here.
    leaders = _leading_indices(background)
    mb_pt = np.full(background["n_events"], -1.0)
    mb_score = np.full(background["n_events"], -1.0)
    event_indices = background["event_index"][leaders]
    mb_pt[event_indices] = background["pt"][leaders]
    mb_score[event_indices] = background["tagger"][leaders]
    working_points = [
        _working_point(mb_pt, mb_score, cut, target_rate_hz, total_rate)
        for cut in score_cuts
    ]
    del leaders, event_indices, mb_pt, mb_score

    for wp in working_points:
        print(f"Score cut {wp['score_cut']}: pT > {wp['pt_threshold']:.6g} GeV; "
              f"achieved rate {wp['achieved_rate_hz']/1000:.4g} kHz "
              f"({wp['background_pass']}/{wp['background_events']} events)", flush=True)
        if wp["background_pass"] < 10:
            print("  WARNING: fewer than 10 passing MinBias events; rate estimate has limited statistics", flush=True)

    numerator = np.zeros((len(working_points), len(mass_centers)), dtype=np.int64)
    denominator = np.zeros(len(mass_centers), dtype=np.int64)
    for mass_index, mass in enumerate(mass_centers):
        # Strict boundaries are unchanged from the original turn-on function.
        matched = np.abs(signal["higgs_mass"] - mass) < mass_window
        leaders = _leading_indices(signal, matched)
        denominator[mass_index] = len(leaders)
        pt, score = signal["pt"][leaders], signal["tagger"][leaders]
        for cut_index, wp in enumerate(working_points):
            numerator[cut_index, mass_index] = np.count_nonzero(
                (pt > wp["pt_threshold"]) & _score_pass(score, wp["score_cut"])
            )
    intervals = [[_interval(int(k), int(n)) for k, n in zip(row, denominator)] for row in numerator]
    result = {
        "signal_dir": signal["directory"], "background_dir": background["directory"],
        "score_classes": SCORE_CLASSES, "mass_centers": mass_centers, "mass_window": mass_window,
        "target_rate_hz": target_rate_hz, "total_mb_rate_hz": total_rate,
        "signal_events_read": signal["n_events"], "signal_denominator": denominator.tolist(),
        "signal_pass": numerator.tolist(), "working_points": working_points,
        "efficiency_and_6827pct_interval": intervals,
        "max_chunks": plot_data["max_chunks"],
        "signal_chunks_read": signal["n_chunks"], "background_chunks_read": background["n_chunks"],
        "definition": "conditional on matched signal jet; leading jet before strict score/pT cuts; PDG25 mass; all decay labels",
    }
    output_dir = Path(output_dir or Path(plot_data["output_directory"]) / "plots/turn_on_mass")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "turn_on_mass.json").open("w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    _plot(result, output_dir)
    return result


def _plot(result, output_dir):
    import matplotlib.pyplot as plt
    import mplhep as hep
    from tagger.plot import style

    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=style.FIGURE_SIZE)
    masses = np.asarray(result["mass_centers"])
    for wp, bounds in zip(result["working_points"], result["efficiency_and_6827pct_interval"]):
        values = np.asarray([[np.nan if v is None else 100*v for v in row] for row in bounds])
        valid = np.isfinite(values[:, 0])
        for m in masses[~valid]:
            print(f"No matched signal events for m_H={m:g} GeV; omitting point", flush=True)
        if not valid.any():
            continue
        efficiency, low, high = values[valid].T
        cut = wp["score_cut"]
        title = "No score cut" if cut is None else f"Score > {cut:g}"
        label = (title + rf", $p_T^{{L1}}>{wp['pt_threshold']:.2f}$ GeV"
                 + f" ({wp['achieved_rate_hz']/1000:.2f} kHz)")
        baseline = cut is None or cut == 0
        ax.errorbar(masses[valid], efficiency, yerr=[efficiency-low, high-efficiency],
                    fmt="o--" if baseline else "o-", fillstyle="none" if baseline else "full",
                    markersize=style.MARKERSIZE, linewidth=style.LINEWIDTH, label=label)
    hep.cms.label(ax=ax, llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT,
                  fontsize=style.CMSHEADER_SIZE)
    ax.set(xlabel=r"Matched Higgs mass $m_H$ [GeV]",
           ylabel=f"Conditional L1 efficiency [%] (target {result['target_rate_hz']/1000:g} kHz)",
           ylim=(0, 105), xlim=(float(masses.min())-5, float(masses.max())+5))
    ax.axhline(100, color="gray", linestyle=":", alpha=0.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=style.SMALL_SIZE)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = output_dir / f"turn_on_mass.{ext}"
        fig.savefig(path)
        print(f"Saved {path}", flush=True)
    plt.close(fig)

