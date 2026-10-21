import json
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np


def _plot_scores_vs_pt(
    plot_data, sample_name, classes, *,
    pt_bins=(0, 50, 75, 100, 125, 150, 200, 300, 500, 800),
    score_bins=50, output_dir=None,
):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as effects
    from matplotlib.colors import LogNorm
    import mplhep as hep
    from tagger.plot import style

    sample = plot_data[sample_name]
    class_names = [
        name for name in sample["source_labels"]
        if any(fnmatchcase(name, pattern) for pattern in classes)
    ]
    class_ids = [sample["source_labels"][name] for name in class_names]
    selected = np.isin(sample["class_label"], class_ids)
    if sample_name == "signal":
        selected &= np.isfinite(sample["higgs_mass"])

    pt_bins = np.asarray(pt_bins, dtype=float)
    score_edges = np.linspace(0, 1, score_bins + 1)
    counts = np.zeros((len(pt_bins) - 1, score_bins), dtype=np.int64)
    jet_counts = []
    median_scores = []
    for index, (low, high) in enumerate(zip(pt_bins[:-1], pt_bins[1:])):
        in_bin = selected & (sample["pt"] > low) & (sample["pt"] <= high)
        scores = sample["tagger"][in_bin]
        counts[index], _ = np.histogram(scores, bins=score_edges)
        jet_counts.append(len(scores))
        median_scores.append(float(np.median(scores)) if len(scores) else None)
        print(f"{sample_name}: {low:g} < L1 pT <= {high:g}: "
              f"{len(scores)} jets, median score={median_scores[-1]}", flush=True)

    jet_counts = np.asarray(jet_counts)
    fractions = np.divide(
        counts, jet_counts[:, None],
        out=np.zeros_like(counts, dtype=float), where=jet_counts[:, None] > 0,
    )
    fig, (ax, count_ax) = plt.subplots(
        2, 1, figsize=style.FIGURE_SIZE, sharex=True,
        gridspec_kw={"height_ratios": [4, 1]}, layout="constrained",
    )
    hep.cms.label(
        ax=ax, llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT,
        fontsize=style.CMSHEADER_SIZE,
    )
    density = ax.pcolormesh(
        pt_bins, score_edges, np.ma.masked_equal(fractions.T, 0),
        cmap="magma", norm=LogNorm(vmin=1e-4, vmax=1), shading="flat",
    )
    fig.colorbar(density, ax=[ax, count_ax], extend="min",
                 label="Jet fraction per score bin (within each pT bin)")
    centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
    ax.plot(
        centers, [np.nan if value is None else value for value in median_scores],
        "o-", color="white", linewidth=style.LINEWIDTH, markersize=4,
        path_effects=[effects.withStroke(linewidth=style.LINEWIDTH + 1, foreground="black")],
        label="Median score",
    )
    title = "Signal: matched Higgs jets" if sample_name == "signal" else "Background: QCD jets"
    ax.text(0.02, 0.97, title, transform=ax.transAxes, va="top",
            fontsize=style.SMALL_SIZE, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.legend(loc="lower right", fontsize=style.SMALL_SIZE)
    ax.set_ylabel("Tagger score")
    ax.set_ylim(0, 1)
    count_ax.bar(centers, jet_counts, width=np.diff(pt_bins), align="center",
                 color="gray", edgecolor="black")
    count_ax.set_yscale("symlog", linthresh=1)
    count_ax.set_ylabel("Jets")
    count_ax.set_xlabel(r"L1 jet $p_T$ [GeV]")
    count_ax.set_xlim(pt_bins[0], pt_bins[-1])

    output_dir = Path(
        output_dir or Path(plot_data["output_directory"]) / "plots/nprong/scores_vs_pt"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sample_name}_scores_vs_pt"
    result = {
        "sample": sample_name, "source_dir": sample["directory"],
        "classes": class_names, "score_classes": list(plot_data["score_classes"]),
        "score_definition": "Existing tagger score: sum of requested model probabilities",
        "require_higgs_match": sample_name == "signal",
        "pt_field": "pt (jet_pt_phys)", "pt_bins": pt_bins.tolist(),
        "pt_bin_convention": "lower exclusive, upper inclusive",
        "score_edges": score_edges.tolist(), "counts": counts.tolist(),
        "jet_counts": jet_counts.tolist(), "median_scores": median_scores,
        "normalization": "Each populated pT bin sums to one over score bins",
        "max_chunks": plot_data["max_chunks"], "n_chunks": sample["n_chunks"],
    }
    with (output_dir / f"{filename}.json").open("w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{filename}.{suffix}", bbox_inches="tight")
    plt.close(fig)
    return result


def plot_signal_scores_vs_pt(
    plot_data, signal_classes=("H_bb", "H_cc", "H_qq", "H_gg"), **kwargs,
):
    return _plot_scores_vs_pt(plot_data, "signal", signal_classes, **kwargs)


def plot_background_scores_vs_pt(
    plot_data, background_classes=("QCD_*",), **kwargs,
):
    return _plot_scores_vs_pt(plot_data, "background", background_classes, **kwargs)

