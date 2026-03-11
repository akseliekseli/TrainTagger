# flake8: noqa
import collections
import os

import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep

# Third parties
import pandas
import tensorflow as tf
from matplotlib.pyplot import cm
from scipy.stats import norm

setattr(collections, "MutableMapping", collections.abc.MutableMapping)
import histbook
import numpy as np
import shap
from sklearn.metrics import auc, roc_curve

from tagger.data.tools import load_data, to_ML
from tagger.plot import style

from .common import PT_BINS, plot_histo

matplotlib.use("Agg")

plt.rcParams.update({"figure.max_open_warning": 0})

# some custom imports for efficiency plots
np.bool = np.bool_


style.set_style()


def plot_tsne(model, X_test, y_test):
    from sklearn.manifold import TSNE
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D

    # invert label mapping: index -> name
    idx_to_label = {v: k for k, v in model.class_labels.items()}
    n_classes = len(idx_to_label)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    X_test_reduced = X_test.mean(axis=1)
    X_embedded = tsne.fit_transform(X_test_reduced)

    y_test_idx = y_test.argmax(axis=1)
    classes = np.unique(y_test_idx)

    base_cmap = cm.get_cmap("Set1", n_classes)
    cmap = ListedColormap(base_cmap(np.arange(n_classes)))
    norm = BoundaryNorm(np.arange(n_classes + 1), n_classes)

    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=y_test_idx,
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        s=4,
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=cmap(norm(cls)),
            label=idx_to_label[cls],
        )
        for cls in classes
    ]
    plt.legend(handles=handles, title="Classes", loc="upper left")
    plt.title("t-SNE visualization")
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tsne_plot_mean.png")


def inspect_asra_data(model):
    """
    Inspecting the data Asra used for binary 2-prong tagger.
    This is for debugging.
    """

    plot_dir = os.path.join(model.output_directory, "plots/training")

    # Load the testing data
    X_test = np.load(
        f"/home/akseli/l1-jet-id/data/16const/processed/proc_test_16const.npy"
    )
    y_test = np.load(
        f"/home/akseli/l1-jet-id/data/16const/processed/proc_labels_test_16const.npy"
    )

    model_outputs = model.jet_model.predict(X_test)
    print("Classes in y_test:", np.unique(y_test))
    print("All classes:", model.class_labels)

    # Get classification outputs
    y_pred = model_outputs[0]
    pt_ratio = model_outputs[1][:, 0]

    print(f"ASRA {X_test.shape}")

    return
