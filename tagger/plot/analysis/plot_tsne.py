"""t-SNE of held-out jet inputs and the learned classification representation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def _sample_indices(labels, class_labels, max_per_class, seed):
    rng = np.random.default_rng(seed)
    counts = {name: np.count_nonzero(labels == index) for name, index in class_labels.items()}
    n_per_class = min(max_per_class, min(count for count in counts.values() if count))
    indices = np.concatenate([
        rng.choice(np.flatnonzero(labels == index), n_per_class, replace=False)
        for name, index in class_labels.items() if counts[name]
    ])
    rng.shuffle(indices)
    return indices


def _feature_model(model, embedding_layer):
    import keras

    network = model.jet_model
    if embedding_layer is None:
        classifier = keras.Model(network.inputs, network.get_layer("jet_id_output").input)
        features = classifier.layers[-1].input
    else:
        features = network.get_layer(embedding_layer).output
    return keras.Model(network.inputs, features)


def _project(features, perplexity, seed, standardize=False):
    features = np.asarray(features, dtype=np.float32).reshape(len(features), -1)
    if standardize:
        features = StandardScaler().fit_transform(features)
    if features.shape[1] > 50:
        features = PCA(
            n_components=min(50, len(features) - 1), random_state=seed,
        ).fit_transform(features)
    return TSNE(
        n_components=2, perplexity=perplexity, init="random",
        learning_rate="auto", random_state=seed,
    ).fit_transform(features)


def plot_tsne(model, X_test, y_test, *, class_labels=None,
              embedding_layer=None, max_per_class=1000, perplexity=30,
              seed=42, batch_size=1024, standardize_inputs=True, output_dir=None):
    class_labels = model.class_labels if class_labels is None else class_labels
    labels = np.argmax(y_test, axis=1) if y_test.ndim == 2 else np.asarray(y_test)
    indices = _sample_indices(labels, class_labels, max_per_class, seed)
    inputs = np.asarray(X_test[indices], dtype=np.float32)
    labels = labels[indices]
    perplexity = min(float(perplexity), len(indices) - 1)

    feature_model = _feature_model(model, embedding_layer)
    print(f"t-SNE: {len(indices)} jets; after-network tensor: {feature_model.output.name}", flush=True)
    latent = np.asarray(feature_model.predict(inputs, batch_size=batch_size, verbose=0))
    print("t-SNE: projecting constituent inputs", flush=True)
    before = _project(inputs, perplexity, seed, standardize=standardize_inputs)
    print("t-SNE: projecting learned features", flush=True)
    after = _project(latent, perplexity, seed)

    output_dir = Path(output_dir or Path(model.output_directory) / "plots/tsne")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "tsne_before_after.npz",
        indices=indices, labels=labels, before=before, after=after,
    )
    metadata = {
        "class_labels": {name: int(index) for name, index in class_labels.items()},
        "counts": {name: int(np.count_nonzero(labels == index)) for name, index in class_labels.items()},
        "seed": seed, "perplexity": perplexity, "max_per_class": max_per_class,
        "standardize_inputs": standardize_inputs,
        "embedding_layer": embedding_layer,
        "embedding_tensor": feature_model.output.name,
        "input_shape": list(inputs.shape[1:]), "latent_shape": list(latent.shape[1:]),
        "before": "Flattened constituent inputs; original constituent order and padding retained",
        "after": "Last hidden classification representation" if embedding_layer is None else embedding_layer,
        "preprocessing": "PCA without whitening to at most 50 components if feature dimension exceeds 50; "
                         "learned features are not standardized",
        "interpretation": "Separately fitted t-SNE maps; coordinates and inter-cluster distances are not comparable across panels",
    }
    with (output_dir / "tsne_before_after.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)
    _plot(before, after, labels, class_labels, metadata, output_dir)
    return {"indices": indices, "labels": labels, "before": before, "after": after}


def _plot(before, after, labels, class_labels, metadata, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = plt.get_cmap("tab10" if len(class_labels) <= 10 else "tab20")
    titles = (
        "Before network: " + ("standardized inputs" if metadata["standardize_inputs"] else "inputs"),
        "After network: " + (metadata["embedding_layer"] or "classification features"),
    )
    for ax, coordinates, title in zip(axes, (before, after), titles):
        for color_index, (name, index) in enumerate(sorted(class_labels.items(), key=lambda item: item[1])):
            selected = labels == index
            if selected.any():
                ax.scatter(
                    coordinates[selected, 0], coordinates[selected, 1],
                    s=7, alpha=0.55, color=colors(color_index), linewidths=0,
                    rasterized=True, label=f"{name} (n={np.count_nonzero(selected)})",
                )
        ax.set(title=title, xlabel="t-SNE 1", ylabel="t-SNE 2")
        ax.set_xticks([])
        ax.set_yticks([])
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=min(5, len(names)), markerscale=2)
    fig.suptitle("Same held-out jets; balanced classes; independent t-SNE maps", fontsize=12)
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    for extension in ("png", "pdf"):
        path = output_dir / f"tsne_before_after.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved {path}", flush=True)
    plt.close(fig)

