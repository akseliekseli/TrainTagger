import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
import copy

# Third parties
import numpy as np
import yaml
import tensorflow as tf

num_threads = 32
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["TF_NUM_INTRAOP_THREADS"] = "32"
os.environ["TF_NUM_INTEROP_THREADS"] = "32"
os.environ["NUMEXPR_NUM_THREADS"] = "32"

tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)
# Enable GPU usage and avoid TF pre-allocating all memory
gpus = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(gpus[2], "GPU")
import optuna

# Import from other modules
from tagger.data.tools import load_data, to_ML
from tagger.model.common import fromFolder, fromYaml
from tagger.plot.basic import basic


if gpus:
    try:
        for gpu in tf.config.get_visible_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print("Error setting GPU memory growth:", e)
else:
    print("No GPUs detected, running on CPU.")
# Silence some TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

"""
def save_test_data(out_dir, X_test, y_test, truth_pt_test, reco_pt_test):
    use_jets = True
    os.makedirs(os.path.join(out_dir, 'testing_data'), exist_ok=True)
    if use_jets:
        np.save(os.path.join(
            out_dir, "testing_data/X_test_constits.npy"), X_test[0])
        np.save(os.path.join(
            out_dir, "testing_data/X_test_jets.npy"), X_test[1])
    else:
        np.save(os.path.join(out_dir, "testing_data/X_test_constits.npy"), X_test)
    #np.save(os.path.join(out_dir, "testing_data/X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "testing_data/y_test.npy"), y_test)
    np.save(os.path.join(out_dir, "testing_data/truth_pt_test.npy"), truth_pt_test)
    np.save(os.path.join(out_dir, "testing_data/reco_pt_test.npy"), reco_pt_test)

    print(f"Test data saved to {out_dir}")
"""


def save_test_data(
    out_dir, X_test, y_test, truth_pt_test, reco_pt_test, mass_target_test
):
    os.makedirs(os.path.join(out_dir, "testing_data"), exist_ok=True)
    # X_test_constits, X_test_jets = X_test
    np.save(os.path.join(out_dir, "testing_data/X_test.npy"), X_test[0])
    np.save(os.path.join(out_dir, "testing_data/y_test.npy"), y_test)
    np.save(os.path.join(out_dir, "testing_data/truth_pt_test.npy"), truth_pt_test)
    np.save(os.path.join(out_dir, "testing_data/reco_pt_test.npy"), reco_pt_test)
    np.save(
        os.path.join(out_dir, "testing_data/mass_target_test.npy"), mass_target_test
    )

    print(f"Test data saved to {out_dir}")


def train_weights(
    y_train,
    reco_pt_train,
    mass_target_train,
    class_labels,
    weightingMethod,
    debug,
    reference_class=None,
):
    if weightingMethod not in ["none", "ptref", "onlyclass", "mass", "ptref_and_mass"]:
        raise ValueError("Unknown weightingMethod.")

    num_samples = y_train.shape[0]
    sample_weights = np.ones(num_samples)

    pt_bins = np.array(
        [
            15,
            17,
            19,
            22,
            25,
            30,
            35,
            40,
            45,
            50,
            60,
            76,
            97,
            122,
            154,
            200,
            250,
            320,
            400,
            500,
            650,
            800,
            1000,
            np.inf,
        ]
    )
    """
    pt_bins = np.array(
        [
            200,
            250,
            320,
            400,
            500,
            650,
            800,
            1000,
            np.inf,
        ]
    )
    """
    min_m, max_m = mass_target_train.min(), mass_target_train.max()
    mass_bins = np.linspace(min_m, max_m, 51)

    if weightingMethod == "onlyclass":
        class_counts = {idx: np.sum(y_train[:, idx]) for idx in class_labels.values()}
        non_zero = np.array([c for c in class_counts.values() if c > 0])
        target = np.median(non_zero) if len(non_zero) > 0 else 1.0
        for idx, count in class_counts.items():
            w = min(target / count, 100.0) if count > 0 else 0.0
            sample_weights[y_train[:, idx] == 1] = w

    # ── Step 1: flat pT reweighting — undersample heavy bins ─────────────────
    if weightingMethod in ["ptref", "ptref_and_mass"]:
        for label, idx in class_labels.items():
            class_mask = y_train[:, idx] == 1
            class_pt = reco_pt_train[class_mask]
            sample_indices = np.where(class_mask)[0]

            hist, _ = np.histogram(class_pt, bins=pt_bins)

            # Normalize by bin width to get density-like counts
            bin_widths = np.diff(pt_bins[:-1])  # exclude inf bin
            bin_widths = np.append(
                bin_widths, bin_widths[-1]
            )  # repeat last width for inf bin
            hist_density = np.where(bin_widths > 0, hist / bin_widths, hist)

            nonempty_density = hist_density[hist > 0]
            if len(nonempty_density) == 0:
                continue

            # Use higher percentile for pT since bins vary wildly in width
            target_density = np.percentile(nonempty_density, 40)

            pt_weights_per_bin = np.zeros(len(pt_bins) - 1)
            for bin_idx in range(len(pt_bins) - 1):
                if hist[bin_idx] > 0:
                    pt_weights_per_bin[bin_idx] = min(
                        target_density / hist_density[bin_idx], 1.0
                    )

            bin_indices = np.digitize(class_pt, pt_bins) - 1
            bin_indices = np.clip(bin_indices, 0, len(pt_bins) - 2)
            sample_weights[sample_indices] = pt_weights_per_bin[bin_indices]

            if debug:
                print(f"DEBUG - pT weights for '{label}':")
                print(f"  hist:         {hist}")
                print(f"  hist_density: {np.round(hist_density, 1)}")
                print(f"  weights:      {np.round(pt_weights_per_bin, 4)}")
    # ── Step 2: flat mass reweighting per pT bin (multiplicative) ─────────────
    if weightingMethod in ["mass", "ptref_and_mass"]:
        for label, idx in class_labels.items():
            class_mask = y_train[:, idx] == 1
            class_mass = mass_target_train[class_mask]
            class_pt = reco_pt_train[class_mask]
            sample_indices = np.where(class_mask)[0]

            mass_weights = np.ones(class_mask.sum())

            pt_bin_indices = np.digitize(class_pt, pt_bins) - 1
            pt_bin_indices = np.clip(pt_bin_indices, 0, len(pt_bins) - 2)

            for pt_bin_idx in range(len(pt_bins) - 1):
                in_pt_bin = pt_bin_indices == pt_bin_idx
                if not in_pt_bin.any():
                    continue

                bin_masses = class_mass[in_pt_bin]
                hist_m, _ = np.histogram(bin_masses, bins=mass_bins)

                nonempty = hist_m[hist_m > 0]
                if len(nonempty) == 0:
                    continue
                target_count = np.percentile(nonempty, 35)

                mass_weights_per_bin = np.zeros(len(mass_bins) - 1)
                for bin_idx in range(len(mass_bins) - 1):
                    if hist_m[bin_idx] > 0:
                        mass_weights_per_bin[bin_idx] = min(
                            target_count / hist_m[bin_idx], 1.0
                        )

                mass_bin_idx = np.digitize(bin_masses, mass_bins) - 1
                mass_bin_idx = np.clip(mass_bin_idx, 0, len(mass_bins) - 2)
                mass_weights[in_pt_bin] = mass_weights_per_bin[mass_bin_idx]

            sample_weights[sample_indices] *= mass_weights

            if debug:
                print(
                    f"DEBUG - mass weights for '{label}': "
                    f"mean={mass_weights.mean():.3f} "
                    f"min={mass_weights.min():.3f} "
                    f"max={mass_weights.max():.3f}"
                )

    # ── Normalize ─────────────────────────────────────────────────────────────
    sample_weights = np.nan_to_num(sample_weights, nan=0.0, posinf=0.0, neginf=0.0)
    mean_w = np.mean(sample_weights)
    if mean_w == 0 or np.isnan(mean_w):
        print("WARNING: sample_weights mean is 0/nan, using uniform weights")
        sample_weights = np.ones(num_samples)
    else:
        sample_weights = sample_weights / mean_w

    if debug:
        for label, idx in class_labels.items():
            mask = y_train[:, idx] == 1
            print(
                f"DEBUG - Mean weight for '{label}': {sample_weights[mask].mean():.4f}"
            )

    if weightingMethod == "none":
        return None
    return sample_weights


def plot_deta_dphi(X_train, y_train, feat_x=1, feat_y=2):
    """
    Plot all set elements for one sample from each of two labels.
    Parameters:
        X_train: np.ndarray of shape (n_elements, n_set, n_features)
        y_train: np.ndarray of shape (n_elements, n_classes) (one-hot encoded)
        feat_x, feat_y: feature indices to plot on X and Y axes
    """
    # Convert one-hot labels to class indices
    y_labels = np.argmax(y_train, axis=1)

    # Get unique class indices
    unique_labels = np.unique(y_labels)
    if len(unique_labels) < 2:
        raise ValueError("Need at least two distinct classes to plot.")

    # Pick one example (element) from the first two classes
    idx1 = np.where(y_labels == unique_labels[0])[0][0]
    idx2 = np.where(y_labels == unique_labels[1])[0][0]

    # Extract the corresponding sets
    set1 = X_train[idx1]  # shape (n_set, n_features)
    set2 = X_train[idx2]
    print(f"set1: {set1.shape}")
    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(set1[:, feat_x], set1[:, feat_y], label=f"H_bb", alpha=0.7)
    plt.scatter(set2[:, feat_x], set2[:, feat_y], label=f"QCD", alpha=0.7)
    plt.xlabel(f"$\Delta \eta$")
    plt.ylabel(f"$\Delta \phi$")
    plt.legend()
    plt.title("H_bb and QCD")
    plt.grid(True)

    # Save and show
    plt.savefig("deta_dphi_scatter.png", dpi=200, bbox_inches="tight")
    # plt.show()

    plt.clf()


def plot_weighting_distributions(
    y_train, reco_pt_train, mass_target_train, sample_weights, class_labels, out_dir
):
    """Plot pT and mass distributions before and after weighting for all classes."""
    import matplotlib.pyplot as plt
    import matplotlib

    os.makedirs(out_dir, exist_ok=True)
    pt_bins = np.array(
        [
            15,
            17,
            19,
            22,
            25,
            30,
            35,
            40,
            45,
            50,
            60,
            76,
            97,
            122,
            154,
            200,
            250,
            320,
            400,
            500,
            650,
            800,
            1000,
        ]
    )
    min_m, max_m = mass_target_train.min(), mass_target_train.max()
    mass_bins = np.linspace(min_m, max_m, 40)
    try:
        cmap = matplotlib.colormaps["Set1"].resampled(len(class_labels))
    except AttributeError:
        cmap = matplotlib.cm.get_cmap("Set1", len(class_labels))
    colors = [cmap(i) for i in range(len(class_labels))]

    def _make_plot(values, bins, xlabel, title, save_name):
        fig, ax = plt.subplots(figsize=(9, 6))
        # fig.suptitle(title, fontsize=14)
        for (label, idx), color in zip(class_labels.items(), colors):
            class_mask = y_train[:, idx] == 1
            class_vals = values[class_mask]
            class_w = (
                sample_weights[class_mask]
                if sample_weights is not None
                else np.ones(class_mask.sum())
            )
            # Unweighted — solid line
            ax.hist(
                class_vals,
                bins=bins,
                histtype="step",
                density=True,
                label=f"{label} (unweighted)",
                color=color,
                linewidth=2,
                linestyle="-",
            )
            # Weighted — dashed line
            ax.hist(
                class_vals,
                bins=bins,
                histtype="step",
                density=True,
                weights=class_w,
                label=f"{label} (weighted)",
                color=color,
                linewidth=2,
                linestyle="--",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(out_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Distributions saved to {save_path}")

    _make_plot(
        reco_pt_train,
        pt_bins,
        "Reco pT [GeV]",
        "pT — Weighted vs Unweighted",
        "pt_weighting_distributions.png",
    )
    _make_plot(
        mass_target_train,
        mass_bins,
        "Jet Mass [GeV]",
        "Mass — Weighted vs Unweighted",
        "mass_weighting_distributions.png",
    )


def normalize(X_train):
    """
    Standardizes the features of jet data (shape: samples, constituents, features),
    skipping features (Axis 2) where ALL values are 0 or 1, regardless of dtype.

    Args:
        X_train (np.ndarray): The input data array with shape (N_samples, N_constituents, N_features).

    Returns:
        np.ndarray: The standardized array.
    """

    print(X_train.shape)
    N_features = X_train.shape[-1]
    X_normalized = X_train.copy()
    non_boolean_indices = []

    # --- 1. Identify Non-Binary/Numeric Features ---
    for i in range(N_features):
        feature_channel = X_train[..., i]

        # This check is what you requested:
        # Check if ALL values are either 0 or 1.
        is_binary = np.all(np.logical_or(feature_channel == 0, feature_channel == 1))

        if not is_binary:
            non_boolean_indices.append(i)

    print(f"Features to normalize (indices): {non_boolean_indices}")

    if not non_boolean_indices:
        print("No numeric features found. Returning original data.")
        return X_train

    # --- 2. Calculate Mean and Standard Deviation (Only on Selected Features) ---
    X_to_normalize = X_train[..., non_boolean_indices]

    # Calculate mean and std deviation across the 'sample' (0) and 'constituents' (1) axes
    mean = np.mean(X_to_normalize, axis=(0, 1), keepdims=True)
    std = np.std(X_to_normalize, axis=(0, 1), keepdims=True)

    # Prevent division by zero
    std[std == 0] = 1.0

    # --- 3. Apply Standardization and Return ---
    X_normalized_subset = (X_to_normalize - mean) / std

    # Place the standardized values back into the result array
    X_normalized[..., non_boolean_indices] = X_normalized_subset

    return X_normalized


def undersample_per_class(y, n_per_class=5, seed=42):
    rng = np.random.default_rng(seed)
    y_labels = np.argmax(y, axis=1)
    unique_classes = np.unique(y_labels)
    selected_indices = []
    for c in unique_classes:
        class_indices = np.where(y_labels == c)[0]
        chosen = rng.choice(
            class_indices, size=min(n_per_class, len(class_indices)), replace=False
        )
        selected_indices.extend(chosen)
    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)  # <-- critical fix
    return selected_indices


def objective(trial, data, yaml_path, yaml_dict, out_dir):
    (
        X_train,
        y_train,
        pt_target_train,
        sample_weights,
        input_vars,
        extra_vars,
        class_labels,
        input_shape,
        output_shape,
    ) = data

    config = copy.deepcopy(yaml_dict)

    # --- update hyperparameters dynamically ---
    config["training_config"]["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-5, 1e-2, log=True
    )
    config["training_config"]["batch_size"] = trial.suggest_categorical(
        "batch_size", [1024, 2048, 4096, 8192]
    )
    config["training_config"]["epochs"] = trial.suggest_int(
        "epochs", 100, 200, step=100
    )
    config["model_config"]["conv1d_layers"][0] = trial.suggest_int(
        "conv1d_1", 32, 128, step=32
    )
    config["model_config"]["conv1d_layers"][1] = trial.suggest_int(
        "conv1d_2", 32, 128, step=32
    )
    config["model_config"]["conv1d_layers"][2] = trial.suggest_int(
        "conv1d_3", 32, 128, step=32
    )

    # --- Update patience settings ---
    config["training_config"]["EarlyStopping_patience"] = trial.suggest_int(
        "EarlyStopping_patience", 5, 10, step=1
    )
    config["training_config"]["ReduceLROnPlateau_patience"] = trial.suggest_int(
        "ReduceLROnPlateau_patience", 5, 30, step=5
    )
    sample_weights = sample_weights / np.mean(sample_weights)
    model = fromYaml(yaml_path, config, out_dir)
    model.set_labels(
        input_vars,
        extra_vars,
        class_labels,
    )
    model.build_model(input_shape, output_shape)
    # Train it with a pruned model
    num_samples = X_train.shape[0] * (1 - model.training_config["validation_split"])
    model.compile_model(num_samples)
    model.fit(X_train, y_train, pt_target_train, sample_weights)
    history = model.history
    val_accuracy = max(
        history.history[
            "val_prune_low_magnitude_jet_id_output_weighted_categorical_accuracy"
        ]
    )

    return 1 - val_accuracy


def hyperparameter_opt(yaml_path, config, data, out_dir):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, data, yaml_path, config, out_dir), n_trials=10
    )
    best_trial = study.best_trial
    print(f"best_trial: {best_trial}")
    return


def train(model, data, args, labels_to_use, config):
    # Load the data, class_labels and input variables name, not really using input variable names to be honest
    out_dir = args.output
    data_train, data_test, class_labels, input_vars, extra_vars = load_data(
        data, percentage=args.percent, test_ratio=0.2
    )

    # Use only labels spesified in the config file
    if labels_to_use == "all":
        labels_to_use = None

    # Make into ML-like data for training
    (
        X_train,
        y_train,
        pt_target_train,
        truth_pt_train,
        reco_pt_train,
        mass_target_train,
        _,
    ) = to_ML(data_train, class_labels, labels_to_use)
    X_test, y_test, _, truth_pt_test, reco_pt_test, mass_target_test, class_labels = (
        to_ML(data_test, class_labels, labels_to_use)
    )
    print(f"CLASS LABELS: {class_labels}")

    model.set_labels(
        input_vars,
        extra_vars,
        class_labels,
    )

    # Get input shape
    use_jets = True
    if use_jets:
        X_train_constits, X_train_jets = X_train
        inputs = {"constituent_inputs": X_train_constits, "jet_inputs": X_train_jets}
        constituents_shape = X_train_constits.shape[
            1:
        ]  # First dimension is batch size, input shape is NCONSTITUENTS x NFEATURES
        jets_shape = X_train_jets.shape[
            1:
        ]  # First dimension is batch size, input shape is NJETS x NFEATURES
    else:
        inputs = {"constituent_inputs": X_train}
        constituents_shape = X_train.shape[
            1:
        ]  # First dimension is batch size, input shape is NCONSTITUENTS x NFEATURES
        jets_shape = (
            None  # First dimension is batch size, input shape is NJETS x NFEATURES
        )
    print(f"JETS: {X_train_jets.shape}")
    print(f"CONSTITS: {X_train_constits.shape}")
    X_train = X_train_constits
    X_test = X_test
    """
    # Filter isfilled
    idx_train = np.sum((X_train != 0).any(axis=2), axis=1) >= 20
    X_train = X_train[idx_train]
    y_train = y_train[idx_train]
    reco_pt_train = reco_pt_train[idx_train]
    mass_target_train = mass_target_train[idx_train]

    idx_test = np.sum((X_test[0] != 0).any(axis=2), axis=1) >= 20
    X_test = (X_test[0][idx_test], X_test[1])
    y_test = y_test[idx_test]
    truth_pt_test = truth_pt_test[idx_test]
    mass_target_test = mass_target_test[idx_test]
    """
    X_train = X_train_constits
    X_test = X_test

    # ── pT > 200 GeV cut ─────────────────────────────────────────────────────
    pt_cut = 10.0
    print(f"Applying pT > {pt_cut} GeV cut...")

    train_pt_mask = reco_pt_train > pt_cut
    X_train = X_train[train_pt_mask]
    y_train = y_train[train_pt_mask]
    pt_target_train = pt_target_train[train_pt_mask]
    reco_pt_train = reco_pt_train[train_pt_mask]
    mass_target_train = mass_target_train[train_pt_mask]
    print(f"  Train: {train_pt_mask.sum()} / {len(train_pt_mask)} jets kept")

    test_pt_mask = reco_pt_test > pt_cut
    X_test = (X_test[0][test_pt_mask], X_test[1])
    y_test = y_test[test_pt_mask]
    truth_pt_test = truth_pt_test[test_pt_mask]
    reco_pt_test = reco_pt_test[test_pt_mask]
    mass_target_test = mass_target_test[test_pt_mask]
    print(f"  Test:  {test_pt_mask.sum()} / {len(test_pt_mask)} jets kept")

    # plot_deta_dphi(X_train, y_train, feat_x=3, feat_y=4)

    if config["columns"]:
        X_train = X_train[:, :, config["columns"]]
        X_test = (X_test[0][:, :, config["columns"]], X_test[1])

    if config["undersampling"]:
        idx = undersample_per_class(y_train, n_per_class=config["undersampling"][0])
        X_train, y_train, pt_target_train, reco_pt_train, mass_target_train = (
            X_train[idx, :, :],
            y_train[idx, :],
            pt_target_train[idx],
            reco_pt_train[idx],
            mass_target_train[idx],
        )

        idx = undersample_per_class(y_test, n_per_class=config["undersampling"][1])
        X_test = (X_test[0][idx, :, :], X_test[1])
        y_test, truth_pt_test, reco_pt_test, mass_target_test = (
            y_test[idx, :],
            truth_pt_test[idx],
            reco_pt_test[idx],
            mass_target_test[idx],
        )
    # Normalize and undersample
    X_train = normalize(X_train)
    X_test_features, X_test_labels = X_test
    X_test = (normalize(X_test_features), X_test_labels)
    save_test_data(
        out_dir, X_test, y_test, truth_pt_test, reco_pt_test, mass_target_test
    )

    sample_weights = train_weights(
        y_train,
        reco_pt_train,
        mass_target_train,
        class_labels,
        weightingMethod=config["training_config"]["weight_method"],
        debug=True,
        reference_class=config["training_config"]["reference"],
    )
    if sample_weights is None:
        sample_weights = np.ones(y_train.shape[0])
    if model.run_config["debug"]:
        print("DEBUG - Checking sample_weights:")
        print(sample_weights)

    # Plot pT and mass distributions before/after weighting
    plot_weighting_distributions(
        y_train,
        reco_pt_train,
        mass_target_train,
        sample_weights,
        class_labels,
        out_dir=os.path.join(out_dir, "plots"),
    )

    input_shape = X_train.shape[1:]  # First dimension is batch size
    output_shape = y_train.shape[1:]

    if config["hyperparameter_opt"]:
        hyperparameter_opt(
            args.yaml_config,
            yaml_dict,
            (
                X_train,
                y_train,
                pt_target_train,
                sample_weights,
                input_vars,
                extra_vars,
                class_labels,
                input_shape,
                output_shape,
            ),
            out_dir,
        )

        return

    min_m, max_m = mass_target_train.min(), mass_target_train.max()
    mass_bins = np.linspace(min_m, max_m, 51)

    # --- Iterate over each class to plot ---
    plt.figure(figsize=(10, 6))

    for label, idx in class_labels.items():
        # 1. Select the data for the current class
        class_mask = y_train[:, idx] == 1
        class_mass_data = mass_target_train[class_mask]
        class_weights = sample_weights[class_mask]

        plt.hist(
            class_mass_data,
            bins=mass_bins,
            histtype="step",
            density=True,  # Normalize so shapes can be compared
            label=f"{label} - Unweighted",
            linestyle="--",
            alpha=0.6,
            linewidth=2,
        )
        """
        plt.hist(
            class_mass_data,
            bins=mass_bins,
            weights=class_weights,  # Apply the weights here!
            histtype="step",
            density=True,  # Normalize so shapes can be compared
            label=f"{label} - Weighted",
            linestyle="-",
            linewidth=2 - 0.3 * (idx + 1),
        )
        """
    reference_class = config["training_config"]["reference"]
    if reference_class:
        ref_idx = class_labels[reference_class]
        ref_mask = y_train[:, ref_idx] == 1
        ref_mass_data = mass_target_train[ref_mask]

        ref_hist, _ = np.histogram(ref_mass_data, bins=mass_bins)
        target_density = ref_hist / np.sum(ref_hist)
        target_hist_plot = target_density * (len(ref_mass_data) / np.sum(ref_hist))

        bin_centers = (mass_bins[:-1] + mass_bins[1:]) / 2.0
        plt.plot(
            bin_centers,
            target_density,
            "r:",
            label=f"Target Shape ({reference_class})",
            linewidth=3,
        )

    plt.xlabel("Mass")
    plt.ylabel("Density")
    plt.title("Mass decorrelation")
    plt.legend()
    plt.grid(axis="y", alpha=0.5)
    plt.show()
    plt.savefig("mass_dist.png")

    y_train_classes = np.argmax(y_train, axis=1)
    _, counts_train = np.unique(y_train_classes, return_counts=True)
    y_test_classes = np.argmax(y_test, axis=1)
    _, counts_test = np.unique(y_test_classes, return_counts=True)
    labels = [
        label for label, idx in sorted(model.class_labels.items(), key=lambda x: x[1])
    ]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    max_count = max(counts_train.max(), counts_test.max())
    max_count_test = counts_test.max()
    ax[0].bar(labels, counts_train, edgecolor="black")
    ax[0].set_xlabel("Class label")
    ax[0].set_ylabel("Count")
    ax[0].set_title("y_train Class Distribution")
    ax[0].set_ylim(0, max_count * 1.05)
    ax[1].bar(labels, counts_test, edgecolor="black")
    ax[1].set_xlabel("Class label")
    ax[1].set_title("y_test Class Distribution")
    ax[1].set_ylim(0, max_count_test * 1.05)
    plt.suptitle("Class Distribution Comparison")
    plt.tight_layout()
    plt.savefig("y_class_distribution_comparison.png", dpi=300)
    plt.show()

    mu_mass = np.mean(mass_target_train)
    sigma_mass = np.std(mass_target_train)
    mass_target_train = (mass_target_train - mu_mass) / sigma_mass
    mass_target_test = (mass_target_test - mu_mass) / sigma_mass
    print(f"max mass: {mass_target_test.max()}")

    print(input_shape, output_shape)
    input_shape = (X_train.shape[1], X_train.shape[2])
    # output_shape = y_train.shape[1]
    print(input_shape, output_shape)
    model.build_model(input_shape, output_shape)
    if args.pretrained_encoder:
        model.load_pretrained_backbone(args.pretrained_encoder)
    # Train it with a pruned model
    num_samples = X_train.shape[0] * (1 - model.training_config["validation_split"])
    model.compile_model(num_samples)
    if config["adversarial"]:
        model.fit(X_train, y_train, pt_target_train, mass_target_train, sample_weights)
    else:
        model.fit(X_train, y_train, pt_target_train, sample_weights)

    model.save()

    model.plot_loss()

    return


def plot_hists_asra(X_train, X_train_asra):
    fatjet, asra = np.sum(X_train, axis=1)[:, 0], np.sum(X_train_asra, axis=1)[:, 0]
    a = min(fatjet.min(), asra.min())
    b = min(fatjet.max(), asra.max())
    print(a, b)
    plt.figure()
    _, bins, _ = plt.hist(fatjet, bins=50, range=[a, b])
    _ = plt.hist(asra, bins=bins, alpha=0.5)
    plt.savefig("pt_comparison.png")
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    # Training argument
    parser.add_argument(
        "-o",
        "--output",
        default="output/baseline",
        help="Output model directory path, also save evaluation plots",
    )
    parser.add_argument(
        "-p",
        "--percent",
        default=100,
        type=int,
        help="Percentage of how much processed data to train on",
    )
    parser.add_argument(
        "-y",
        "--yaml_config",
        default="tagger/model/configs/baseline.yaml",
        help="YAML config for model",
    )

    # Basic ploting
    parser.add_argument(
        "--plot-basic",
        action="store_true",
        help="Plot all the basic performance if set",
    )
    parser.add_argument(
        "-sig",
        "--signal-processes",
        default=[],
        nargs="*",
        help="Specify all signal process for individual plotting",
    )

    parser.add_argument(
        "--pretrained_encoder",
        default=None,
        help="Path to encoder_weights.h5 from pretrain_jetclr.py, warm-starts the conv1d backbone",
    )

    args = parser.parse_args()
    with open(args.yaml_config, "r") as stream:
        yaml_dict = yaml.safe_load(stream)
    dataset = yaml_dict["data"]
    labels_to_use = yaml_dict["labels"]
    # mlflow.set_experiment(os.getenv('CI_COMMIT_REF_NAME'))

    if args.plot_basic:
        # All the basic plots!
        model = fromFolder(args.output, yaml_dict)
        results = basic(model, args.signal_processes, yaml_dict["pt_regress"])
        # if os.path.isfile("mlflow_run_id.txt"):
        #     f = open("mlflow_run_id.txt", "r")
        #     run_id = (f.read())
        #     mlflow.get_experiment_by_name(os.getenv('CI_COMMIT_REF_NAME'))
        #     with mlflow.start_run(experiment_id=1,
        #                         run_name=args.name,
        #                         run_id=run_id # pass None to start a new run
        #                         ):
        #         for class_label in results.keys():
        #             mlflow.log_metric(class_label + ' ROC AUC',results[class_label])

    else:
        model = fromYaml(args.yaml_config, yaml_dict, args.output)
        train(model, dataset, args, labels_to_use, yaml_dict)
        # with mlflow.start_run(run_name=args.name) as run:
        #     mlflow.set_tag('gitlab.CI_JOB_ID', os.getenv('CI_JOB_ID'))
        #     mlflow.keras.autolog()
        #     train(args.output, args.yaml_config, args.percent)
        #     run_id = run.info.run_id
        # sourceFile = open('mlflow_run_id.txt', 'w')
        # print(run_id, end="", file = sourceFile)
        # sourceFile.close()
