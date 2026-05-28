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
tf.config.set_visible_devices(gpus[1], "GPU")
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
    """
    Re-balancing the class weights and then flatten them based on truth pT
    """
    if weightingMethod not in ["none", "ptref", "onlyclass", "mass"]:
        raise ValueError(
            "Oops!  Given weightingMethod not defined in train_weights(). Use either none, ptref, or onlyclass."
        )
    num_samples = y_train.shape[0]
    num_classes = y_train.shape[1]

    sample_weights = np.ones(num_samples)

    # Define pT bins (without the high pT part we don't care about)
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
            np.inf,  # Use np.inf to cover all higher values
        ]
    )

    if weightingMethod == "onlyclass":
        pt_bins = np.array(
            [
                0.0,
                np.inf,  # Use np.inf to cover all higher values
            ]
        )

    # Initialize counts per class per pT bin
    class_pt_counts = {}

    # Calculate counts per class per pT bin
    for label, idx in class_labels.items():
        class_mask = y_train[:, idx] == 1
        class_pt_counts[idx], _ = np.histogram(reco_pt_train[class_mask], bins=pt_bins)

    # Compute the maximum counts per pT bin over all classes
    max_counts_per_bin = np.zeros(len(pt_bins) - 1)
    min_counts_per_bin = np.zeros(len(pt_bins) - 1)
    for bin_idx in range(len(pt_bins) - 1):
        counts_in_bin = [class_pt_counts[idx][bin_idx] for idx in class_labels.values()]
        max_counts_per_bin[bin_idx] = max(counts_in_bin)
        min_counts_per_bin[bin_idx] = min(counts_in_bin)

    # Weight all to one base class (b = 0)
    counts_per_bin = class_pt_counts[0]

    if weightingMethod == "ptref":
        # Try minimum and flat
        counts_per_bin = [min(min_counts_per_bin) for __ in min_counts_per_bin]
        # Try maximum and flat
        # counts_per_bin = [max(max_counts_per_bin) for __ in max_counts_per_bin]

    # Compute weights per class per pT bin
    weights_per_class_pt_bin = {}
    for idx in class_labels.values():
        weights_per_class_pt_bin[idx] = np.zeros(len(pt_bins) - 1)
        for bin_idx in range(len(pt_bins) - 1):
            class_count = class_pt_counts[idx][bin_idx]
            if class_count == 0:
                weights_per_class_pt_bin[idx][bin_idx] = 0.0
            else:
                weights_per_class_pt_bin[idx][bin_idx] = (
                    counts_per_bin[bin_idx] / class_count
                )

    # Multiply by some custom class weights
    # All same weight
    weights_per_class = {idx: 1.0 for idx in class_labels.values()}
    for idx in class_labels.values():
        weights_per_class_pt_bin[idx] = (
            weights_per_class_pt_bin[idx] * weights_per_class[idx]
        )

    # Assign weights to samples
    for idx in class_labels.values():
        class_mask = y_train[:, idx] == 1
        class_truth_pt = reco_pt_train[class_mask]
        sample_indices = np.where(class_mask)[0]
        bin_indices = (
            np.digitize(class_truth_pt, pt_bins) - 1
        )  # Subtract 1 to get 0-based index
        bin_indices[bin_indices == len(pt_bins) - 1] = (
            len(pt_bins) - 2
        )  # Handle right edge
        sample_weights[sample_indices] = weights_per_class_pt_bin[idx][bin_indices]

        # Print weighted jets as closure test in debug mode
        if debug and weightingMethod != "none":
            print(
                "DEBUG - Checking jets weighted by sample_weights as a function of pT:"
            )
            print(
                np.histogram(
                    class_truth_pt, bins=pt_bins, weights=sample_weights[sample_indices]
                )
            )

    if weightingMethod == "mass":
        min_m, max_m = mass_target_train.min(), mass_target_train.max()
        mass_bins = np.linspace(min_m, max_m, 51)
        MAX_MASS_WEIGHT_CAP = 5.0

        # Get reference (signal) mass distribution as target
        ref_idx = class_labels[reference_class]
        ref_mask = y_train[:, ref_idx] == 1
        ref_mass_data = mass_target_train[ref_mask]
        ref_hist, _ = np.histogram(ref_mass_data, bins=mass_bins)

        if np.sum(ref_hist) == 0:
            raise ValueError(f"Reference class '{reference_class}' has no mass data.")

        target_density = ref_hist / np.sum(ref_hist)

        for label, idx in class_labels.items():
            class_mask = y_train[:, idx] == 1
            sample_indices = np.where(class_mask)[0]
            class_mass_data = mass_target_train[
                class_mask
            ]  # ← move this BEFORE the continue

            # Skip all signal classes — keep their weights at 1.0
            if label != "background":
                sample_weights[sample_indices] = 1.0
                continue

            # Only reweight background to match reference shape
            hist, _ = np.histogram(class_mass_data, bins=mass_bins)

            mass_weights_per_bin = np.zeros(len(mass_bins) - 1)
            class_total = np.sum(hist)

            if class_total > 0:
                class_density = hist / class_total
                for bin_idx in range(len(mass_bins) - 1):
                    p_class = class_density[bin_idx]
                    p_target = target_density[bin_idx]
                    if p_class > 0:
                        mass_weights_per_bin[bin_idx] = min(
                            p_target / p_class, MAX_MASS_WEIGHT_CAP
                        )

            mass_bin_indices = np.digitize(class_mass_data, mass_bins) - 1
            mass_bin_indices = np.clip(mass_bin_indices, 0, len(mass_bins) - 2)

            # Multiply into existing pT weights rather than overwrite
            sample_weights[sample_indices] *= mass_weights_per_bin[mass_bin_indices]

            if debug:
                print(f"DEBUG - Mass reweighted class '{label}' (idx {idx})")
                print(f"  mass_weights_per_bin: {mass_weights_per_bin}")

    # Normalize sample weights
    sample_weights = sample_weights / np.mean(sample_weights)

    if weightingMethod == "none":
        return None
    else:
        return sample_weights


def train_weights22(
    y_train,
    reco_pt_train,
    mass_target_train,
    class_labels,
    weightingMethod,
    debug,
    reference_class=None,
):
    if weightingMethod not in ["none", "ptref", "onlyclass", "mass"]:
        raise ValueError("Oops! Given weightingMethod not defined.")

    num_samples = y_train.shape[0]
    sample_weights = np.ones(num_samples)

    print(f"y_train shape: {y_train.shape}")
    print(f"reco_pt_train shape: {reco_pt_train.shape}")
    print(f"Mass: {mass_target_train.min()} and {mass_target_train.max()}")

    if weightingMethod == "onlyclass":
        class_counts = {idx: np.sum(y_train[:, idx]) for idx in class_labels.values()}
        non_zero_counts = np.array([c for c in class_counts.values() if c > 0])
        target_count = np.median(non_zero_counts) if len(non_zero_counts) > 0 else 1.0

        weights_per_class = {}
        for idx, count in class_counts.items():
            if count == 0:
                weights_per_class[idx] = 0.0
            else:
                raw_weight = target_count / count
                weights_per_class[idx] = min(raw_weight, 100.0)

            class_mask = y_train[:, idx] == 1
            sample_weights[class_mask] = weights_per_class[idx]

    mean_weight = np.mean(sample_weights)
    if mean_weight == 0:
        sample_weights = np.ones(num_samples)
    else:
        sample_weights = sample_weights / mean_weight

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


def normalize_old(X_train):
    print(X_train.shape)
    mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    std = np.std(X_train, axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0
    return (X_train - mean) / std


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
    np.random.seed(seed)
    y_labels = np.argmax(y, axis=1)
    unique_classes = np.unique(y_labels)

    selected_indices = []

    for c in unique_classes:
        class_indices = np.where(y_labels == c)[0]
        chosen = np.random.choice(
            class_indices, size=min(n_per_class, len(class_indices)), replace=False
        )
        selected_indices.extend(chosen)

    return np.array(selected_indices)


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
    if model.run_config["debug"]:
        print("DEBUG - Checking sample_weights:")
        print(sample_weights)

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

    """
    X_train_asra = np.load(
        f"/home/akseli/l1-jet-id/data/16const/processed/proc_train_16const.npy"
    )
    X_test_asra = np.load(
        f"/home/akseli/l1-jet-id/data/16const/processed/proc_test_16const.npy"
    )
    y_train_asra = np.load(
        f"/home/akseli/l1-jet-id/data/16const/processed/proc_labels_train_16const.npy"
    )
    y_test_asra = np.load(
        f"/home/akseli/l1-jet-id/data/16const/processed/proc_labels_test_16const.npy"
    )

    print(
        f"X_train shape {X_train.shape}, X_test {X_test[0].shape}, y_train {y_train.shape}, y_test {y_test.shape}"
    )

    plot_hists_asra(X_train, normalize(X_train_asra))
    """
    print(input_shape, output_shape)
    input_shape = (X_train.shape[1], X_train.shape[2])
    # output_shape = y_train.shape[1]
    print(input_shape, output_shape)
    model.build_model(input_shape, output_shape)
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
