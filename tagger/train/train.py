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

os.environ["KERAS_BACKEND"] = "torch"

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


def train(model, out_dir, percent, ebops):

    # Load the data, class_labels and input variables name, not really using input variable names to be honest
    data_train, _, class_labels, input_vars, extra_vars = load_data("training_data/", percentage=percent)
    
    data_test, _, class_labels, input_vars, extra_vars = load_data("testing_data/", percentage=100)
    
    model.set_labels(
        input_vars,
        extra_vars,
        class_labels,
    )

    # Make into ML-like data for training
    X_train, y_train, pt_target_train, truth_pt_train, reco_pt_train = to_ML(data_train, class_labels)

    # Save X_test, y_test, and truth_pt_test for plotting later
    X_test, y_test, _, truth_pt_test, reco_pt_test = to_ML(data_test, class_labels)
    save_test_data(out_dir, X_test, y_test, truth_pt_test, reco_pt_test)

    # Calculate the sample weights for training
    sample_weight = train_weights(
        y_train,
        reco_pt_train,
        class_labels,
        weightingMethod=model.training_config['weight_method'],
        debug=model.run_config['debug'],
    )
    if model.run_config['debug']:
        print("DEBUG - Checking sample_weight:")
        print(sample_weight)

    # Get input shape
    input_shape = X_train.shape[1:]  # First dimension is batch size
    output_shape = y_train.shape[1:]
    
    print("Total jets for training: ", X_train.shape[0:] )

    model.build_model(input_shape, output_shape)
    # Train it with a pruned model
    num_samples = X_train.shape[0] * (1 - model.training_config['validation_split'])
    model.compile_model(num_samples, ebops)
    model.fit(X_train, y_train, pt_target_train, sample_weight)

    model.save()

    model.plot_loss()

    return


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
        '-e', '--ebops', default=300000, type=int
    )

    args = parser.parse_args()

    if args.plot_basic:
        # All the basic plots!
        model = fromFolder(args.output)
        results = basic(model, args.signal_processes)

    else:
        model = fromYaml(args.yaml_config, args.output)
        train(model, args.output, args.percent, args.ebops)
