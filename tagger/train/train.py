import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
import copy

# Third parties
import numpy as np
import yaml
import tensorflow as tf

# Enable GPU usage and avoid TF pre-allocating all memory
gpus = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(gpus[2:], "GPU")
import optuna

# Import from other modules
from tagger.data.tools import load_data, to_ML
from tagger.model.common import fromFolder, fromYaml
from tagger.plot.basic import basic


if gpus:
    try:
        for gpu in gpus:
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


def save_test_data(out_dir, X_test, y_test, truth_pt_test, reco_pt_test):
    os.makedirs(os.path.join(out_dir, "testing_data"), exist_ok=True)
    # X_test_constits, X_test_jets = X_test
    np.save(os.path.join(out_dir, "testing_data/X_test.npy"), X_test[0])
    np.save(os.path.join(out_dir, "testing_data/y_test.npy"), y_test)
    np.save(os.path.join(out_dir, "testing_data/truth_pt_test.npy"), truth_pt_test)
    np.save(os.path.join(out_dir, "testing_data/reco_pt_test.npy"), reco_pt_test)

    print(f"Test data saved to {out_dir}")


def train_weights(y_train, reco_pt_train, class_labels, weightingMethod, debug):
    """
    Re-balancing the class weights and then flatten them based on truth pT
    """
    if weightingMethod not in ["none", "ptref", "onlyclass"]:
        raise ValueError(
            "Oops!  Given weightingMethod not defined in train_weights(). Use either none, ptref, or onlyclass."
        )
    num_samples = y_train.shape[0]

    sample_weights = np.ones(num_samples)

    print("y_train shape:", y_train.shape)
    print("reco_pt_train shape:", reco_pt_train.shape)

    # Define pT bins (without the high pT part we don't care about)
    pt_bins = np.array(
        [15, 17, 19, 22, 25, 30, 35, 40, 45, 50, 60, 76, 97, 122, 154, np.inf]
    )  # Use np.inf to cover all higher values

    if weightingMethod == "onlyclass":
        # Use np.inf to cover all higher values
        pt_bins = np.array([0.0, np.inf])

    # Initialize counts per class per pT bin
    class_pt_counts = {}
    print("y_train shape:", y_train.shape)
    print("type(y_train):", type(y_train))
    print("reco_pt_train shape:", reco_pt_train.shape)

    # Calculate counts per class per pT bin
    for _label, idx in class_labels.items():
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
    weights_per_class = {
        0: 1,  # b
        1: 1,  # charm
        2: 1.0,  # light
        3: 1.0,  # gluon
        4: 1.0,  # taup
        5: 1.0,  # taum
        6: 1.0,  # muon
        7: 1.0,  # electron
    }
    num_classes = y_train.shape[-1]
    weights_per_class = {idx: 1.0 for idx in range(num_classes)}
    for idx in class_labels.values():
        weights_per_class_pt_bin[idx] = (
            weights_per_class_pt_bin[idx] * weights_per_class[idx]
        )

    # Assign weights to samples
    for idx in class_labels.values():
        class_mask = y_train[:, idx] == 1
        class_truth_pt = reco_pt_train[class_mask]
        sample_indices = np.where(class_mask)[0]
        # Subtract 1 to get 0-based index
        bin_indices = np.digitize(class_truth_pt, pt_bins) - 1
        # Handle right edge
        bin_indices[bin_indices == len(pt_bins) - 1] = len(pt_bins) - 2
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

    # Normalize sample weights
    sample_weights = sample_weights / np.mean(sample_weights)

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

    plt.hist(X_train[:, :, 5])
    plt.savefig("mass.png", dpi=200, bbox_inches="tight")


def normalize(X_train):
    print(X_train.shape)
    mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    std = np.std(X_train, axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0
    return (X_train - mean) / std


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
        sample_weight,
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
        "batch_size", [256, 512, 1024]
    )
    config["training_config"]["epochs"] = trial.suggest_int(
        "epochs", 100, 600, step=100
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
    model.fit(X_train, y_train, pt_target_train, sample_weight)
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
        lambda trial: objective(trial, data, yaml_path, config, out_dir), n_trials=30
    )
    best_trial = study.best_trial
    print(f"best_trial: {best_trial}")
    return


def train(model, data, args, labels_to_use, config):
    # Load the data, class_labels and input variables name, not really using input variable names to be honest
    out_dir = args.output
    data_train, data_test, class_labels, input_vars, extra_vars = load_data(
        data, percentage=args.percent
    )

    """
    # Use only labels spesified in the config file
    if labels_to_use != 'all':
        class_labels = {k: v for k, v in class_labels.items()
                                                            if k in labels_to_use}
        class_labels = {k: i for i, k in enumerate(class_labels.keys())}
    """

    # Make into ML-like data for training
    X_train, y_train, pt_target_train, truth_pt_train, reco_pt_train, _ = to_ML(
        data_train, class_labels, labels_to_use
    )
    X_test, y_test, _, truth_pt_test, reco_pt_test, class_labels = to_ML(
        data_test, class_labels, labels_to_use
    )

    print(f"CLASS LABELS: {class_labels}")

    model.set_labels(
        input_vars,
        extra_vars,
        class_labels,
    )

    def undersample_majority_class(X, y, keep_frac=0.1, random_state=42):
        """
        Keep only a fraction of the samples from the largest class.

        Args:
            X (np.ndarray): Features, shape (n_samples, ...).
            y (np.ndarray): One-hot labels, shape (n_samples, n_classes).
            keep_frac (float): Fraction of majority class to keep (e.g., 0.1 = 10%).
            random_state (int): Random seed for reproducibility.

        Returns:
            X_new, y_new (undersampled arrays).
        """
        rng = np.random.default_rng(random_state)

        # Get class counts
        class_counts = y.sum(axis=0)
        majority_class = np.argmax(class_counts)

        print(
            f"Majority class: {majority_class}",
            f"count = {class_counts[majority_class]}",
        )

        # Indices of samples in each group
        idx_majority = np.where(y[:, majority_class] == 1)[0]
        idx_other = np.where(y[:, majority_class] == 0)[0]

        # Randomly choose 10% of majority class
        keep_size = int(len(idx_majority) * keep_frac)
        idx_keep_majority = rng.choice(idx_majority, size=keep_size, replace=False)

        # Combine back
        new_idx = np.concatenate([idx_keep_majority, idx_other]).astype(int)
        rng.shuffle(new_idx)
        print(new_idx)
        print(new_idx.shape)

        return new_idx

    print("Before:", y_train.sum(axis=0)[np.argmax(y_train.sum(axis=0))])
    new_idx = undersample_majority_class(X_train, y_train, keep_frac=0.2)
    y_train = y_train[new_idx]
    reco_pt_train = reco_pt_train[new_idx]
    X_train = tuple(x[new_idx] for x in X_train)
    print("After:", y_train.sum(axis=0)[np.argmax(y_train.sum(axis=0))])

    new_idx = undersample_majority_class(X_test, y_test, keep_frac=0.2)
    y_test = y_test[new_idx]
    reco_pt_test = reco_pt_test[new_idx]
    X_test = tuple(x[new_idx] for x in X_test)
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

    X_train = X_train_constits
    X_test = X_test
    '''
    # Check for multi-label rows (more than one "1" per sample)
    invalid_rows = np.where(y_train.sum(axis=1) > 1)[0]
    num_invalid = len(invalid_rows)

    # Check for samples missing a label (all zeros)
    missing_rows = np.where(y_train.sum(axis=1) == 0)[0]
    num_missing = len(missing_rows)

    print(X_train.shape)
    # Check input/label length consistency
    if len(X_train) != len(y_train):
        print(
            f" Mismatch: X_train has {len(X_train)} entries, y_train has",
            f"{len(y_train)}.",
        )

    # Report label anomalies
    if num_invalid > 0:
        print(
            f"⚠️ Found {num_invalid}",
            "samples with multiple active labels (multi-label rows).",
        )
        print(f"Example indices: {invalid_rows[:10]}")
    else:
        print("✅ All rows have a single active label.")

    if num_missing > 0:
        print(f"⚠️ Found {num_missing}",
              "samples with no active label (all zeros).")
        print(f"Example indices: {missing_rows[:10]}")
    
    # Flatten each jet to a 1D vector
    X_flat = X_train.reshape(X_train.shape[0], -1)

    # Find unique rows and their counts
    unique_X, unique_indices, counts = np.unique(
        X_flat, axis=0, return_index=True, return_counts=True
    )

    # Find duplicate indices (jets that appear more than once)
    duplicate_indices = np.where(counts > 1)[0]

    if len(duplicate_indices) > 0:
        print(
            f"Found {len(duplicate_indices)} duplicated jets out of",
            f"{len(X_train)} total.",
        )
        print(
            "Indices of duplicates (first occurrences):",
            unique_indices[duplicate_indices],
        )
    else:
        print("No duplicate jets found.")

    def check_shifted_duplicates(X_train, max_shift=5, tol=1e-8):
        """Detect jets that are identical up to a shift along the constituent axis."""
        n_jets = X_train.shape[0]
        shifted_pairs = []

        for i in range(n_jets):
            for j in range(i + 1, n_jets):
                for shift in range(1, max_shift + 1):
                    # compare forward shift
                    if np.allclose(
                        X_train[i, :-shift, :], X_train[j, shift:, :], atol=tol
                    ):
                        shifted_pairs.append((i, j, shift, "forward"))
                        break
                    # compare backward shift
                    if np.allclose(
                        X_train[i, shift:, :], X_train[j, :-shift, :], atol=tol
                    ):
                        shifted_pairs.append((i, j, shift, "backward"))
                        break
        return shifted_pairs

    # Example use
    shifted = check_shifted_duplicates(X_train, max_shift=5)
    if shifted:
        for a, b, shift, direction in shifted:
            print(f"Jets {a} and {b} match with a",
                  f"{direction} shift of {shift}")
    else:
        print("No shifted duplicates found.")
    '''

    X_train = normalize(X_train)
    X_test_features, X_test_labels = X_test
    X_test = (normalize(X_test_features), X_test_labels)

    """
    idx = undersample_per_class(y_train, n_per_class=5000)
    X_train, y_train, pt_target_train, reco_pt_train = (
        X_train[idx, :, :],
        y_train[idx, :],
        pt_target_train[idx],
        reco_pt_train[idx],
    )

    idx = undersample_per_class(y_test, n_per_class=1000)
    X_test = (X_test[0][idx, :, :], X_test[1])
    y_test, truth_pt_test, reco_pt_test = (
        y_test[idx, :],
        truth_pt_test[idx],
        reco_pt_test[idx],
    )
    """
    plot_deta_dphi(X_train, y_train, feat_x=3, feat_y=4)

    save_test_data(out_dir, X_test, y_test, truth_pt_test, reco_pt_test)
    print("y_train shape:", y_train.shape)
    print("type(y_train):", type(y_train))
    print("reco_pt_train shape:", reco_pt_train.shape)

    print(f"Y_DIST: {np.sum(y_train, axis=0)}")
    print(f"Y_1: {y_train[0]}")

    # Calculate the sample weights for training
    sample_weight = train_weights(
        y_train,
        reco_pt_train,
        class_labels,
        weightingMethod=model.training_config["weight_method"],
        debug=model.run_config["debug"],
    )
    if model.run_config["debug"]:
        print("DEBUG - Checking sample_weight:")
        print(sample_weight)

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
                sample_weight,
                input_vars,
                extra_vars,
                class_labels,
                input_shape,
                output_shape,
            ),
            out_dir,
        )

        return
    model.build_model(input_shape, output_shape)
    # Train it with a pruned model
    num_samples = X_train.shape[0] * (1 - model.training_config["validation_split"])
    model.compile_model(num_samples)
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
