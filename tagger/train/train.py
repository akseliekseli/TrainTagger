import gc
import os
from argparse import ArgumentParser

# Set the backend BEFORE importing model modules/Keras.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

# Third parties
import numpy as np
import awkward as ak
import yaml

# Import from other modules
from tagger.data.tools import load_data, to_ML
from tagger.model.common import fromFolder, fromYaml
from tagger.plot.basic import basic

from tagger.train.streaming_data import (
    RootTrainingData, check_test_samples, save_test_data_streaming,
)

def save_test_data(out_dir, X_test, y_test, truth_pt_test, reco_pt_test):

    os.makedirs(os.path.join(out_dir, 'testing_data'), exist_ok=True)

    np.save(os.path.join(out_dir, "testing_data/X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "testing_data/y_test.npy"), y_test)
    np.save(os.path.join(out_dir, "testing_data/truth_pt_test.npy"), truth_pt_test)
    np.save(os.path.join(out_dir, "testing_data/reco_pt_test.npy"), reco_pt_test)

    print(f"Test data saved to {out_dir}")


def train_weights(y_train, reco_pt_train, class_labels, weightingMethod, reco_mass_train=None, debug=False):
    """
    Re-balancing the class weights and then flatten them based on truth pT or mass
    """
    if weightingMethod not in ["none", "ptref", "onlyclass", "massref","ptmassref"]:
        raise ValueError(
            "weightingMethod must be none, ptref, onlyclass, massref, or ptmassref"
        )
    if weightingMethod == "none":
        return None
    
    num_samples = y_train.shape[0]

    sample_weights = np.ones(num_samples)

    if weightingMethod == "massref":
        if reco_mass_train is None:
            raise ValueError("massref requires reco_mass_train")
    
        mass = np.asarray(reco_mass_train)
    
        if mass.ndim != 1 or len(mass) != len(y_train):
            raise ValueError("Expected one reconstructed mass per training jet")
    
        # Starting bin choices, in GeV.
        mass_bins = np.array(
            [0, 40, 70, 90, 110, 130, 150, 180, 220, np.inf]
        )
        valid = np.isfinite(mass) & (mass >= mass_bins[0])
    
        counts = {}
        for label, idx in class_labels.items():
            mask = (y_train[:, idx] == 1) & valid
            if not np.any(mask):
                raise ValueError(
                    f"Class {label} has no jets with valid mass"
                )
            counts[idx], _ = np.histogram(
                mass[mask], bins=mass_bins
            )
    
        # Give every class the same weighted mass histogram.
        target = np.minimum.reduce(list(counts.values()))
    
        if not np.any(target > 0):
            raise ValueError(
                "No mass bins populated by every class. "
                "Use coarser mass bins or review the class selection."
            )
    
        mass_bin = np.searchsorted(
            mass_bins, mass, side="right"
        ) - 1
    
        weights = np.zeros(len(y_train), dtype=np.float32)
    
        for idx in class_labels.values():
            mask = (y_train[:, idx] == 1) & valid
            bins = mass_bin[mask]
            weights[mask] = target[bins] / counts[idx][bins]
    
        mean_weight = weights.mean(dtype=np.float64)
        if not np.isfinite(mean_weight) or mean_weight <= 0:
            raise ValueError("Invalid mass weights")
    
        weights /= mean_weight
    
        if debug:
            print(
                "Jets with nonzero mass weight:",
                np.count_nonzero(weights),
                "/",
                len(weights),
            )
            for label, idx in class_labels.items():
                mask = (y_train[:, idx] == 1) & valid
                histogram, _ = np.histogram(
                    mass[mask],
                    bins=mass_bins,
                    weights=weights[mask],
                )
                print(f"{label}: weighted mass counts = {histogram}")
    
        return weights

    # Define pT bins (without the high pT part we don't care about)
    pt_bins = np.array(
        [15, 17, 19, 22, 25, 30, 35, 40, 45, 50, 60, 76, 97, 122, 154, np.inf]
    )  # Use np.inf to cover all higher values

    if weightingMethod == "onlyclass":
        pt_bins = np.array([0.0, np.inf])  # Use np.inf to cover all higher values

    # Initialize counts per class per pT bin
    class_pt_counts = {}

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
    
    if weightingMethod == "ptmassref":
        if reco_mass_train is None:
            raise ValueError("ptmassref requires reco_mass_train")
    
        # Starting bins only: inspect your SC8 distributions and adjust.
        pt_bins = np.array([15, 30, 50, 80, 120, 200, 350, np.inf])
        mass_bins = np.array([0, 40, 70, 90, 110, 130, 150, 180, 220, np.inf])
    
        counts = {}
        for idx in class_labels.values():
            mask = y_train[:, idx] == 1
            if not np.any(mask):
                raise ValueError(f"Class {idx} has no training jets")
            counts[idx], _, _ = np.histogram2d(
                reco_pt_train[mask], reco_mass_train[mask],
                bins=(pt_bins, mass_bins),
            )
    
        # Use only (pT, mass) bins populated by every class.
        target = np.minimum.reduce(list(counts.values()))
        if not np.any(target):
            raise ValueError("No common (pT, mass) bins; use coarser bins")
    
        pt_bin = np.searchsorted(pt_bins, reco_pt_train, side="right") - 1
        mass_bin = np.searchsorted(mass_bins, reco_mass_train, side="right") - 1
        valid = (
            (pt_bin >= 0) & (pt_bin < len(pt_bins) - 1)
            & (mass_bin >= 0) & (mass_bin < len(mass_bins) - 1)
        )
    
        weights = np.zeros(len(y_train))
        for idx in class_labels.values():
            mask = (y_train[:, idx] == 1) & valid
            weights[mask] = (
                target[pt_bin[mask], mass_bin[mask]]
                / counts[idx][pt_bin[mask], mass_bin[mask]]
            )
    
        if debug:
            print("Jets with nonzero pT–mass weight:",
                  np.count_nonzero(weights), "/", len(weights))
        return weights / weights.mean()
    
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
                weights_per_class_pt_bin[idx][bin_idx] = counts_per_bin[bin_idx] / class_count

    # Multiply by some custom class weights
    # All same weight
    #weights_per_class = {
    #    0: 1,  # b
    #    1: 1,  # charm
    #    2: 1.0,  # light
    #    3: 1.0,  # gluon
    #    4: 1.0,  # taup
    #    5: 1.0,  # taum
    #    6: 1.0,  # muon
    #    7: 1.0,  # electron
    #}
    weights_per_class = {
        index: 1.0
        for index in class_labels.values()
    }
    for idx in class_labels.values():
        weights_per_class_pt_bin[idx] = weights_per_class_pt_bin[idx] * weights_per_class[idx]

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
            print("DEBUG - Checking jets weighted by sample_weights as a function of pT:")
            print(np.histogram(class_truth_pt, bins=pt_bins, weights=sample_weights[sample_indices]))

    # Normalize sample weights
    sample_weights = sample_weights / np.mean(sample_weights)

    if weightingMethod == "none":
        return None
    return sample_weights

def apply_class_config(
    data_train,
    data_test,
    source_class_labels,
    class_config_path,
):
    with open(class_config_path) as handle:
        config = yaml.safe_load(handle)

    class_groups = config["classes"]

    if not isinstance(class_groups, dict):
        raise ValueError(
            "'classes' in the class configuration must be a mapping"
        )

    known_labels = set(source_class_labels)
    already_used = set()
    active_groups = []

    for output_label, source_labels in class_groups.items():
        if not isinstance(source_labels, list) or not source_labels:
            raise ValueError(
                f"Class group '{output_label}' must contain a list"
            )

        unknown = set(source_labels) - known_labels
        if unknown:
            raise ValueError(
                f"Unknown source labels in '{output_label}': "
                f"{sorted(unknown)}"
            )

        duplicated = set(source_labels) & already_used
        if duplicated:
            raise ValueError(
                f"Source labels occur in more than one group: "
                f"{sorted(duplicated)}"
            )

        already_used.update(source_labels)

        train_mask = ak.zeros_like(
            data_train["class_label"],
            dtype=bool,
        )
        test_mask = ak.zeros_like(
            data_test["class_label"],
            dtype=bool,
        )

        for source_label in source_labels:
            old_index = source_class_labels[source_label]

            train_mask = train_mask | (
                data_train["class_label"] == old_index
            )
            test_mask = test_mask | (
                data_test["class_label"] == old_index
            )

        train_count = int(ak.sum(train_mask))
        test_count = int(ak.sum(test_mask))

        print(
            f"{output_label}: "
            f"{train_count} training jets, "
            f"{test_count} testing jets"
        )

        if train_count == 0:
            print(
                f"Skipping configured class '{output_label}' "
                "because it has no training entries"
            )
            continue

        active_groups.append((output_label, source_labels))

    if len(active_groups) < 2:
        raise ValueError(
            "The class configuration produced fewer than two "
            "non-empty classes"
        )

    output_class_labels = {
        output_label: new_index
        for new_index, (output_label, _) in enumerate(active_groups)
    }

    def remap(data):
        new_labels = ak.full_like(
            data["class_label"],
            -1,
            dtype=np.int64,
        )

        for new_index, (_, source_labels) in enumerate(active_groups):
            group_mask = ak.zeros_like(
                data["class_label"],
                dtype=bool,
            )

            for source_label in source_labels:
                old_index = source_class_labels[source_label]
                group_mask = group_mask | (
                    data["class_label"] == old_index
                )

            new_labels = ak.where(
                group_mask,
                new_index,
                new_labels,
            )

        # Drop jets whose labels were not included in the configuration.
        keep = new_labels >= 0

        return ak.with_field(
            data[keep],
            new_labels[keep],
            "class_label",
        )

    data_train = remap(data_train)
    data_test = remap(data_test)

    print("Final model classes:", output_class_labels)

    return data_train, data_test, output_class_labels


def train(
    model, out_dir, percent, ebops, data_dir, class_config, test_data_dirs=None,
    stream_read_entries=32768, stream_shuffle_jets=50000,
    stream_prefetch=1, stream_seed=42,
):
    import json
    import keras

    if keras.backend.backend() != "tensorflow":
        raise ValueError("This streaming HGQ2 path requires KERAS_BACKEND=tensorflow")

    training = RootTrainingData(
        os.path.join(data_dir, "training_data"),
        class_config, percent, model.training_config["weight_method"],
        read_entries=stream_read_entries,
    )
    if test_data_dirs is None:
        test_data_dirs = [os.path.join(data_dir, "testing_data")]
    test_samples = check_test_samples(test_data_dirs, training)

    validation_split = model.training_config["validation_split"]
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must lie between 0 and 1")
    # Match Keras's existing array split: the final fraction BEFORE shuffling.
    split_at = int(training.n_samples * (1 - validation_split))
    if not 0 < split_at < training.n_samples:
        raise ValueError("Not enough selected jets for training and validation")

    print("Model output classes:", training.class_labels, flush=True)
    print(f"Fit jets: {split_at}; validation jets: {training.n_samples-split_at}", flush=True)
    model.set_labels(
        training.variables["inputs"], training.variables["extras"], training.class_labels,
    )
    model.build_model(training.input_shape, (len(training.class_labels),))
    model.compile_model(split_at, ebops)

    batch_size = model.training_config["batch_size"]
    train_dataset = training.dataset(
        0, split_at, batch_size, shuffle_jets=stream_shuffle_jets,
        seed=stream_seed, prefetch=stream_prefetch,
    )
    validation_dataset = training.dataset(
        split_at, training.n_samples, batch_size, prefetch=stream_prefetch,
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "streaming_training.json"), "w") as stream:
        json.dump({
            "class_labels": training.class_labels,
            "class_groups": training.groups,
            "selected_jets": training.n_samples,
            "fit_jets": split_at,
            "validation_jets": training.n_samples-split_at,
            "validation_selection": "last fraction of selected rows, before shuffle",
            "weight_statistics": "full selected training directory, including validation (unchanged)",
            "weight_method": training.weights.method,
            "weight_fields": training.weights.fields,
            "weight_bin_edges": [
                [float(value) if np.isfinite(value) else "inf" for value in edges]
                for edges in training.weights.edges
            ],
            "weight_table": None if training.weights.table is None else training.weights.table.tolist(),
            "read_entries": stream_read_entries,
            "shuffle_jets": stream_shuffle_jets,
            "prefetch_batches": stream_prefetch,
            "seed": stream_seed,
            "training_files": training.sample["files"],
            "test_directories": [str(path) for path in test_data_dirs],
        }, stream, indent=2)

    # Keep one fit call and the callbacks created by compile_model().
    # The wrapper's array-only fit() cannot accept a streamed validation set.
    history = model.jet_model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=model.training_config["epochs"],
        verbose=model.run_config["verbose"],
        callbacks=model.callbacks,
    )
    model.history = history.history
    model.save()
    del train_dataset, validation_dataset
    gc.collect()

    # Held-out samples never enter fit() or its validation dataset.
    save_test_data_streaming(out_dir, test_samples, training)
    model.plot_loss()


if __name__ == "__main__":

    parser = ArgumentParser()
    # Training argument
    parser.add_argument(
        '-o', '--output', default='output/baseline', help='Output model directory path, also save evaluation plots'
    )
    parser.add_argument('-p', '--percent', default=100,type=float, help='Percentage of how much processed data to train on')
    parser.add_argument(
        '-y', '--yaml_config', default='tagger/model/configs/baseline_larger.yaml', help='YAML config for model'
    )
    parser.add_argument(
        "--data-dir",
        default="/eos/home-a/asuutari/FastPUPPI/XtoHH-qcd",
        help="Directory containing training_data/ and testing_data/",
    )

    # Basic ploting
    parser.add_argument('--plot-basic', action='store_true', help='Plot all the basic performance if set')
    parser.add_argument(
        '-sig', '--signal-processes', default=[], nargs='*', help='Specify all signal process for individual plotting'
    )
    
    parser.add_argument(
        '-e', '--ebops', default=300000, type=int
    )

    parser.add_argument(
        "--class-config",
        default=None,
        help="YAML configuration defining how source labels are grouped",
    )
    parser.add_argument(
        "--test-data-dirs",
        nargs="+",
        default=None,
        help="Prepared evaluation dataset directories",
    )

    parser.add_argument("--stream-read-entries", type=int, default=32768,
                        help="Maximum requested rows per ROOT read block")
    parser.add_argument("--stream-shuffle-jets", type=int, default=50000,
                        help="Bounded training shuffle buffer; 0 disables shuffling")
    parser.add_argument("--stream-prefetch", type=int, default=1,
                        help="Number of batches to prefetch; 0 disables prefetch")
    parser.add_argument("--stream-seed", type=int, default=42)

    args = parser.parse_args()


    if args.plot_basic:
        # All the basic plots!
        model = fromFolder(args.output)
        results = basic(model, args.signal_processes)

    else:
        model = fromYaml(args.yaml_config, args.output)
        train(
            model, args.output, args.percent, args.ebops, args.data_dir,
            args.class_config, args.test_data_dirs,
            stream_read_entries=args.stream_read_entries,
            stream_shuffle_jets=args.stream_shuffle_jets,
            stream_prefetch=args.stream_prefetch,
            stream_seed=args.stream_seed,
        )

