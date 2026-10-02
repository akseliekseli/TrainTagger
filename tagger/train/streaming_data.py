"""Bounded ROOT reads and tf.data batches for TrainTagger's two-output models."""

import json
import itertools
import math
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
import yaml


def read_sample(directory, percentage=100):
    directory = Path(directory)
    with (directory / "variables.json").open() as stream:
        variables = json.load(stream)
    with (directory / "metadata.json").open() as stream:
        metadata = json.load(stream)
    if not 0 < percentage <= 100 or not metadata:
        raise ValueError(f"Invalid percentage or empty metadata: {directory}")
    metadata = metadata[:max(1, math.ceil(len(metadata) * percentage / 100))]
    files = []
    for entry in metadata:
        path = Path(entry["file"])
        path = path if path.is_absolute() else directory / path
        files.append({"path": str(path), "entries": int(entry["entries"])})
    if len({os.path.realpath(item["path"]) for item in files}) != len(files):
        raise ValueError(f"Duplicate files in {directory}/metadata.json")
    return {"directory": str(directory), "variables": variables, "files": files}


def read_class_groups(path, source_labels):
    if path is None:
        groups = {name: [name] for name in source_labels}
    else:
        with open(path) as stream:
            groups = yaml.safe_load(stream)["classes"]
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Class configuration must contain a nonempty 'classes' mapping")
    used = set()
    for name, labels in groups.items():
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"Class {name} must contain a nonempty list")
        if len(set(labels)) != len(labels) or used.intersection(labels):
            raise ValueError(f"Repeated source labels in class {name}")
        missing = set(labels) - set(source_labels)
        if missing:
            raise ValueError(f"Unknown source labels in {name}: {sorted(missing)}")
        used.update(labels)
    return groups


def remap_labels(labels, groups, source_labels):
    labels = ak.to_numpy(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("class_label must contain one integer per jet")
    result = np.full(len(labels), -1, dtype=np.int32)
    for index, names in enumerate(groups.values()):
        ids = [source_labels[name] for name in names]
        result[np.isin(labels, ids)] = index
    return result


def read_blocks(file_info, fields, read_entries):
    path = file_info["path"]
    try:
        with uproot.open(path, object_cache=None, array_cache=None) as root:
            tree = root["data"]
            if tree.num_entries != file_info["entries"]:
                raise ValueError("Jet count differs from metadata")
            missing = set(fields) - set(tree.keys())
            if missing:
                raise ValueError(f"Missing branches: {sorted(missing)}")
            yield from tree.iterate(filter_name=fields, step_size=read_entries,
                                    library="ak", how=dict)
    except Exception as exc:
        raise RuntimeError(f"Cannot read {path}: {exc}") from exc


class HistogramWeights:
    """Dataset-wide equivalent of the supplied train_weights(), not per-chunk weights."""

    def __init__(self, method, n_classes):
        self.method = method
        self.fields = []
        self.edges = []
        if method in ("ptref", "onlyclass", "ptmassref"):
            self.fields.append("jet_pt_phys")
            pt_bins = [15, 17, 19, 22, 25, 30, 35, 40, 45, 50, 60, 76, 97, 122, 154, np.inf]
            if method == "onlyclass":
                pt_bins = [0, np.inf]
            elif method == "ptmassref":
                pt_bins = [15, 30, 50, 80, 120, 200, 350, np.inf]
            self.edges.append(np.asarray(pt_bins))
        if method in ("massref", "ptmassref"):
            self.fields.append("jet_mass_phys")
            self.edges.append(np.array([0, 40, 70, 90, 110, 130, 150, 180, 220, np.inf]))
        if method not in ("none", "onlyclass", "ptref", "massref", "ptmassref"):
            raise ValueError(f"Unsupported weight_method: {method}")
        self.bin_shape = tuple(len(edges)-1 for edges in self.edges)
        self.n_bins = math.prod(self.bin_shape)
        self.counts = np.zeros((n_classes, self.n_bins), dtype=np.int64)
        self.assigned_counts = np.zeros_like(self.counts)

    def bin_indices(self, arrays, *, for_histogram):
        n = len(arrays["class_label"])
        indices = np.zeros(n, dtype=np.int64)
        valid = np.ones(n, dtype=bool)
        for field, edges in zip(self.fields, self.edges):
            values = ak.to_numpy(arrays[field])
            if values.ndim != 1 or len(values) != n:
                raise ValueError(f"{field} must contain one value per jet")
            nbins = len(edges)-1
            index = np.searchsorted(edges, values, side="right") - 1
            if for_histogram:
                # np.histogram includes its final right edge (+inf here).
                index = np.where(values == edges[-1], nbins-1, index)
            elif self.method in ("ptref", "onlyclass"):
                # Retain the supplied function's underflow/overflow handling.
                index = np.where(index < 0, nbins-1, np.minimum(index, nbins-1))
            valid &= (index >= 0) & (index < nbins)
            if self.method == "massref":
                valid &= np.isfinite(values)
            indices = indices * nbins + np.clip(index, 0, nbins-1)
        return indices, valid

    def add(self, arrays, labels):
        if self.method == "none":
            return
        for histogram, destination in ((True, self.counts), (False, self.assigned_counts)):
            bins, valid = self.bin_indices(arrays, for_histogram=histogram)
            valid &= labels >= 0
            flat_indices = labels[valid] * self.n_bins + bins[valid]
            destination += np.bincount(flat_indices, minlength=destination.size).reshape(destination.shape)

    def finalize(self, active, n_samples):
        if self.method == "none":
            self.table = None
            return
        counts = self.counts[active]
        if np.any(counts.sum(axis=1) == 0):
            raise ValueError("An active class has no jets inside the weighting bins")
        if self.method == "onlyclass":
            target = counts[0]
        elif self.method == "ptref":
            target = np.full(self.n_bins, counts.min())
        else:
            target = counts.min(axis=0)
        table = np.divide(target[None, :], counts, out=np.zeros(counts.shape), where=counts > 0)
        if self.method == "massref":
            table = table.astype(np.float32)  # Match the original massref rounding.
        mean_weight = np.sum(table.astype(np.float64) * self.assigned_counts[active]) / n_samples
        if not np.isfinite(mean_weight) or mean_weight <= 0:
            raise ValueError(
                "Weighting gives zero/nonfinite total weight. In particular, ptref uses the "
                "minimum over ALL class/pT bins; one empty bin makes all its weights zero. "
                "Review the bins or weighting method, rather than train on NaNs."
            )
        self.table = (table / mean_weight).astype(np.float32)

    def evaluate(self, arrays, labels):
        if self.table is None:
            return np.ones(len(labels), dtype=np.float32)
        bins, valid = self.bin_indices(arrays, for_histogram=False)
        weights = np.zeros(len(labels), dtype=np.float32)
        weights[valid] = self.table[labels[valid], bins[valid]]
        return weights


class RootTrainingData:
    def __init__(self, directory, class_config, percentage, weight_method, read_entries=32768):
        if not isinstance(read_entries, int) or read_entries < 1:
            raise ValueError("read_entries must be a positive integer")
        self.sample = read_sample(directory, percentage)
        self.variables = self.sample["variables"]
        self.groups = read_class_groups(class_config, self.variables["outputs"])
        self.read_entries = read_entries
        self.weights = HistogramWeights(weight_method, len(self.groups))
        class_counts = np.zeros(len(self.groups), dtype=np.int64)
        self.input_shape = self.target_shape = None
        for i, file_info in enumerate(self.sample["files"], 1):
            print(f"Scan training [{i}/{len(self.sample['files'])}]: {file_info['path']}", flush=True)
            file_counts = np.zeros_like(class_counts)
            for arrays in read_blocks(file_info, ["class_label", *self.weights.fields], read_entries):
                labels = remap_labels(arrays["class_label"], self.groups, self.variables["outputs"])
                file_counts += np.bincount(labels[labels >= 0], minlength=len(self.groups))
                self.weights.add(arrays, labels)
            file_info["selected"] = int(file_counts.sum())
            class_counts += file_counts
            if self.input_shape is None and file_info["selected"]:
                with uproot.open(file_info["path"], object_cache=None, array_cache=None) as root:
                    first = root["data"].arrays(filter_name=["nn_inputs", "target_pt"],
                                                 entry_stop=1, library="ak", how=dict)
                self.input_shape = ak.to_numpy(first["nn_inputs"]).shape[1:]
                self.target_shape = ak.to_numpy(first["target_pt"]).shape[1:]
        active = np.flatnonzero(class_counts)
        for name, count in zip(self.groups, class_counts):
            print(f"{name}: {count} selected training-directory jets", flush=True)
        if len(active) < 2:
            raise ValueError("Fewer than two configured classes have training entries")
        self.groups = {name: labels for i, (name, labels) in enumerate(self.groups.items()) if class_counts[i]}
        self.class_labels = {name: i for i, name in enumerate(self.groups)}
        self.n_samples = int(class_counts.sum())
        self.weights.finalize(active, self.n_samples)

    def blocks(self, start=0, stop=None, *, shuffle_seed=None):
        stop = self.n_samples if stop is None else stop
        if not 0 <= start < stop <= self.n_samples:
            raise ValueError("Invalid or empty training/validation range")
        fields = list(dict.fromkeys(["nn_inputs", "class_label", "target_pt", *self.weights.fields]))
        files = self.sample["files"]
        file_starts = np.r_[0, np.cumsum([item["selected"] for item in files])]
        file_order = np.flatnonzero((file_starts[1:] > start) & (file_starts[:-1] < stop))
        if shuffle_seed is not None:
            np.random.default_rng(shuffle_seed).shuffle(file_order)
        for file_index in file_order:
            file_info = files[file_index]
            offset = int(file_starts[file_index])
            file_end = offset + file_info["selected"]
            for arrays in read_blocks(file_info, fields, self.read_entries):
                labels = remap_labels(arrays["class_label"], self.groups, self.variables["outputs"])
                selected = np.flatnonzero(labels >= 0)
                end = offset + len(selected)
                low, high = max(0, start-offset), min(len(selected), stop-offset)
                if high > low:
                    selected = selected[low:high]
                    arrays = {name: values[selected] for name, values in arrays.items()}
                    labels = labels[selected]
                    X = ak.to_numpy(arrays["nn_inputs"]).astype(np.float32, copy=False)
                    target_pt = ak.to_numpy(arrays["target_pt"]).astype(np.float32, copy=False)
                    if X.shape[1:] != self.input_shape or target_pt.shape[1:] != self.target_shape:
                        raise ValueError(f"Inconsistent input/target shape in {file_info['path']}")
                    if not np.isfinite(X).all() or not np.isfinite(target_pt).all():
                        raise ValueError(f"Nonfinite model input/target in {file_info['path']}")
                    y = np.eye(len(self.class_labels), dtype=np.float32)[labels]
                    weights = self.weights.evaluate(arrays, labels)
                    yield {"model_input": X}, (y, target_pt), (weights, weights)
                offset = end
            if offset != file_end:
                raise ValueError(f"Selected jet count changed since scan: {file_info['path']}")

    def dataset(self, start, stop, batch_size, *, shuffle_jets=0, seed=42, prefetch=1):
        import tensorflow as tf
        if batch_size < 1 or shuffle_jets < 0 or prefetch < 0:
            raise ValueError("Invalid batch/buffer size")
        signature = (
            {"model_input": tf.TensorSpec((None, *self.input_shape), tf.float32)},
            (tf.TensorSpec((None, len(self.class_labels)), tf.float32),
             tf.TensorSpec((None, *self.target_shape), tf.float32)),
            (tf.TensorSpec((None,), tf.float32), tf.TensorSpec((None,), tf.float32)),
        )
        epochs = itertools.count()
        def generate():
            shuffle_seed = seed + next(epochs) if shuffle_jets else None
            yield from self.blocks(start, stop, shuffle_seed=shuffle_seed)
        dataset = tf.data.Dataset.from_generator(generate, output_signature=signature)
        dataset = dataset.unbatch().apply(tf.data.experimental.assert_cardinality(stop-start))
        if shuffle_jets:
            dataset = dataset.shuffle(min(shuffle_jets, stop-start), seed=seed, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        options.autotune.enabled = False
        dataset = dataset.with_options(options)
        return dataset.prefetch(prefetch) if prefetch else dataset


def check_test_samples(directories, training):
    samples = [read_sample(directory) for directory in directories]
    seen = {os.path.realpath(item["path"]) for item in training.sample["files"]}
    for sample in samples:
        for field in ("outputs", "inputs", "extras"):
            if sample["variables"][field] != training.variables[field]:
                raise ValueError(f"Training/testing {field} mappings differ: {sample['directory']}")
        for file_info in sample["files"]:
            path = os.path.realpath(file_info["path"])
            if path in seen:
                raise ValueError(f"Repeated or overlapping train/test file: {path}")
            seen.add(path)
    return samples


def save_test_data_streaming(out_dir, samples, training):
    """Write the same four .npy files as save_test_data, without full test arrays."""
    fields = ["nn_inputs", "class_label", "target_pt_phys", "jet_pt_phys"]
    total = 0
    for sample in samples:
        for file_info in sample["files"]:
            for arrays in read_blocks(file_info, ["class_label"], training.read_entries):
                labels = remap_labels(arrays["class_label"], training.groups, sample["variables"]["outputs"])
                total += np.count_nonzero(labels >= 0)
    if not total:
        raise ValueError("No test jets remain after applying the training class groups")
    shapes = {
        "X_test": (total, *training.input_shape),
        "y_test": (total, len(training.class_labels)),
        "truth_pt_test": (total,), "reco_pt_test": (total,),
    }
    destination = Path(out_dir) / "testing_data"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stream-test-", dir=destination) as temporary:
        with ExitStack() as stack:
            files = {name: stack.enter_context(open(Path(temporary) / f"{name}.npy", "wb")) for name in shapes}
            for name, stream in files.items():
                np.lib.format.write_array_header_2_0(stream, {
                    "descr": np.lib.format.dtype_to_descr(np.dtype("float32")),
                    "fortran_order": False, "shape": shapes[name],
                })
            written = 0
            for sample in samples:
                for file_info in sample["files"]:
                    print(f"Export test: {file_info['path']}", flush=True)
                    for arrays in read_blocks(file_info, fields, training.read_entries):
                        labels = remap_labels(arrays["class_label"], training.groups, sample["variables"]["outputs"])
                        keep = labels >= 0
                        values = {
                            "X_test": ak.to_numpy(arrays["nn_inputs"][keep]),
                            "y_test": np.eye(len(training.class_labels), dtype=np.float32)[labels[keep]],
                            "truth_pt_test": ak.to_numpy(arrays["target_pt_phys"][keep]),
                            "reco_pt_test": ak.to_numpy(arrays["jet_pt_phys"][keep]),
                        }
                        for name, value in values.items():
                            if value.shape[1:] != shapes[name][1:]:
                                raise ValueError(f"Unexpected test shape: {name} {value.shape}")
                            np.ascontiguousarray(value, dtype=np.float32).tofile(files[name])
                        written += int(keep.sum())
            if written != total:
                raise ValueError("Test jet count changed during export")
        for name in shapes:
            os.replace(Path(temporary) / f"{name}.npy", destination / f"{name}.npy")
    print(f"Saved {total} test jets to {destination}", flush=True)

