# Python
import gc
import json
import os
import shutil

import awkward as ak

# Third party
import numpy as np
import uproot
import yaml
from tqdm import tqdm

# Dataset configuration
from .config import EXTRA_FIELDS, FILTER_PATTERN, INPUT_TAG, N_PARTICLES

gc.set_threshold(0)


# >>>>>>>>>>>>>>>>>>>PRIVATE FUNCTIONS<<<<<<<<<<<<<<<<<<<<<<

def make_signal_data(
    infile,
    outdir,
    tag=INPUT_TAG,
    extras=EXTRA_FIELDS,
    n_parts=N_PARTICLES,
    ratio=1.0,
    step_size="100MB",
    tree="outnanoSC8/Jets",
    num_workers=8,
    classes=None,
    label_branch="sc8_label",
    pt_min=None,
    pt_max=None,
    event_tree="outnanoSC8/Events",
    events_per_chunk=1000,
    keep_labels=None,
):
    if ratio != 1.0 or events_per_chunk < 1:
        raise ValueError(
            "Use ratio=1.0 and events_per_chunk >= 1"
        )
    if (
        not classes
        or len(set(classes)) != len(classes)
        or not label_branch
    ):
        raise ValueError("Specify SC8 label_branch and unique classes")

    labels = {name: i for i, name in enumerate(classes)}
    keep_indices = _keep_label_indices(keep_labels, labels)

    files = sorted(set(
        item[0]
        for item in uproot.num_entries(_uproot_source(infile, tree))
    ))
    if not files:
        raise ValueError("No input files found")

    identifiers = ["run", "lumi", "event"]

    def event_keys(obj):
        missing = set(identifiers) - set(obj.keys())
        if missing:
            raise ValueError(
                f"Missing event identifiers: {sorted(missing)}"
            )

        arrays = obj.arrays(identifiers, library="ak")
        return np.rec.fromarrays(
            [
                ak.to_numpy(arrays[name]).astype(np.uint64)
                for name in identifiers
            ],
            names=",".join(identifiers),
        )

    os.makedirs(outdir, exist_ok=False)
    with open(os.path.join(outdir, "metadata.json"), "w") as stream:
        json.dump([], stream)

    filters = [FILTER_PATTERN, label_branch] + identifiers
    stored = identifiers + [
        "source_id", "jet_reject",
    ]
    chunk = 0

    for source_id, filename in enumerate(files):
        print(f"Processing complete events: {filename}", flush=True)

        # File reader for your mounted /eos/... paths.
        with uproot.open(
            filename,
            handler=uproot.source.file.MultithreadedFileSource,
            num_workers=num_workers,
        ) as root_file:
            jets = root_file[tree]
            event_records = root_file[event_tree]

            ek = event_keys(event_records)
            jk = event_keys(jets)

            if len(np.unique(ek)) != len(ek):
                raise ValueError(
                    f"Duplicate event identifiers in {filename}"
                )

            # Map each jet to its corresponding Events entry.
            order = np.argsort(ek)
            positions = np.searchsorted(ek[order], jk)

            if np.any(positions >= len(ek)):
                raise ValueError(
                    f"Jets without matching Events entries: {filename}"
                )
            if np.any(ek[order][positions] != jk):
                raise ValueError(
                    f"Jets without matching Events entries: {filename}"
                )

            jet_event = order[positions]

            if np.any(jet_event[1:] < jet_event[:-1]):
                raise ValueError(
                    f"Jets are not ordered like Events in {filename}"
                )

            for start in range(0, len(ek), events_per_chunk):
                stop = min(start + events_per_chunk, len(ek))

                # All jets belonging to this complete batch of events.
                first, last = np.searchsorted(
                    jet_event, [start, stop]
                )

                events = event_records.arrays(
                    entry_start=start,
                    entry_stop=stop,
                    library="ak",
                    how=dict,
                )
                data = jets.arrays(
                    entry_start=int(first),
                    entry_stop=int(last),
                    filter_name=filters,
                    library="ak",
                    how="zip",
                )

                data, _ = _split_sc8_labels(
                    data, label_branch, labels, drop_unknown=False
                )
                # Exactly the original reconstructed-jet cuts.
                jet_cut = (
                    (data["jet_pt_phys"] > 15)
                    & (np.abs(data["jet_eta_phys"]) < 2.4)
                    & (data["jet_reject"] == 0)
                )
                
                if pt_min is not None:
                    jet_cut = jet_cut & (data["jet_pt_phys"] >= pt_min)
                
                if pt_max is not None:
                    jet_cut = jet_cut & (data["jet_pt_phys"] < pt_max)
                
                data = data[jet_cut]
                
                # Exactly the original class selection: discard unconfigured labels.
                data, _ = _split_sc8_labels(
                    data,
                    label_branch,
                    labels,
                    drop_unknown=True,
                )
                if keep_indices is not None:
                    data = data[np.isin(ak.to_numpy(data["class_label"]), keep_indices)]
                
                # Compact the selected jet buffers before writing.
                data = ak.to_packed(data)
                
                data["source_id"] = np.full(
                    len(data), source_id, dtype=np.int64
                )
                
                # Keep ALL event records in this batch, including events whose
                # jets were all removed. These records are for physics analysis only.
                events["source_id"] = np.full(
                    stop - start, source_id, dtype=np.int64
                )

                _process_chunk(
                    data, tag, extras, n_parts, chunk, outdir,
                    additional_fields=stored,
                    events=events,
                )
                chunk += 1

    _save_dataset_metadata(outdir, labels, tag, extras)

    path = os.path.join(outdir, "variables.json")
    with open(path) as stream:
        variables = json.load(stream)

    variables.update(
        event_aware=True,
        source_files=files,
        keep_labels=keep_labels,
    )

    with open(path, "w") as stream:
        json.dump(variables, stream, indent=2)


def _add_response_vars(data):
    data['jet_ptUncorr_div_ptGen'] = ak.nan_to_num(
        data['jet_pt_phys'] / data['jet_genmatch_pt'], copy=True, nan=0.0, posinf=0.0, neginf=0.0
    )
    data['jet_ptCorr_div_ptGen'] = ak.nan_to_num(
        data['jet_pt_corr'] / data['jet_genmatch_pt'], copy=True, nan=0.0, posinf=0.0, neginf=0.0
    )
    data['jet_ptRaw_div_ptGen'] = ak.nan_to_num(
        data['jet_pt_raw'] / data['jet_genmatch_pt'], copy=True, nan=0.0, posinf=0.0, neginf=0.0
    )

def _keep_label_indices(keep_labels, class_labels):
    """Select a subset without changing the shared class-to-ID mapping."""
    if keep_labels is None:
        return None
    if not isinstance(keep_labels, (list, tuple)) or not keep_labels:
        raise ValueError("keep_labels must be a nonempty list of class names")
    if any(not isinstance(name, str) for name in keep_labels):
        raise ValueError("keep_labels must contain class names")
    missing = set(keep_labels) - set(class_labels)
    if missing:
        raise ValueError(f"keep_labels not present in classes: {sorted(missing)}")
    return [class_labels[name] for name in keep_labels]


def _split_sc8_labels(
    data, label_branch, class_labels, drop_unknown=True
):
    # One label per jet, stored as a singleton vector.
    counts = ak.num(data[label_branch], axis=1)
    if not bool(ak.all(counts == 1)):
        raise ValueError(
            f"Expected exactly one label per jet in {label_branch}"
        )

    labels = ak.to_list(data[label_branch][:, 0])

    indices = np.asarray(
        [class_labels.get(label, -1) for label in labels],
        dtype=np.int32,
    )

    if drop_unknown:
        keep = indices >= 0
        data = data[keep]
        indices = indices[keep]

    data["class_label"] = indices

    truth_pt = ak.nan_to_num(
        data["jet_genmatch_pt"],
        nan=0,
        posinf=0,
        neginf=0,
    )
    data["target_pt_phys"] = truth_pt

    data["target_pt"] = np.clip(
        ak.nan_to_num(
            truth_pt / data["jet_pt_phys"],
            nan=0,
            posinf=0,
            neginf=0,
        ),
        0.3,
        2,
    )

    return data, class_labels

def _split_flavor(data):
    """Split data by particle flavor and create the classification/pT targets."""
    genmatch_pt_base = data['jet_genmatch_pt'] > 0

    conditions = {
        "b": (
            genmatch_pt_base
            & (data['jet_muflav'] == 0)
            & (data['jet_tauflav'] == 0)
            & (data['jet_elflav'] == 0)
            & (data['jet_genmatch_hflav'] == 5)
        ),
        "charm": (
            genmatch_pt_base
            & (data['jet_muflav'] == 0)
            & (data['jet_tauflav'] == 0)
            & (data['jet_elflav'] == 0)
            & (data['jet_genmatch_hflav'] == 4)
        ),
        "light": (
            genmatch_pt_base
            & (data['jet_muflav'] == 0)
            & (data['jet_tauflav'] == 0)
            & (data['jet_elflav'] == 0)
            & (data['jet_genmatch_hflav'] == 0)
            & (
                (abs(data['jet_genmatch_pflav']) == 0)
                | (abs(data['jet_genmatch_pflav']) == 1)
                | (abs(data['jet_genmatch_pflav']) == 2)
                | (abs(data['jet_genmatch_pflav']) == 3)
            )
        ),
        "gluon": (
            genmatch_pt_base
            & (data['jet_muflav'] == 0)
            & (data['jet_tauflav'] == 0)
            & (data['jet_elflav'] == 0)
            & (data['jet_genmatch_hflav'] == 0)
            & (data['jet_genmatch_pflav'] == 21)
        ),
        "taup": (
            genmatch_pt_base
            & (data['jet_muflav'] == 0)
            & (data['jet_tauflav'] == 1)
            & (data['jet_taucharge'] > 0)
            & (data['jet_elflav'] == 0)
        ),
        "taum": (
            genmatch_pt_base
            & (data['jet_muflav'] == 0)
            & (data['jet_tauflav'] == 1)
            & (data['jet_taucharge'] < 0)
            & (data['jet_elflav'] == 0)
        ),
        "muon": (
            genmatch_pt_base
            & (data['jet_muflav'] == 1)
            & (data['jet_tauflav'] == 0)
            & (data['jet_elflav'] == 0)
        ),
        "electron": (
            genmatch_pt_base
            & (data['jet_muflav'] == 0)
            & (data['jet_tauflav'] == 0)
            & (data['jet_elflav'] == 1)
        ),
    }

    class_labels = {label: idx for idx, label in enumerate(conditions)}
    data['class_label'] = ak.full_like(data['jet_genmatch_pt'], -1)
    for label, condition in conditions.items():
        data['class_label'] = ak.where(condition, class_labels[label], data['class_label'])

    hadrons = conditions["b"] | conditions["charm"] | conditions["light"] | conditions["gluon"]
    leptons = conditions["taup"] | conditions["taum"] | conditions["muon"] | conditions["electron"]

    hadron_pt_ratio = ak.nan_to_num(data["jet_genmatch_pt"] / data["jet_pt_phys"], nan=0, posinf=0, neginf=0)
    lepton_pt_ratio = ak.nan_to_num(
        data["jet_genmatch_lep_vis_pt"] / data["jet_pt_phys"], nan=0, posinf=0, neginf=0
    )
    hadron_pt = ak.nan_to_num(data["jet_genmatch_pt"], nan=0, posinf=0, neginf=0)
    lepton_pt = ak.nan_to_num(data["jet_genmatch_lep_vis_pt"], nan=0, posinf=0, neginf=0)

    data['target_pt'] = np.clip(hadrons * hadron_pt_ratio + leptons * lepton_pt_ratio, 0.3, 2)
    data['target_pt_phys'] = hadrons * hadron_pt + leptons * lepton_pt

    jet_ptmin_gen = data['target_pt_phys'] > 5.0
    for key in conditions:
        conditions[key] = conditions[key] & jet_ptmin_gen

    split_data_sum = sum(sum(conditions[label]) for label in conditions)
    if split_data_sum != len(data[jet_ptmin_gen]):
        raise ValueError(
            f"Data splitting error: Total entries ({split_data_sum}) "
            f"do not match the filtered data length ({len(data[jet_ptmin_gen])})."
        )

    return data[jet_ptmin_gen], class_labels


def _get_puppicand_fields(tag):
    current_dir = os.path.dirname(__file__)
    puppicand_fields_path = os.path.join(current_dir, "puppicand_fields.yml")
    with open(puppicand_fields_path, "r") as file:
        puppicand_fields = yaml.safe_load(file)
    return puppicand_fields[tag]


def _pad_fill(array, target):
    """Pad an array to target length and fill missing values with zero."""
    return ak.fill_none(ak.pad_none(array, target, axis=1, clip=True), 0)


def _make_nn_inputs(data_split, tag, n_parts):
    features = _get_puppicand_fields(tag)
    puppicands = data_split["jet_puppicand"]
    available_fields = set(ak.fields(puppicands))
    inputs_list = []

    for field in features:
        if field not in available_fields:
            raise ValueError(
                f"Constituent feature {field!r} from tag {tag!r} was not found "
                f"inside jet_puppicand. Available fields: "
                f"{sorted(available_fields)}"
            )

        field_array = puppicands[field]
        padded_filled_array = _pad_fill(field_array, n_parts)
        inputs_list.append(padded_filled_array[:, :, np.newaxis])

    data_split["nn_inputs"] = ak.concatenate(inputs_list, axis=2)

def _save_chunk_metadata(metadata_file, chunk, entries, outfile):
    chunk_info = {"chunk": chunk, "entries": entries, "file": outfile}
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            content = f.read()
            metadata = json.loads(content) if content.strip() else []
    else:
        metadata = []
    metadata.append(chunk_info)
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)


def _save_dataset_metadata(outdir, class_labels, tag, extras):
    dataset_metadata_file = os.path.join(outdir, 'variables.json')
    metadata = {
        "outputs": class_labels,
        "inputs": _get_puppicand_fields(tag),
        "extras": _get_puppicand_fields(extras),
    }
    with open(dataset_metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)


def _process_chunk(
    data_split, tag, extras, n_parts, chunk, outdir,
    additional_fields=(), events=None,
):
    """Save jet data and its corresponding events in the same ROOT file."""
    if len(data_split) == 0 and events is None:
        return

    _make_nn_inputs(data_split, tag, n_parts)

    extra_features = _get_puppicand_fields(extras)
    save_fields = [
        "nn_inputs",
        "class_label",
        "target_pt",
        "target_pt_phys",
    ] + extra_features + list(additional_fields)

    filtered_data = {
        field: data_split[field]
        for field in save_fields
    }

    outfile = os.path.abspath(
        os.path.join(outdir, f"data_chunk_{chunk}.root")
    )

    with uproot.recreate(outfile) as f:
        f["data"] = filtered_data

        if events is not None:
            f["events"] = events

    metadata_file = os.path.join(outdir, "metadata.json")
    _save_chunk_metadata(
        metadata_file, chunk, len(data_split), outfile
    )

    del data_split, filtered_data, outfile
    gc.collect()


def _next_chunk(outdir):
    metadata_file = os.path.join(outdir, "metadata.json")
    if not os.path.exists(metadata_file):
        return 0
    with open(metadata_file, "r") as f:
        return len(json.load(f))


def _uproot_source(infile, tree):
    """ Build an Uproot source from one or more files/directories/patterns.

        Supports both data/sample_1/*.root and data/*/*.root paths 
    """

    if isinstance(infile, (str, bytes, os.PathLike)):
        inputs = [infile]
    else:
        inputs = infile
    sources = {}
    for item in inputs:
        item = os.fspath(item).rstrip("/")
        if os.path.isdir(item):
            item = os.path.join(item, "*.root")
        sources[item] = tree
    return sources

# >>>>>>FUNCTIONS THAT SHOULD BE USED EXTERNALLY!<<<<<<<


def extract_array(tree, field, entry_stop):
    """Extract an array from the tree with a limit on the number of entries."""
    return tree[field].array(entry_stop=entry_stop)


def extract_nn_inputs(data, input_vars, n_parts=16, n_entries=None):
    """Extract NN inputs based on the input_vars list."""
    inputs_list = []
    for field in input_vars:
        field_array = extract_array(data, f"jet_puppicand_{field}", n_entries)
        padded_filled_array = _pad_fill(field_array, n_parts)
        inputs_list.append(padded_filled_array[:, :, np.newaxis])
    return ak.concatenate(inputs_list, axis=2)


def group_id_values(event_id, *arrays, num_elements=2):
    """Group values by event ID and remove groups with too few elements."""
    sorted_indices = ak.argsort(event_id)
    sorted_event_id = event_id[sorted_indices]
    _, counts = np.unique(sorted_event_id, return_counts=True)
    grouped_id = ak.unflatten(sorted_event_id, counts)
    grouped_arrays = [ak.unflatten(arr[sorted_indices], counts) for arr in arrays]
    mask = ak.num(grouped_id) >= num_elements
    filtered_grouped_arrays = [arr[mask] for arr in grouped_arrays]
    return grouped_id[mask], filtered_grouped_arrays


def to_ML(data, class_labels):
    """Convert data made by make_data into arrays ready for training."""
    X = np.asarray(data['nn_inputs'])
    labels = np.asarray(data['class_label'], dtype=int)
    num_classes = len(class_labels)
    y = np.eye(num_classes)[labels]
    pt_target = np.asarray(data['target_pt'])
    truth_pt = np.asarray(data['target_pt_phys'])
    reco_pt = np.asarray(data['jet_pt_phys'])
    return X, y, pt_target, truth_pt, reco_pt

    chunks_to_load = int(np.ceil((percentage / 100) * total_chunks))
    chunk_files = [metadata[i]["file"] for i in range(chunks_to_load)]
    data = uproot.concatenate(
        [f"{file}:data" for file in chunk_files], filter_name=fields, library="ak"
    )

    indices = np.arange(len(data))
    np.random.shuffle(indices)

    data_metadata_file = os.path.join(outdir, "variables.json")
    with open(data_metadata_file, "r") as f:
        variables = json.load(f)
        class_labels = variables['outputs']
        input_vars = variables['inputs']
        extra_vars = variables['extras']

    return data, 0, class_labels, input_vars, extra_vars


def load_data(outdir, percentage, test_ratio=0.0, fields=None):
    """
    Load a specified percentage of the dataset using uproot.concatenate.

    Parameters:
        outdir (str): The output directory containing the data chunks.
        percentage (float): The percentage of TOTAL data to load (0-100).
        test_ratio (float): how much of the total data would be used for testing (0-1)
        fields (list, optional): Specific fields to load. If None, load all fields.

    Returns:
        awkward.Array: Concatenated data arrays from selected chunks.
    """

    print("Loading data from: ", outdir)
    print("Loading percentage: ", percentage)
    print("With test ratio of: ", test_ratio)

    # Load metadata to determine chunks to load
    metadata_file = os.path.join(outdir, "metadata.json")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    if not 0 < percentage <= 100:
        raise ValueError("percentage must be between 0 and 100")

    total_chunks = len(metadata)
    if total_chunks == 0:
        raise ValueError(f"No chunks were found in {metadata_file}")

    chunks_to_load = max(1, int(np.ceil((percentage / 100) * total_chunks)))

    # Collect the file paths for the chunks to load
    chunk_files = [metadata[i]["file"] for i in range(chunks_to_load)]

    # Use uproot.concatenate to load and combine data from multiple files
    # data = uproot.concatenate(
    #     [f"{filename}:data" for filename in chunk_files],
    #     filter_name=fields,
    #     library="ak",
    # )

    print("Uproot version:", uproot.__version__, flush=True)

    parts = []
    
    for filename in chunk_files:
        print(f"Reading: {filename}", flush=True)
    
        with uproot.open(
            filename,
            object_cache=None,
            array_cache=None,
        ) as root_file:
            part = root_file["data"].arrays(
                filter_name=fields,
                library="ak",
            )
    
        parts.append(part)
    
    data = ak.concatenate(parts, axis=0)
    del parts

    # Load corresponding metadata for classlabels/input variables
    data_metadata_file = os.path.join(outdir, "variables.json")
    with open(data_metadata_file, "r") as f:
        variables = json.load(f)
        class_labels = variables['outputs']
        input_vars = variables['inputs']
        extra_vars = variables['extras']

    return data, 0, class_labels, input_vars, extra_vars


def make_data(
    infile,
    outdir="training_data/",
    tag=INPUT_TAG,
    extras=EXTRA_FIELDS,
    n_parts=N_PARTICLES,
    ratio=1.0,
    step_size="100MB",
    tree="outnano/Jets",
    num_workers=8,
    classes=None,
    label_branch=None,
    pt_min=None,
    pt_max=None,
):
    class_labels = {
        name: index for index, name in enumerate(classes or [])
    }
    # Legacy paths/lists still work. New groups may select labels independently.
    if isinstance(infile, dict):
        samples = [infile]
    elif isinstance(infile, (list, tuple)) and any(isinstance(item, dict) for item in infile):
        if not all(isinstance(item, dict) for item in infile):
            raise ValueError("Do not mix input groups and plain paths in input")
        samples = infile
    else:
        samples = [{"input": infile}]

    sources = []
    for sample in samples:
        if sample.get("keep_labels") is not None and label_branch is None:
            raise ValueError("keep_labels requires a label_branch, e.g. sc8_label")
        keep_indices = _keep_label_indices(sample.get("keep_labels"), class_labels)
        sources.append((_uproot_source(sample["input"], tree), keep_indices))

    num_entries = sum(
        item[-1]
        for source, _ in sources
        for item in uproot.num_entries(source)
    )

    if num_entries == 0:
        raise ValueError(f"No entries found in {infile}")

    if os.path.exists(outdir):
        answer = input(
            f"The directory '{outdir}' already exists. "
            "Delete it and continue? [y/n]: "
        )
        if answer.lower() != "y":
            print("Exiting without making changes.")
            return
        shutil.rmtree(outdir)

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "metadata.json"), "w") as stream:
        json.dump([], stream)

    filters = [FILTER_PATTERN]
    if label_branch is not None:
        filters.append(label_branch)

    iterator = (
        (data, keep_indices)
        for source, keep_indices in sources
        for data in uproot.iterate(
            source,
            filter_name=filters,
            how="zip",
            step_size=step_size,
            num_workers=num_workers,
        )
    )

    chunk = 0
    entries_done = 0
    print("Output directory:", outdir)
    print("Input entries:", num_entries)

    with tqdm(total=num_entries, unit="jets") as pbar:
        for data, keep_indices in iterator:
            entries_done += len(data)
            pbar.update(len(data))
            pbar.set_description(f"Processing chunk {chunk}")

            jet_cut = (
                (data["jet_pt_phys"] > 15)
                & (np.abs(data["jet_eta_phys"]) < 2.4)
                & (data["jet_reject"] == 0)
            )

            if pt_min is not None:
                jet_cut = jet_cut & (data["jet_pt_phys"] >= pt_min)
            if pt_max is not None:
                jet_cut = jet_cut & (data["jet_pt_phys"] < pt_max)

            data = data[jet_cut]

            if len(data):
                if label_branch is None:
                    data_split, class_labels = _split_flavor(data)
                else:
                    data_split, class_labels = _split_sc8_labels(
                        data, label_branch, class_labels
                    )

                if keep_indices is not None:
                    data_split = data_split[
                        np.isin(ak.to_numpy(data_split["class_label"]), keep_indices)
                    ]

                if len(data_split):
                    _process_chunk(
                        data_split, tag, extras, n_parts, chunk, outdir
                    )
                    chunk += 1

            if entries_done / num_entries >= ratio:
                break

    _save_dataset_metadata(outdir, class_labels, tag, extras)

def make_data_from_config(config_file, force=False):
    """Process every dataset folder from a YAML configuration."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    output_dir = os.path.expandvars(os.path.expanduser(config['output_dir']))
    train_dir = os.path.join(output_dir, "training_data")

    if os.path.exists(output_dir):
        if not force:
            raise FileExistsError(f"{output_dir} already exists; use --force to replace it")
        shutil.rmtree(output_dir)

    datasets = config.get('datasets', [])
    if not datasets:
        raise ValueError("No datasets were provided in the YAML config")

    for dataset_number, dataset in enumerate(datasets):
        print(f"Processing dataset: {dataset['name']} ({dataset['path']})")
        make_data(
            infile=dataset['path'],
            outdir=train_dir,
            tag=config.get('tag', INPUT_TAG),
            extras=config.get('extras', EXTRA_FIELDS),
            n_parts=int(config.get('n_parts', N_PARTICLES)),
            ratio=float(config.get('ratio', 1.0)),
            step_size=config.get('step_size', '100MB'),
            tree=config.get('tree', 'outnano/Jets'),
            num_workers=int(config.get('num_workers', 8)),
            append=True,
        )

