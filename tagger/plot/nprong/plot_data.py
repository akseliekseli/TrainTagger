"""Read each SC8 chunk once and return compact data shared by physics plots."""

import json
from pathlib import Path
from time import perf_counter

import awkward as ak
import numpy as np
import uproot

KEYS = ("source_id", "run", "lumi", "event")
GEN_FIELDS = (
    "GenPart_eta", "GenPart_phi", "GenPart_mass", "GenPart_pdgId",
    "GenJetAK8_eta", "GenJetAK8_phi",
)


def _read(tree, fields):
    missing = set(fields) - set(tree.keys())
    if missing:
        raise ValueError(f"Missing branches in {tree.name}: {sorted(missing)}")
    return tree.arrays(filter_name=list(fields), library="ak", how=dict)


def _keys(arrays):
    return list(zip(*(ak.to_list(arrays[name]) for name in KEYS)))


def _groups(jets, events):
    event_keys = _keys(events)
    lookup = {key: index for index, key in enumerate(event_keys)}
    if len(lookup) != len(event_keys):
        raise ValueError("Duplicate event keys within chunk")
    groups = [[] for _ in event_keys]
    event_index = np.empty(len(jets["jet_pt_phys"]), dtype=np.int64)
    for jet_index, key in enumerate(_keys(jets)):
        if key not in lookup:
            raise ValueError(f"Jet without an events row: {key}")
        event_index[jet_index] = lookup[key]
        groups[lookup[key]].append(jet_index)
    return groups, event_index


def _dr2(first_eta, first_phi, second_eta, second_phi):
    deta = first_eta[:, None] - second_eta[None, :]
    dphi = first_phi[:, None] - second_phi[None, :]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    return deta**2 + dphi**2


def _match_signal(jets, events, groups):
    jet_eta = ak.to_numpy(jets["jet_eta_phys"])
    jet_phi = ak.to_numpy(jets["jet_phi_phys"])
    if not np.isfinite(jet_eta).all() or not np.isfinite(jet_phi).all():
        raise ValueError("Nonfinite jet eta/phi")
    matched_mass = np.full(len(jet_eta), np.nan)
    # Convert once per chunk, rather than slicing Awkward arrays in every event.
    generator = {}
    for name in GEN_FIELDS:
        lengths = ak.to_numpy(ak.num(events[name], axis=1))
        offsets = np.r_[0, np.cumsum(lengths)]
        generator[name] = (ak.to_numpy(ak.flatten(events[name], axis=1)), offsets)
    for event_index, jet_indices in enumerate(groups):
        if not jet_indices:
            continue
        gen = {
            name: values[offsets[event_index]:offsets[event_index+1]]
            for name, (values, offsets) in generator.items()
        }
        particle_lengths = {len(v) for name, v in gen.items() if name.startswith("GenPart_")}
        if len(particle_lengths) != 1 or len(gen["GenJetAK8_eta"]) != len(gen["GenJetAK8_phi"]):
            raise ValueError("Inconsistent generator arrays within an event")
        if any(not np.isfinite(values).all() for values in gen.values()):
            raise ValueError("Nonfinite generator value")
        is_higgs = np.abs(gen["GenPart_pdgId"]) == 25
        if not is_higgs.any() or not len(gen["GenJetAK8_eta"]):
            continue
        jet_indices = np.asarray(jet_indices)
        dr_lg = _dr2(jet_eta[jet_indices], jet_phi[jet_indices],
                     gen["GenJetAK8_eta"], gen["GenJetAK8_phi"])
        dr_gh = _dr2(gen["GenJetAK8_eta"], gen["GenJetAK8_phi"],
                     gen["GenPart_eta"][is_higgs], gen["GenPart_phi"][is_higgs])
        closest_genjet = np.argmin(dr_lg, axis=1)
        closest_higgs = np.argmin(dr_gh, axis=1)
        matched = dr_lg[np.arange(len(jet_indices)), closest_genjet] < 0.3**2
        matched &= dr_gh[closest_genjet, closest_higgs[closest_genjet]] < 0.8**2
        masses = gen["GenPart_mass"][is_higgs][closest_higgs[closest_genjet]]
        matched_mass[jet_indices[matched]] = masses[matched]
    return matched_mass


def _predict(model, inputs, columns, batch_size):
    scores = np.empty((len(inputs), len(columns)), dtype=np.float64)
    tagger = np.empty(len(inputs), dtype=np.float64)
    for start in range(0, len(inputs), batch_size):
        stop = min(start + batch_size, len(inputs))
        outputs = model.predict(ak.to_numpy(inputs[start:stop]))
        probabilities = np.asarray(outputs[0] if isinstance(outputs, (tuple, list)) else outputs)
        if probabilities.shape != (stop-start, len(model.class_labels)):
            raise ValueError(f"Unexpected prediction shape: {probabilities.shape}")
        if (not np.isfinite(probabilities).all()
                or np.any(probabilities < -1e-6) or np.any(probabilities > 1+1e-6)
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4)):
            raise ValueError("model.predict() must return finite softmax probabilities")
        selected = probabilities[:, columns]
        scores[start:stop] = np.clip(selected, 0, 1)
        # Preserve the original sum in the model's output dtype, before casting.
        tagger[start:stop] = np.clip(selected.sum(axis=1), 0, 1)
    return scores, tagger


def _load_sample(model, directory, *, signal, score_classes, columns, batch_size, max_chunks):
    directory = Path(directory)
    with (directory / "variables.json").open() as stream:
        variables = json.load(stream)
    with (directory / "metadata.json").open() as stream:
        metadata = json.load(stream)
    if max_chunks is not None:
        metadata = metadata[:max_chunks]
    if not metadata:
        raise ValueError(f"No chunks listed in {directory}")
    if variables["inputs"] != list(model.input_vars):
        raise ValueError(f"Model input feature order differs: {directory}")
    source_labels = variables["outputs"]
    ids = list(source_labels.values())
    if any(type(index) is not int or index < 0 for index in ids) or len(set(ids)) != len(ids):
        raise ValueError("Dataset outputs must map names to unique nonnegative integer IDs")
    if signal and set(score_classes) - set(source_labels):
        raise ValueError(f"Signal labels must include {score_classes}")

    parts = {name: [] for name in ("pt", "scores", "tagger", "class_label", "event_index")}
    if signal:
        parts["higgs_mass"] = []
    n_events = 0
    seen_paths = set()
    for chunk_index, entry in enumerate(metadata, 1):
        path = Path(entry["file"])
        if not path.is_absolute():
            path = directory / path
        if path in seen_paths:
            raise ValueError(f"Duplicate chunk in metadata: {path}")
        seen_paths.add(path)
        print(f"Load {directory.name} [{chunk_index}/{len(metadata)}]: {path}", flush=True)
        try:
            started = perf_counter()
            with uproot.open(path, object_cache=None, array_cache=None) as root:
                fields = [*KEYS, "nn_inputs", "jet_pt_phys", "class_label"]
                if signal:
                    fields += ["jet_eta_phys", "jet_phi_phys"]
                jets = _read(root["data"], fields)
                events = _read(root["events"], (*KEYS, *GEN_FIELDS) if signal else KEYS)
            pt = ak.to_numpy(jets["jet_pt_phys"])
            if len(pt) != entry["entries"]:
                raise ValueError("Jet count differs from metadata")
            if pt.ndim != 1 or not np.isfinite(pt).all() or np.any(pt < 0):
                raise ValueError("Expected flat, finite, nonnegative jet pT")
            labels = ak.to_numpy(jets["class_label"])
            if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError("class_label must be a flat integer array")
            unknown = np.unique(labels[~np.isin(labels, ids + [-1])])
            if len(unknown):
                raise ValueError(f"Stored class IDs absent from variables.json: {unknown.tolist()}")
            print(f"  Read: {perf_counter()-started:.2f}s; {len(pt)} jets, {len(events['event'])} events", flush=True)
            started = perf_counter()
            groups, event_index = _groups(jets, events)
            if signal:
                parts["higgs_mass"].append(_match_signal(jets, events, groups))
            event_index += n_events
            n_events += len(events["event"])
            del groups, events
            print(f"  Join/match: {perf_counter()-started:.2f}s", flush=True)
            started = perf_counter()
            scores, tagger = _predict(model, jets["nn_inputs"], columns, batch_size)
            print(f"  Predict: {perf_counter()-started:.2f}s", flush=True)
            parts["pt"].append(pt.copy())
            parts["class_label"].append(labels.copy())
            parts["event_index"].append(event_index)
            parts["scores"].append(scores)
            parts["tagger"].append(tagger)
            del jets, pt, labels, event_index, scores, tagger
        except Exception as exc:
            raise RuntimeError(f"Plot input failed: {path}: {exc}") from exc

    sample = {}
    for name in list(parts):
        sample[name] = np.concatenate(parts.pop(name), axis=0)
    size_mb = sum(values.nbytes for values in sample.values()) / 1024**2
    sample.update(n_events=n_events, n_chunks=len(metadata), source_labels=source_labels,
                  directory=str(directory))
    print(f"Cached {directory.name}: {size_mb:.1f} MiB of plot arrays", flush=True)
    return sample


def load_plot_data(model, signal_dir, background_dir, *, score_classes,
                   batch_size=4096, max_chunks=None):
    """Return reusable NumPy plot data; no open ROOT files or constituent arrays.

    Memory scales with compact per-jet arrays, not the full ROOT contents.
    max_chunks selects the first N metadata entries separately for each sample.
    scores columns follow score_classes; tagger is their unnormalised sum.
    """
    if not isinstance(score_classes, (list, tuple)) or not score_classes:
        raise ValueError("score_classes must be a nonempty list or tuple of names")
    score_classes = tuple(score_classes)
    if any(not isinstance(name, str) for name in score_classes):
        raise ValueError("score_classes must contain class names")
    if len(set(score_classes)) != len(score_classes):
        raise ValueError("score_classes must not contain duplicates")
    missing = set(score_classes) - set(model.class_labels)
    if missing:
        raise ValueError(f"Model has no outputs for: {sorted(missing)}")
    if sorted(model.class_labels.values()) != list(range(len(model.class_labels))):
        raise ValueError("Model class indices must be contiguous from zero")
    columns = [model.class_labels[name] for name in score_classes]
    data = {"score_classes": score_classes, "max_chunks": max_chunks,
            "output_directory": str(model.output_directory)}
    for name, directory in (("background", background_dir), ("signal", signal_dir)):
        data[name] = _load_sample(model, directory, signal=name == "signal", columns=columns,
                                  score_classes=score_classes, batch_size=batch_size, max_chunks=max_chunks)
    return data

