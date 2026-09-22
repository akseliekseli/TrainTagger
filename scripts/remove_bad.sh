/eos/home-a/asuutari/conda-envs/tagger/bin/python <<'PY'
import gc
import json
import os
import shutil
from datetime import datetime

import uproot

base = "/eos/home-a/asuutari/FastPUPPI/XtoHH-qcd"
quarantine = os.path.join(base, "bad_chunks")
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for subset in ("training_data", "testing_data"):
    dataset_dir = os.path.join(base, subset)
    metadata_path = os.path.join(dataset_dir, "metadata.json")

    with open(metadata_path) as handle:
        metadata = json.load(handle)

    valid = []
    bad = []

    print(f"\nChecking {subset}: {len(metadata)} chunks")

    for number, entry in enumerate(metadata, start=1):
        path = entry["file"]

        try:
            with uproot.open(path) as root_file:
                tree = root_file["data"]

                if tree.num_entries == 0:
                    raise ValueError("data tree has zero entries")

                # Read and decompress every field and every page.
                for arrays in tree.iterate(
                    step_size="50 MB",
                    library="ak",
                ):
                    del arrays

        except Exception as error:
            print(f"\nBAD: {path}")
            print(f"     {type(error).__name__}: {error}")
            bad.append((entry, path))
        else:
            valid.append(entry)

        gc.collect()

        if number % 100 == 0 or number == len(metadata):
            print(f"Checked {number}/{len(metadata)}")

    if not bad:
        print(f"{subset}: all {len(valid)} chunks are valid")
        continue

    backup = f"{metadata_path}.before_full_scan_{stamp}"
    shutil.copy2(metadata_path, backup)

    destination_dir = os.path.join(quarantine, subset)
    os.makedirs(destination_dir, exist_ok=True)

    for _, path in bad:
        if os.path.exists(path):
            destination = os.path.join(
                destination_dir,
                f"{stamp}_{os.path.basename(path)}",
            )
            shutil.move(path, destination)
            print(f"Moved {path} -> {destination}")

    with open(metadata_path, "w") as handle:
        json.dump(valid, handle, indent=4)

    print(f"{subset}: retained {len(valid)} valid chunks")
    print(f"{subset}: quarantined {len(bad)} bad chunks")
    print(f"Metadata backup: {backup}")
PY
