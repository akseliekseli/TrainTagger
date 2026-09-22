#!/usr/bin/env python3

import argparse
import json
import random
import shutil
from pathlib import Path


def write_metadata(directory, records):
    metadata_path = directory / "metadata.json"

    with metadata_path.open("w") as stream:
        json.dump(records, stream, indent=4)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Split processed TrainTagger chunks into training and testing directories"
    )
    parser.add_argument("directory", help="Directory currently containing data_chunk_*.root")
    parser.add_argument("--test-fraction", default=0.1, type=float)
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files; without this option, only show the planned split",
    )
    args = parser.parse_args()

    base_directory = Path(args.directory).resolve()
    training_directory = base_directory / "training_data"
    testing_directory = base_directory / "testing_data"

    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1")

    metadata_path = base_directory / "metadata.json"
    variables_path = base_directory / "variables.json"

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    if not variables_path.exists():
        raise FileNotFoundError(variables_path)

    chunks = sorted(base_directory.glob("data_chunk_*.root"))

    if len(chunks) < 2:
        raise ValueError(
            f"Expected at least two data_chunk_*.root files in {base_directory}"
        )

    if training_directory.exists() or testing_directory.exists():
        raise FileExistsError(
            "training_data or testing_data already exists; "
            "remove or rename them before running this script"
        )

    with metadata_path.open() as stream:
        original_metadata = json.load(stream)

    metadata_by_filename = {
        Path(record["file"]).name: record
        for record in original_metadata
    }

    missing_metadata = [
        chunk.name
        for chunk in chunks
        if chunk.name not in metadata_by_filename
    ]

    if missing_metadata:
        raise ValueError(
            f"Chunks missing from metadata.json: {missing_metadata[:10]}"
        )

    shuffled_chunks = chunks.copy()
    random.Random(args.seed).shuffle(shuffled_chunks)

    number_testing = max(
        1,
        round(len(shuffled_chunks) * args.test_fraction),
    )

    testing_chunks = sorted(shuffled_chunks[:number_testing])
    training_chunks = sorted(shuffled_chunks[number_testing:])

    print("Total chunks:", len(chunks))
    print("Training chunks:", len(training_chunks))
    print("Testing chunks:", len(testing_chunks))
    print("Random seed:", args.seed)

    if not args.apply:
        print()
        print("Dry run only. Re-run with --apply to move the files.")
        return

    training_directory.mkdir()
    testing_directory.mkdir()

    # Both datasets use exactly the same variables and class definitions.
    shutil.copy2(
        variables_path,
        training_directory / "variables.json",
    )
    shutil.copy2(
        variables_path,
        testing_directory / "variables.json",
    )

    split_metadata = {
        "training": [],
        "testing": [],
    }

    for group_name, destination, selected_chunks in (
        ("training", training_directory, training_chunks),
        ("testing", testing_directory, testing_chunks),
    ):
        for source_path in selected_chunks:
            destination_path = destination / source_path.name
            shutil.move(source_path, destination_path)

            record = dict(metadata_by_filename[source_path.name])
            record["file"] = str(destination_path)
            split_metadata[group_name].append(record)

    write_metadata(
        training_directory,
        split_metadata["training"],
    )
    write_metadata(
        testing_directory,
        split_metadata["testing"],
    )

    manifest = {
        "random_seed": args.seed,
        "test_fraction": args.test_fraction,
        "training_files": [
            record["file"]
            for record in split_metadata["training"]
        ],
        "testing_files": [
            record["file"]
            for record in split_metadata["testing"]
        ],
    }

    with (base_directory / "split_manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=4)
        stream.write("\n")

    # Preserve the original metadata as a backup.
    metadata_path.rename(
        base_directory / "metadata_before_split.json"
    )

    print()
    print("Split completed:")
    print("Training:", training_directory)
    print("Testing:", testing_directory)


if __name__ == "__main__":
    main()
