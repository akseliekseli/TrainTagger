import os
from argparse import ArgumentParser

import yaml

from tagger.data.tools import make_data


COLLECTIONS = {
    "sc4": {
        "tree": "outnano/Jets",
        "label_branch": None,
    },
    "sc8": {
        "tree": "outnanoSC8/Jets",
        "label_branch": "sc8_label",
    },
}


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "config",
        help="Dataset YAML configuration",
    )

    args = parser.parse_args()

    with open(args.config) as stream:
        config = yaml.safe_load(stream)

    collection = config["jet_collection"]

    if collection not in COLLECTIONS:
        raise ValueError(
            "jet_collection must be 'sc4' or 'sc8'"
        )

    collection_config = COLLECTIONS[collection]

    make_data(
        infile=config["input"],
        outdir=config["output"],
        step_size=config.get("step_size", "100MB"),
        extras=config.get("extras", "extra_fields"),
        ratio=config.get("ratio", 1.0),
        tree=collection_config["tree"],
        num_workers=config.get("num_workers", 8),
        label_branch=collection_config[
            "label_branch"
        ],
        label_classes=config.get("classes"),
        overwrite=config.get("overwrite", False),
    )

    # Process optional signal samples after the main dataset.
    for signal in config.get(
        "signal_processes",
        [],
    ):
        signal_output = signal["output"]

        if (
            os.path.exists(signal_output)
            and not config.get("overwrite", False)
        ):
            print(
                "Signal output exists, skipping: "
                f"{signal_output}"
            )
            continue

        make_data(
            infile=signal["input"],
            outdir=signal_output,
            step_size=config.get(
                "step_size",
                "100MB",
            ),
            extras=config.get(
                "extras",
                "extra_fields",
            ),
            ratio=config.get("ratio", 1.0),
            tree=collection_config["tree"],
            num_workers=config.get(
                "num_workers",
                8,
            ),
            label_branch=collection_config[
                "label_branch"
            ],
            label_classes=config.get("classes"),
            overwrite=config.get(
                "overwrite",
                False,
            ),
        )
