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
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as stream:
        config = yaml.safe_load(stream)

    collection = COLLECTIONS[config["jet_collection"]]
    processing = config.get("processing", {})

    common = {
        "tree": collection["tree"],
        "label_branch": collection["label_branch"],
        "classes": config.get("classes"),
        "tag": processing.get(
            "input_tag", "baseline_hardware_inputs"
        ),
        "extras": processing.get("extras", "extra_fields"),
        "n_parts": processing.get("n_particles", 16),
        "step_size": processing.get("step_size", "100MB"),
        "num_workers": processing.get("num_workers", 8),
    }

    # Process the main training dataset
    make_data(
        infile=config["input"],
        outdir=config["output"],
        ratio=processing.get("ratio", 1.0),
        **common,
    )

    # Separate samples for evaluation.
    for signal in config.get("signal_processes", []):
        if os.path.exists(signal["output"]):
            print(
                f"Signal output exists, skipping: {signal['output']}"
            )
            continue

        make_data(
            infile=signal["input"],
            outdir=signal["output"],
            ratio=signal.get("ratio", 1.0),
            pt_min=signal.get("pt_min"),
            pt_max=signal.get("pt_max"),
            **common,
        )
