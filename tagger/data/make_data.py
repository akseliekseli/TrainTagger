import os
from argparse import ArgumentParser

import yaml

from tagger.data.tools import make_data
from tagger.config_schema import UNIFIED_CONFIG_SCHEMA


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Unified experiment configuration",
    )
    args = parser.parse_args()

    with open(args.config) as stream:
        config = yaml.safe_load(stream)

    UNIFIED_CONFIG_SCHEMA.validate(config)

    data_config = config["data_config"]

    make_data(
        infile=data_config["input"],
        outdir=data_config["output"],
        step_size=data_config.get("step_size", "100MB"),
        extras=data_config.get("extras", "extra_fields"),
        ratio=data_config.get("ratio", 1.0),
        tree=data_config.get("tree", "outnano/Jets"),
        num_workers=data_config.get("num_workers", 8),
    )

    for signal in data_config.get("signal_processes", []):
        if os.path.exists(signal["output"]):
            print(f"Signal output exists, skipping: {signal['output']}")
            continue

        make_data(
            infile=signal["input"],
            outdir=signal["output"],
            step_size=data_config.get("step_size", "100MB"),
            extras=data_config.get("extras", "extra_fields"),
            ratio=data_config.get("ratio", 1.0),
            tree=signal.get(
                "tree",
                data_config.get("tree", "outnano/Jets"),
            ),
            num_workers=data_config.get("num_workers", 8),
        )
