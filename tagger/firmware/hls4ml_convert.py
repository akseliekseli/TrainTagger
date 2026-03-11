import glob
from argparse import ArgumentParser
import yaml

from tagger.model.common import fromFolder

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_path",
        default="output/baseline",
        help="Input model path for conversion",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        default="firmware/L1TSC4NGJetModel",
        help="Jet tagger synthesized output directory",
    )

    args = parser.parse_args()

    config = [f for f in glob.glob(f"{args.model_path}/*.yaml")]
    print(config)
    with open(config[0], "r") as stream:
        yaml_dict = yaml.safe_load(stream)
    dataset = yaml_dict["data"]
    labels_to_use = yaml_dict["labels"]
    # Load the model

    # DONE: Add model yaml_dict to the fromFolder()
    model = fromFolder(args.model_path, yaml_dict)
    model.hls4ml_convert(args.outpath, build=False)
