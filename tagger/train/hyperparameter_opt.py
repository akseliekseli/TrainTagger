import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser

# Third parties
import numpy as np
import yaml
import tensorflow as tf
import optuna

from tagger.model.common import fromFolder, fromYaml


def objective():
    model = fromYaml(args.yaml_config, args.output)
