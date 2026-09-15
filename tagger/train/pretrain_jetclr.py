"""
JetCLR-style contrastive pretraining of the DeepSetModel conv1d backbone.

Usage:
    python pretrain_jetclr.py -y tagger/model/configs/baseline.yaml -o output/pretrain

Saves:
    <out_dir>/encoder_weights.h5   -- weights for the shared conv1d backbone only.
    Load these into a freshly-built DeepSetModel before the normal supervised
    training run (see the load_pretrained_encoder() helper added to train.py).
"""

import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
import yaml

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tagger.data.tools import load_data, to_ML
from tagger.model.contrastive import (
    ContrastivePretrainer,
    build_encoder,
    build_projection_head,
)


def normalize(X_train):
    """Copied from train.py rather than imported, since importing train.py
    re-runs its module-level tf.config.threading calls and crashes with
    'Inter op parallelism cannot be modified after initialization' once TF
    has already been initialized here. Keep this in sync with train.py's
    version if that one changes.
    """
    N_features = X_train.shape[-1]
    X_normalized = X_train.copy()
    non_boolean_indices = []

    for i in range(N_features):
        feature_channel = X_train[..., i]
        is_binary = np.all(np.logical_or(feature_channel == 0, feature_channel == 1))
        if not is_binary:
            non_boolean_indices.append(i)

    print(f"Features to normalize (indices): {non_boolean_indices}")
    if not non_boolean_indices:
        print("No numeric features found. Returning original data.")
        return X_train

    X_to_normalize = X_train[..., non_boolean_indices]
    mean = np.mean(X_to_normalize, axis=(0, 1), keepdims=True)
    std = np.std(X_to_normalize, axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0

    X_normalized_subset = (X_to_normalize - mean) / std
    X_normalized[..., non_boolean_indices] = X_normalized_subset
    return X_normalized


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-y", "--yaml_config", default="tagger/model/configs/baseline.yaml"
    )
    parser.add_argument("-o", "--output", default="output/pretrain")
    parser.add_argument("-p", "--percent", default=100, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--temperature", default=0.1, type=float)

    # Feature-column assumptions -- POST config["columns"] selection.
    # Fix these if your Δη / Δφ / pT-like feature aren't at these indices.
    parser.add_argument("--deta_col", default=1, type=int)
    parser.add_argument("--dphi_col", default=2, type=int)
    parser.add_argument("--pt_col", default=0, type=int)
    parser.add_argument("--smear_sigma", default=0.01, type=float)
    parser.add_argument("--drop_frac", default=0.15, type=float)
    args = parser.parse_args()

    with open(args.yaml_config, "r") as f:
        yaml_dict = yaml.safe_load(f)

    os.makedirs(args.output, exist_ok=True)

    print("Loading data...")
    data_train, _, class_labels, input_vars, extra_vars = load_data(
        yaml_dict["data"], percentage=args.percent, test_ratio=0.2
    )
    labels_to_use = yaml_dict["labels"]
    if labels_to_use == "all":
        labels_to_use = None

    (
        X_train,
        y_train,
        pt_target_train,
        truth_pt_train,
        reco_pt_train,
        mass_target_train,
        _,
    ) = to_ML(data_train, class_labels, labels_to_use)
    X_train_constits, _ = X_train

    # Same pT cut as train.py, contrastive pretraining should see the same phase space
    pt_cut = 10.0
    pt_mask = reco_pt_train > pt_cut
    X_train_constits = X_train_constits[pt_mask]
    print(f"  Kept {pt_mask.sum()} / {len(pt_mask)} jets after pT cut")

    if yaml_dict["columns"]:
        X_train_constits = X_train_constits[:, :, yaml_dict["columns"]]

    print(
        "Normalizing (note: this changes the physical meaning of deta/dphi/pt "
        "columns -- if you want augmentations in physical units, consider "
        "augmenting BEFORE normalization instead; left as normalize-then-augment "
        "here for simplicity, revisit if augmentation strength looks off)"
    )
    X_train_constits = normalize(X_train_constits)

    input_shape = X_train_constits.shape[1:]
    print(f"Input shape: {input_shape}")

    encoder = build_encoder(
        input_shape,
        yaml_dict["model_config"],
        yaml_dict["quantization_config"],
        name_prefix="",
    )
    encoder.summary()

    # infer embedding dim from encoder output
    embed_dim = encoder.output_shape[-1]
    proj_head = build_projection_head(embed_dim, proj_dim=64, hidden_dim=128)

    aug_config = dict(
        deta_col=args.deta_col,
        dphi_col=args.dphi_col,
        pt_col=args.pt_col,
        smear_sigma=args.smear_sigma,
        drop_frac=args.drop_frac,
    )

    pretrainer = ContrastivePretrainer(
        encoder=encoder,
        proj_head=proj_head,
        aug_config=aug_config,
        temperature=args.temperature,
    )
    pretrainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr))

    dataset = (
        tf.data.Dataset.from_tensor_slices(X_train_constits.astype(np.float32))
        .shuffle(10000)
        .batch(args.batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="contrastive_loss", patience=10),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="contrastive_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(args.output, "pretrain_log.csv")),
    ]

    print("Starting contrastive pretraining...")
    pretrainer.fit(
        dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    weights_path = os.path.join(args.output, "encoder_weights.h5")
    encoder.save_weights(weights_path)
    print(f"Encoder weights saved to {weights_path}")
    print(
        "Next: pass --pretrained_encoder "
        f"{weights_path} to train.py to warm-start the tagger's backbone."
    )


if __name__ == "__main__":
    main()
