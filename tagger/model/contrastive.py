"""
JetCLR-style self-supervised contrastive pretraining for the DeepSet jet backbone.

Physics-motivated augmentations (following JetCLR, Dillon et al. 2108.04253):
  - rotation of constituents in the (deta, dphi) plane around the jet axis
  - small IRC-inspired smearing: gaussian jitter on (deta, dphi), plus
    soft-constituent dropout (randomly zero out low-pT constituents)

These are applied on the fly to produce two augmented "views" of each jet,
which are then treated as a positive pair for an NT-Xent (InfoNCE) loss.

IMPORTANT: column indices below are POST feature-selection (i.e. indices
into the array after config["columns"] has already been applied in train.py).
Adjust PT_COL / DETA_COL / DPHI_COL in the config if your selected feature
order differs from the assumption documented in pretrain_jetclr.py.
"""

import numpy as np
import tensorflow as tf
from qkeras import QConv1D
from qkeras.qlayers import QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from tensorflow.keras.layers import BatchNormalization, Dense, Activation

from tagger.model.common import choose_aggregator


# ──────────────────────────────────────────────────────────────────────────
# Augmentations
# ──────────────────────────────────────────────────────────────────────────
def _is_filled_mask(x):
    """(B, N, F) -> (B, N) boolean mask of constituents that aren't zero-padding."""
    return tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)


def augment_rotate(x, deta_col, dphi_col, mask):
    """Random rotation of each jet's constituents in the (deta, dphi) plane.

    x: (B, N, F). Rebuilds the feature axis via unstack/stack rather than
    scatter_nd, since scatter_nd's shape bookkeeping is easy to get subtly
    wrong when F is small and static-shape inference is involved.
    """
    batch = tf.shape(x)[0]
    theta = tf.random.uniform((batch, 1), 0.0, 2.0 * np.pi)  # (B, 1), broadcasts over N
    cos_t, sin_t = tf.cos(theta), tf.sin(theta)

    deta = x[..., deta_col]  # (B, N)
    dphi = x[..., dphi_col]  # (B, N)

    new_deta = cos_t * deta - sin_t * dphi
    new_dphi = sin_t * deta + cos_t * dphi

    new_deta = tf.where(mask, new_deta, deta)
    new_dphi = tf.where(mask, new_dphi, dphi)

    cols = tf.unstack(x, axis=-1)  # list of F tensors, each (B, N)
    cols[deta_col] = new_deta
    cols[dphi_col] = new_dphi
    return tf.stack(cols, axis=-1)


def augment_smear(x, deta_col, dphi_col, mask, sigma=0.01):
    """Small gaussian smearing of (deta, dphi) -- crude IRC-safety-inspired jitter."""
    mask_f = tf.cast(mask, x.dtype)
    deta_noise = tf.random.normal(tf.shape(mask_f), stddev=sigma) * mask_f
    dphi_noise = tf.random.normal(tf.shape(mask_f), stddev=sigma) * mask_f

    cols = tf.unstack(x, axis=-1)
    cols[deta_col] = cols[deta_col] + deta_noise
    cols[dphi_col] = cols[dphi_col] + dphi_noise
    return tf.stack(cols, axis=-1)


def augment_soft_drop(x, pt_col, mask, drop_frac=0.15):
    """
    Randomly zero out a fraction of the SOFTEST filled constituents per jet
    (crude stand-in for a collinear/soft-safety augmentation -- not a rigorous
    IRC-safe procedure, just a cheap way to perturb without touching hard substructure).
    """
    pt = tf.where(
        mask,
        x[..., pt_col],
        tf.fill(tf.shape(x[..., pt_col]), tf.constant(1e9, x.dtype)),
    )
    n_filled = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)  # (B,)
    n_drop = tf.cast(tf.cast(n_filled, tf.float32) * drop_frac, tf.int32)

    # rank constituents by pT ascending (softest first); drop the bottom n_drop of them
    order = tf.argsort(pt, axis=1, direction="ASCENDING")
    rank = tf.argsort(
        order, axis=1
    )  # rank[i] = position of constituent i in ascending order
    n_drop_b = tf.expand_dims(n_drop, 1)
    drop_this = tf.logical_and(mask, rank < n_drop_b)

    keep = tf.logical_not(drop_this)
    keep_f = tf.cast(tf.expand_dims(keep, -1), x.dtype)
    return x * keep_f


def jetclr_augment(
    x, deta_col=1, dphi_col=2, pt_col=0, smear_sigma=0.01, drop_frac=0.15
):
    """Apply the full augmentation pipeline -> one 'view' of the batch."""
    mask = _is_filled_mask(x)
    x = augment_rotate(x, deta_col, dphi_col, mask)
    x = augment_smear(x, deta_col, dphi_col, mask, sigma=smear_sigma)
    x = augment_soft_drop(x, pt_col, mask, drop_frac=drop_frac)
    return x


# ──────────────────────────────────────────────────────────────────────────
# Encoder (mirrors DeepSetModel's conv1d backbone so weights transfer 1:1)
# ──────────────────────────────────────────────────────────────────────────
def build_encoder(input_shape, model_config, quantization_config, name_prefix=""):
    """Builds the SAME conv1d + aggregator backbone as DeepSetModel.build_model,
    with matching layer names, so weights can be loaded into the full tagger later.
    """
    common_args = {
        "kernel_quantizer": quantized_bits(
            quantization_config["quantizer_bits"],
            quantization_config["quantizer_bits_int"],
            alpha=quantization_config["quantizer_alpha_val"],
        ),
        "bias_quantizer": quantized_bits(
            quantization_config["quantizer_bits"],
            quantization_config["quantizer_bits_int"],
            alpha=quantization_config["quantizer_alpha_val"],
        ),
        "kernel_initializer": model_config["kernel_initializer"],
    }

    inputs = tf.keras.layers.Input(shape=input_shape, name=name_prefix + "model_input")
    main = BatchNormalization(name=name_prefix + "norm_input")(inputs)

    for iconv1d, depthconv1d in enumerate(model_config["conv1d_layers"]):
        main = QConv1D(
            filters=depthconv1d,
            kernel_size=1,
            name=name_prefix + "Conv1D_" + str(iconv1d + 1),
            **common_args,
        )(main)
        main = QActivation(
            activation=quantized_relu(quantization_config["quantizer_bits"], 0),
            name=name_prefix + "relu_" + str(iconv1d + 1),
        )(main)

    main = QActivation(
        activation="quantized_bits(18,8)", name=name_prefix + "act_pool"
    )(main)
    agg = choose_aggregator(
        choice=model_config["aggregator"], name=name_prefix + "pool"
    )
    pooled = agg(main)

    return tf.keras.Model(inputs=inputs, outputs=pooled, name=name_prefix + "encoder")


def build_projection_head(embed_dim, proj_dim=64, hidden_dim=128, name="proj_head"):
    """Standard SimCLR-style MLP projection head, only used during pretraining
    and discarded afterwards (only the encoder backbone gets transferred)."""
    inp = tf.keras.layers.Input(shape=(embed_dim,))
    h = Dense(hidden_dim, activation=None, name=name + "_dense1")(inp)
    h = Activation("relu", name=name + "_relu1")(h)
    out = Dense(proj_dim, activation=None, name=name + "_dense2")(h)
    return tf.keras.Model(inputs=inp, outputs=out, name=name)


# ──────────────────────────────────────────────────────────────────────────
# NT-Xent / InfoNCE loss
# ──────────────────────────────────────────────────────────────────────────
def nt_xent_loss(z1, z2, temperature=0.1):
    """
    z1, z2: (B, D) projected embeddings of the two augmented views.
    Standard SimCLR NT-Xent over the 2B-sample batch, one positive per anchor.
    """
    batch = tf.shape(z1)[0]
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)

    z = tf.concat([z1, z2], axis=0)  # (2B, D)
    sim = tf.matmul(z, z, transpose_b=True) / temperature  # (2B, 2B)

    # mask out self-similarity
    diag_mask = tf.eye(2 * batch, dtype=tf.bool)
    sim = tf.where(diag_mask, tf.fill(tf.shape(sim), -1e9), sim)

    # positive index for row i is (i + B) mod 2B
    positives = tf.concat([tf.range(batch, 2 * batch), tf.range(0, batch)], axis=0)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=positives, logits=sim)
    return tf.reduce_mean(loss)


# ──────────────────────────────────────────────────────────────────────────
# Full contrastive trainer
# ──────────────────────────────────────────────────────────────────────────
class ContrastivePretrainer(tf.keras.Model):
    """Wraps encoder + projection head, custom train_step does the augment -> encode
    -> project -> NT-Xent loop. Only `encoder` gets saved for downstream transfer.
    """

    def __init__(self, encoder, proj_head, aug_config, temperature=0.1, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.proj_head = proj_head
        self.aug_config = aug_config
        self.temperature = temperature
        self.loss_tracker = tf.keras.metrics.Mean(name="contrastive_loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, x, training=False):
        return self.proj_head(self.encoder(x, training=training), training=training)

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data

        view1 = jetclr_augment(
            x,
            deta_col=self.aug_config["deta_col"],
            dphi_col=self.aug_config["dphi_col"],
            pt_col=self.aug_config["pt_col"],
            smear_sigma=self.aug_config["smear_sigma"],
            drop_frac=self.aug_config["drop_frac"],
        )
        view2 = jetclr_augment(
            x,
            deta_col=self.aug_config["deta_col"],
            dphi_col=self.aug_config["dphi_col"],
            pt_col=self.aug_config["pt_col"],
            smear_sigma=self.aug_config["smear_sigma"],
            drop_frac=self.aug_config["drop_frac"],
        )

        with tf.GradientTape() as tape:
            z1 = self.proj_head(self.encoder(view1, training=True), training=True)
            z2 = self.proj_head(self.encoder(view2, training=True), training=True)
            loss = nt_xent_loss(z1, z2, temperature=self.temperature)

        trainable_vars = (
            self.encoder.trainable_variables + self.proj_head.trainable_variables
        )
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.loss_tracker.update_state(loss)
        return {"contrastive_loss": self.loss_tracker.result()}

    def test_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        view1 = jetclr_augment(
            x, **{k: v for k, v in self.aug_config.items() if k != "batch"}
        )
        view2 = jetclr_augment(
            x, **{k: v for k, v in self.aug_config.items() if k != "batch"}
        )
        z1 = self.proj_head(self.encoder(view1, training=False), training=False)
        z2 = self.proj_head(self.encoder(view2, training=False), training=False)
        loss = nt_xent_loss(z1, z2, temperature=self.temperature)
        self.loss_tracker.update_state(loss)
        return {"contrastive_loss": self.loss_tracker.result()}
