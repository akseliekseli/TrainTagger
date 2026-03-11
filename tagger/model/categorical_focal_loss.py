import numpy as np
import tensorflow as tf


def categorical_focal_loss(gamma=1.0, alpha=None):
    def loss(y_true, y_pred):
        # Clip to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # standard categorical CE: -sum(y_true * log(y_pred))
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

        # p_t: predicted probability of the true class
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)

        # alpha balance
        if alpha is not None:
            alpha_factor = tf.reduce_sum(y_true * alpha, axis=-1)
        else:
            alpha_factor = 1.0

        modulating_factor = (1.0 - p_t) ** gamma

        return alpha_factor * modulating_factor * ce

    return loss
