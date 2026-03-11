import json
import os

import hls4ml
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from qkeras import QConv1D
from qkeras.qlayers import QActivation, QDense

# Qkeras
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.utils import load_qmodel
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, BatchNormalization

from tagger.model.common import AAtt, AttentionPooling, choose_aggregator
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tensorflow.python.framework import ops


from tagger.plot.basic import loss_history_adversarial
from tagger.plot.basic import loss_history

# Set some tensorflow constants
NUM_THREADS = 24
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(NUM_THREADS)
os.environ["TF_NUM_INTEROP_THREADS"] = str(NUM_THREADS)

tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)

tf.keras.utils.set_random_seed(420)


# ==============================================================================
# 1. Custom Gradient Reversal Layer (GRL)
# ==============================================================================
class GradientReversal(tf.keras.layers.Layer):
    """
    Gradient Reversal Layer (GRL): identity in forward pass, reverses gradient
    and scales it by lambda (l) in backward pass.
    """

    def __init__(self, l=1.0, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        # Ensure lambda is a constant
        self.l = tf.constant(l, dtype=tf.float32)

    def call(self, x):
        @tf.custom_gradient
        def reverse_gradient(x, l):
            def grad(dy):
                # Reverse the gradient and scale by lambda (l)
                return -l * dy, None

            # Forward pass is identity
            return tf.identity(x), grad

        return reverse_gradient(x, self.l)

    def get_config(self):
        config = super(GradientReversal, self).get_config()
        # Ensure lambda is saved as a standard float for config reconstruction
        config.update({"l": self.l.numpy()})
        return config

    # 🌟 CRITICAL FIX: Implement this method to satisfy tfmot.sparsity.keras.prune_low_magnitude
    def get_prunable_weights(self):
        """Returns the list of prunable weight tensors, which is empty for GRL."""
        return []


# ==============================================================================
# 2. Loss Function
# ==============================================================================
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


# ==============================================================================
# 3. Adversarial DeepSet Model Class
# ==============================================================================
@JetModelFactory.register("AdversarialDeepSetModel")
class AdversarialDeepSetModel(JetTagModel):
    """
    AdversarialDeepSetModel class with Classification, pT Regression, and
    Adversarial Mass Regression (using Gradient Reversal Layer).
    """

    output_mass_name = "mass_output"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom objects needed for loading/saving model that includes GRL
        self.custom_objects_ = {
            "AAtt": AAtt,
            "AttentionPooling": AttentionPooling,
            "GradientReversal": GradientReversal,
        }

    def build_model(self, inputs_shape: tuple, outputs_shape: tuple):
        """
        Builds the model layer by layer, creating three output heads.
        """

        # Define some common arguments, taken from the yaml config
        common_args = {
            "kernel_quantizer": quantized_bits(
                self.quantization_config["quantizer_bits"],
                self.quantization_config["quantizer_bits_int"],
                alpha=self.quantization_config["quantizer_alpha_val"],
            ),
            "bias_quantizer": quantized_bits(
                self.quantization_config["quantizer_bits"],
                self.quantization_config["quantizer_bits_int"],
                alpha=self.quantization_config["quantizer_alpha_val"],
            ),
            "kernel_initializer": self.model_config["kernel_initializer"],
        }

        # Initialize inputs
        inputs = tf.keras.layers.Input(shape=inputs_shape, name="model_input")

        # Main feature extractor branch
        main = BatchNormalization(name="norm_input")(inputs)

        # Make Conv1D layers (Feature Extractor)
        for iconv1d, depthconv1d in enumerate(self.model_config["conv1d_layers"]):
            main = QConv1D(
                filters=depthconv1d,
                kernel_size=1,
                name="Conv1D_" + str(iconv1d + 1),
                **common_args,
            )(main)
            main = QActivation(
                activation=quantized_relu(
                    self.quantization_config["quantizer_bits"], 0
                ),
                name="relu_" + str(iconv1d + 1),
            )(main)

        # Pooling layer to aggregate features
        main = QActivation(activation="quantized_bits(18,8)", name="act_pool")(main)
        agg = choose_aggregator(choice=self.model_config["aggregator"], name="pool")
        aggregated_features = agg(main)  # Shared features

        # --- 1. Classification branch (Jet ID) ---
        jet_id = aggregated_features
        for iclass, depthclass in enumerate(self.model_config["classification_layers"]):
            jet_id = QDense(
                depthclass,
                name="Dense_" + str(iclass + 1) + "_jetID",
                **common_args,
            )(jet_id if iclass > 0 else aggregated_features)
            jet_id = QActivation(
                activation=quantized_relu(
                    self.quantization_config["quantizer_bits"], 0
                ),
                name="relu_" + str(iclass + 1) + "_jetID",
            )(jet_id)

        jet_id = QDense(
            outputs_shape[0],
            name="Dense_"
            + str(len(self.model_config["classification_layers"]) + 1)
            + "_jetID",
            **common_args,
        )(jet_id)
        jet_id = Activation("softmax", name=self.output_id_name)(jet_id)

        # --- 2. pT Regression branch ---
        pt_regress = aggregated_features
        for ireg, depthreg in enumerate(self.model_config["regression_layers"]):
            pt_regress = QDense(
                depthreg, name="Dense_" + str(ireg + 1) + "_pT", **common_args
            )(pt_regress if ireg > 0 else aggregated_features)
            pt_regress = QActivation(
                activation=quantized_relu(
                    self.quantization_config["quantizer_bits"], 0
                ),
                name="relu_" + str(ireg + 1) + "_pT",
            )(pt_regress)

        pt_regress = QDense(
            1,
            name=self.output_pt_name,
            kernel_quantizer=quantized_bits(
                self.quantization_config["pt_output_quantization"][0],
                self.quantization_config["pt_output_quantization"][1],
                alpha=self.quantization_config["quantizer_alpha_val"],
            ),
            bias_quantizer=quantized_bits(
                self.quantization_config["pt_output_quantization"][0],
                self.quantization_config["pt_output_quantization"][1],
                alpha=self.quantization_config["quantizer_alpha_val"],
            ),
            kernel_initializer="lecun_uniform",
        )(pt_regress)

        # --- 3. Adversarial Mass Regression branch ---

        # Insert Gradient Reversal Layer (GRL)
        grl_lambda = self.model_config.get("grl_lambda", 0.1)
        reversed_features = GradientReversal(l=grl_lambda, name="GRL_Mass")(
            aggregated_features
        )

        mass_regress = reversed_features
        mass_reg_layers = self.model_config.get(
            "mass_regression_layers", self.model_config["regression_layers"]
        )

        # Mass Regressor MLP (Adversary Network)
        for imass_reg, depth_mass_reg in enumerate(mass_reg_layers):
            mass_regress = QDense(
                depth_mass_reg,
                name="Dense_" + str(imass_reg + 1) + "_mass_ADV",
                **common_args,
            )(mass_regress if imass_reg > 0 else reversed_features)
            mass_regress = QActivation(
                activation=quantized_relu(
                    self.quantization_config["quantizer_bits"], 0
                ),
                name="relu_" + str(imass_reg + 1) + "_mass_ADV",
            )(mass_regress)

        # Final Mass Output Layer
        mass_output_quant = self.quantization_config.get(
            "mass_output_quantization",
            self.quantization_config["pt_output_quantization"],
        )

        mass_regress = QDense(
            1,
            name=self.output_mass_name,
            kernel_quantizer=quantized_bits(
                mass_output_quant[0],
                mass_output_quant[1],
                alpha=self.quantization_config["quantizer_alpha_val"],
            ),
            bias_quantizer=quantized_bits(
                mass_output_quant[0],
                mass_output_quant[1],
                alpha=self.quantization_config["quantizer_alpha_val"],
            ),
            kernel_initializer="lecun_uniform",
        )(mass_regress)

        # Define the model using all three branches
        self.jet_model = tf.keras.Model(
            inputs=inputs, outputs=[jet_id, pt_regress, mass_regress]
        )

        print(self.jet_model.summary())

    def _prune_model(self, num_samples: int):
        """Pruning setup for the model, internal model function called by compile"""
        print("Begin pruning the model...")

        # Calculate the ending step for pruning
        end_step = (
            np.ceil(num_samples / self.training_config["batch_size"]).astype(np.int32)
            * self.training_config["epochs"]
        )

        # Define the pruned model
        pruning_params = {
            "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=self.training_config["initial_sparsity"],
                final_sparsity=self.training_config["final_sparsity"],
                begin_step=0,
                end_step=end_step,
            )
        }

        # This now works because GradientReversal has get_prunable_weights()
        self.jet_model = tfmot.sparsity.keras.prune_low_magnitude(
            self.jet_model, **pruning_params
        )

        # Add preface to loss name
        self.loss_name = "prune_low_magnitude_"

        # Add pruning callback
        self.callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

    def compile_model(self, num_samples: int):
        """compile the model generating callbacks and loss function"""

        # Define the callbacks using hyperparameters in the config
        self.callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.training_config["EarlyStopping_patience"],
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.training_config["ReduceLROnPlateau_factor"],
                patience=self.training_config["ReduceLROnPlateau_patience"],
                min_lr=self.training_config["ReduceLROnPlateau_min_lr"],
            ),
        ]

        # Define the pruning
        self._prune_model(num_samples)
        focal = categorical_focal_loss(gamma=2.0, alpha=None)

        regression_loss = tf.keras.losses.Huber()

        # compile the tensorflow model setting the loss and metrics
        self.jet_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.training_config["learning_rate"]
            ),
            loss={
                self.loss_name + self.output_id_name: focal,
                self.loss_name + self.output_pt_name: regression_loss,
                self.loss_name + self.output_mass_name: regression_loss,
            },
            loss_weights=self.training_config["loss_weights"],
            metrics={
                self.loss_name + self.output_id_name: "categorical_accuracy",
                self.loss_name + self.output_pt_name: ["mae", "mean_squared_error"],
                self.loss_name + self.output_mass_name: ["mae", "mean_squared_error"],
            },
            weighted_metrics={
                self.loss_name + self.output_id_name: "categorical_accuracy",
                self.loss_name + self.output_pt_name: ["mae", "mean_squared_error"],
                self.loss_name + self.output_mass_name: ["mae", "mean_squared_error"],
            },
        )

    def fit(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        pt_target_train: npt.NDArray[np.float64],
        mass_target_train: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64],
    ):
        """Fit the model to the training dataset with all three targets"""

        # Train the model using hyperparameters in yaml config
        self.history = self.jet_model.fit(
            {"model_input": X_train},
            {
                self.loss_name + self.output_id_name: y_train,
                self.loss_name + self.output_pt_name: pt_target_train,
                self.loss_name + self.output_mass_name: mass_target_train,
            },
            sample_weight=sample_weight,
            epochs=self.training_config["epochs"],
            batch_size=self.training_config["batch_size"],
            verbose=self.run_config["verbose"],
            validation_split=self.training_config["validation_split"],
            callbacks=self.callbacks,
            shuffle=True,
        )

    def plot_loss(self):
        out_dir = self.output_directory
        # Produce some basic plots with the training for diagnostics
        plot_path = os.path.join(out_dir, "plots/training")
        os.makedirs(plot_path, exist_ok=True)

        # Define the names of the losses to plot together (EXACT keys from history)
        # NOTE: The keys are the output name prefixed by the pruning layer name.
        loss_metrics_to_plot = [
            # 1. Classification Loss (Jet ID)
            self.loss_name + self.output_id_name + "_loss",
            # 2. Adversarial Loss (Mass Regression)
            self.loss_name + self.output_mass_name + "_loss",
        ]

        # Call the MODIFIED loss_history to plot the two losses on one figure.
        loss_history_adversarial(plot_path, loss_metrics_to_plot, self.history)
        loss_history(
            plot_path,
            [
                self.loss_name + self.output_id_name,
                self.loss_name + self.output_pt_name,
            ],
            self.history,
        )

    # Decorated with save decorator for added functionality
    @JetTagModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the model file"""
        # Export the model
        model_export = tfmot.sparsity.keras.strip_pruning(self.jet_model)

        os.makedirs(os.path.join(out_dir, "model"), exist_ok=True)
        # Use keras save format !NOT .h5! due to depreciation
        export_path = os.path.join(out_dir, "model/saved_model.keras")
        model_export.save(export_path)
        print(f"Model saved to {export_path}")

    @JetTagModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file"""

        # Custom objects must include GRL
        custom_objects_ = {
            "AAtt": AAtt,
            "AttentionPooling": AttentionPooling,
            "GradientReversal": GradientReversal,
        }
        # Load the model
        self.jet_model = load_qmodel(
            f"{out_dir}/model/saved_model.keras", custom_objects=custom_objects_
        )

    def hls4ml_convert(self, firmware_dir: str, build: bool = False):
        """Run the hls4ml model conversion"""

        # Remove the old directory if it exists
        hls4ml_outdir = firmware_dir + "/" + self.hls4ml_config["project_name"]
        os.system(f"rm -rf {hls4ml_outdir}")

        # Create default config
        config = hls4ml.utils.config_from_keras_model(
            self.jet_model, granularity="name"
        )
        config["IOType"] = "io_parallel"
        config["LayerName"]["model_input"]["Precision"]["result"] = self.hls4ml_config[
            "input_precision"
        ]

        # Additional config
        for layer in self.jet_model.layers:
            layer_name = layer.__class__.__name__

            if layer_name in ["BatchNormalization", "InputLayer"]:
                config["LayerName"][layer.name]["Precision"] = self.hls4ml_config[
                    "input_precision"
                ]
                config["LayerName"][layer.name]["result"] = self.hls4ml_config[
                    "input_precision"
                ]
                config["LayerName"][layer.name]["Trace"] = not build
            elif layer_name in [
                "Permute",
                "Concatenate",
                "Flatten",
                "Reshape",
                "UpSampling1D",
                "Add",
                "GradientReversal",
            ]:
                print("Skipping trace for:", layer.name)
            else:
                config["LayerName"][layer.name]["Trace"] = not build

        config["LayerName"][self.output_id_name]["Precision"]["result"] = (
            self.hls4ml_config["class_precision"]
        )
        config["LayerName"][self.output_id_name]["Implementation"] = "latency"

        config["LayerName"][self.output_pt_name]["Precision"]["result"] = (
            self.hls4ml_config["reg_precision"]
        )
        config["LayerName"][self.output_pt_name]["Implementation"] = "latency"

        # Mass output configuration
        config["LayerName"][self.output_mass_name]["Precision"]["result"] = (
            self.hls4ml_config["reg_precision"]
        )
        config["LayerName"][self.output_mass_name]["Implementation"] = "latency"

        # Write HLS
        self.hls_jet_model = hls4ml.converters.convert_from_keras_model(
            self.jet_model,
            backend="Vitis",
            project_name=self.hls4ml_config["project_name"],
            clock_period=2.5,  # 1/360MHz = 2.8ns
            hls_config=config,
            output_dir=f"{hls4ml_outdir}",
            part="xcvu13p-flga2577-2-e",
        )

        # Compile the project
        self.hls_jet_model.compile()

        # Save config  as json file
        print("Saving default config as config.json ...")
        with open(hls4ml_outdir + "/config.json", "w") as fp:
            json.dump(config, fp)

        if build:
            # build the project
            self.hls_jet_model.build(csim=False, reset=True)
