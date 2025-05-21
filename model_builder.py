"""
Advanced Hyperparameter Search Script for Surrogate Models

Performs batch training of surrogate models over a broad hyperparameter space.
Designed for robust testing of architectural and optimization variations.

Author: Eiríkur Bernharðsson
"""

import itertools
import random
import os
import tensorflow as tf
from libraries.DNN import SurrogateModelBuilder

# --- PATHS & DATA ---
DATA_FOLDER = "data_comsol"
DATA_FILE = "2D_002.csv"
DATA_PATH = os.path.join(DATA_FOLDER, DATA_FILE)

MODELS_FOLDER = "data_models"
LOG_FILE = "data_models_docs.csv"
OUTPUTS = 1
TEST_SIZE = 0.2
USE_EARLY_STOPPING = True

# --- HYPERPARAMETER GRID ---

layer_options = [
    [128, 64],
    [128, 64, 32],
    [256, 128, 64]
]

learning_rates = [1e-3, 5e-4]

epochs_list = [30, 50, 100, 150]

batch_sizes = [16, 32]

activations = ['swish', tf.keras.layers.LeakyReLU(alpha=0.2)]

dropout_rates = [0.1, 0.2]

loss_functions = ['huber', 'mae']

optimizers = [tf.keras.optimizers.Adam, tf.keras.optimizers.Nadam]


# --- COMBINATORICS ---

all_combinations = list(itertools.product(
    layer_options,
    learning_rates,
    epochs_list,
    batch_sizes,
    activations,
    dropout_rates,
    loss_functions,
    optimizers
))

# --- SAMPLE SET ---

random.seed(42)
MAX_CONFIGS = 100  # Reduce if training is too slow
sampled_combinations = random.sample(all_combinations, k=min(MAX_CONFIGS, len(all_combinations)))

# --- TRAIN LOOP ---

for idx, (layer_sizes, lr, epochs, batch_size, activation,
          dropout_rate, loss_function, optimizer_class) in enumerate(sampled_combinations):
    
    act_name = activation if isinstance(activation, str) else activation.__class__.__name__
    print(f"\n[{idx+1}/{len(sampled_combinations)}] Training with: "
          f"layers={layer_sizes}, lr={lr}, epochs={epochs}, batch={batch_size}, "
          f"activation={act_name}, dropout={dropout_rate}, "
          f"loss={loss_function}, optimizer={optimizer_class.__name__}")

    hyperparams = {
        'layer_sizes': layer_sizes,
        'learning_rate': lr,
        'epochs': epochs,
        'batch_size': batch_size,
        'test_size': TEST_SIZE,
        'activation': activation,
        'optimizer_class': optimizer_class,
        'loss_function': loss_function,
        'use_early_stopping': USE_EARLY_STOPPING,
        'dropout_rate': dropout_rate,
        'verbose': False
    }

    builder = SurrogateModelBuilder(
        data_path=DATA_PATH,
        outputs=OUTPUTS,
        hyperparameters=hyperparams,
        models_log=LOG_FILE,
        models_folder=MODELS_FOLDER
    )

    builder.build_and_train(v=True)
