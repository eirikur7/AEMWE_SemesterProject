## This script is used to build and train a surrogate model using the SurrogateModelBuilder class.
# It defines a grid of hyperparameters, samples a subset for training, and logs the models.

from libraries.SurrogateModelBuilder import SurrogateModelBuilder
from os import path
import itertools
import random


FOLDER_DATA = "comsol_data"
FILE_NAME_DATA = "2D_002.csv"
FILE_PATH_DATA = path.join(FOLDER_DATA, FILE_NAME_DATA)
FOLDER_MODELS = "models"
LOG_FILE = "models.csv"

# Define a grid of hyperparameters
layer_sizes = [128, 64, 32]
lr = 1e-3
epochs = 50
batch_size = 16

dropout_rate = 0.2
test_size = 0.2
outputs = 1

# Generate all hyperparameter combinations
# all_combinations = list(itertools.product(layer_options, learning_rates, epochs_list, batch_sizes))

# Sample a subset for training
# random.seed(42)
# sampled_configs = random.sample(all_combinations, k=50)

# # Train and log models for each configuration
# for idx, (layer_sizes, lr, epochs, batch_size) in enumerate(sampled_configs):
#     print(f"\nTraining model {idx+1}/50 with layers={layer_sizes}, lr={lr}, epochs={epochs}, batch_size={batch_size}")

builder = SurrogateModelBuilder(
    data_path=FILE_PATH_DATA,
    models_log=LOG_FILE,
    models_folder=FOLDER_MODELS,
    outputs=outputs,
    layer_sizes=layer_sizes,
    learning_rate=lr,
    epochs=epochs,
    batch_size=batch_size,
    test_size=test_size,
    use_early_stopping=True,
    dropout_rate=dropout_rate
)

model, scaler, history, X_test, y_test = builder.build_and_train()