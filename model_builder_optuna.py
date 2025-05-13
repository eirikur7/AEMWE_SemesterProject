import optuna
import os
import tensorflow as tf
from libraries import SurrogateModelBuilder

# --- Configuration ---
DATA_FOLDER = "comsol_data"
DATA_FILE = "2D_002.csv"
DATA_PATH = os.path.join(DATA_FOLDER, DATA_FILE)

MODELS_FOLDER = "optuna_models"
LOG_FILE = "optuna_models_log.csv"
OUTPUTS = 1
TEST_SIZE = 0.2
USE_EARLY_STOPPING = True

# --- Objective Function ---
def objective(trial):
    # Suggest hyperparameters
    layer_depth = trial.suggest_int("n_layers", 1, 4)
    layer_sizes = [trial.suggest_int(f"n_units_l{i}", 32, 512, step=32) for i in range(layer_depth)]

    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    epochs = trial.suggest_categorical("epochs", [30, 50, 100, 200])
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "swish"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    loss_function = trial.suggest_categorical("loss_function", ["mse", "mae", "huber"])
    optimizer_class = trial.suggest_categorical("optimizer", [
        tf.keras.optimizers.Adam,
        tf.keras.optimizers.Nadam,
        tf.keras.optimizers.RMSprop
    ])

    # Prepare model
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

    builder.build_and_train(v=False)

    # Minimize MAE
    return builder.metrics["mae"]

# --- Run Optuna ---
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, timeout=None)  # You can increase trials

    print("\nBest trial:")
    best = study.best_trial
    print(f"  Value (MAE): {best.value:.6f}")
    print("  Params:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")
