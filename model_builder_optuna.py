import optuna
import os
import tensorflow as tf
from libraries.DNN import SurrogateModelBuilder

# --- Configuration ---
DATA_FOLDER = "data_comsol"
DATA_FILE = "3D_001.csv"
DATA_PATH = os.path.join(DATA_FOLDER, DATA_FILE)

MODELS_FOLDER = "data_models"
LOG_FILE = "data_models_docs.csv"
OUTPUTS = 1
TEST_SIZE = 0.2
USE_EARLY_STOPPING = True

# --- Objective Function ---
def objective(trial):
    # Architecture
    n_layers = trial.suggest_int("n_layers", 3, 6)
    layer_sizes = [trial.suggest_int(f"n_units_l{i}", 128, 1024, step=64) for i in range(n_layers)]

    # Regularization
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])

    # Optimization
    lr = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
    optimizer_class = trial.suggest_categorical("optimizer", [
        tf.keras.optimizers.Adam,
        tf.keras.optimizers.Nadam,
        tf.keras.optimizers.AdamW
    ])
    
    # Training setup
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    epochs = trial.suggest_categorical("epochs", [50, 100, 150])
    
    # Activation and loss
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "swish"])
    loss_function = trial.suggest_categorical("loss_function", ["mse", "mae", "huber"])

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
        'verbose': True
    }

    builder = SurrogateModelBuilder(
        data_path=DATA_PATH,
        outputs=OUTPUTS,
        hyperparameters=hyperparams,
        models_log=LOG_FILE,
        models_folder=MODELS_FOLDER
    )

    builder.build_and_train(v=True)

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
