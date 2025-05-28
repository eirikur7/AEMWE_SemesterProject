import optuna
import os
import tensorflow as tf
from modules.DNN import SurrogateModelBuilder

# --- Configuration ---
DATA_FOLDER = os.path.join("data", "COMSOL")
DATA_FILE = "results_3D_GE_Applied_Current_1MKOH_63_02_1MKOH_input_parameters_DOE_maximin_lhs_processed_003.csv"
DATA_PATH = os.path.join(DATA_FOLDER, DATA_FILE)

MODELS_FOLDER = os.path.join("data", "DNN_trained_models")
LOG_FILE = os.path.join("data", "DNN_trained_models_docs.csv")
OUTPUTS = 1
TEST_SIZE = 0.2
USE_EARLY_STOPPING = True

# --- Hyperparameter Search Space ---
HYPERPARAMETER_SEARCH_SPACE = {
    "n_layers": (1, 20),  # deeper nets may help capture time dynamics
    "n_units": (8, 2040, 16),  # allow wide layers for expressive capacity
    "dropout_rate": (0.0, 0.4),  # allow for no dropout, up to moderate
    "use_batch_norm": [True, False],
    "learning_rate": (1e-5, 5e-2),  # full log-scale range
    "optimizer": ["adam", "nadam", "adamw", "rmsprop"],  # use strings, map later
    "batch_size": [4, 8, 16, 32, 64, 128, 256, 512],  # allow smaller batches due to small dataset
    "epochs": [5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 350, 400, 500],  # allow more training if needed
    "activation": ["relu", "tanh", "swish", "gelu"],  # gelu added for smooth regression
    "loss_function": ["mse", "mae", "huber"],  # all relevant for regression
}




# --- Objective Function ---
def objective(trial):
    n_layers = trial.suggest_int("n_layers", *HYPERPARAMETER_SEARCH_SPACE["n_layers"])
    layer_sizes = [
        trial.suggest_int(f"n_units_l{i}",
                          HYPERPARAMETER_SEARCH_SPACE["n_units"][0],
                          HYPERPARAMETER_SEARCH_SPACE["n_units"][1],
                          step=HYPERPARAMETER_SEARCH_SPACE["n_units"][2])
        for i in range(n_layers)
    ]

    dropout_rate = trial.suggest_float("dropout_rate", *HYPERPARAMETER_SEARCH_SPACE["dropout_rate"])
    batch_norm = trial.suggest_categorical("use_batch_norm", HYPERPARAMETER_SEARCH_SPACE["use_batch_norm"])

    lr = trial.suggest_float("learning_rate", HYPERPARAMETER_SEARCH_SPACE["learning_rate"][0],
                             HYPERPARAMETER_SEARCH_SPACE["learning_rate"][1], log=True)
    
    optimizer_name = trial.suggest_categorical("optimizer", HYPERPARAMETER_SEARCH_SPACE["optimizer"])
    optimizer_map = {
        "adam": tf.keras.optimizers.Adam,
        "nadam": tf.keras.optimizers.Nadam,
        "adamw": tf.keras.optimizers.AdamW,
        "rmsprop": tf.keras.optimizers.RMSprop,
    }
    optimizer_class = optimizer_map[optimizer_name]

    
    batch_size = trial.suggest_categorical("batch_size", HYPERPARAMETER_SEARCH_SPACE["batch_size"])
    epochs = trial.suggest_categorical("epochs", HYPERPARAMETER_SEARCH_SPACE["epochs"])
    
    activation = trial.suggest_categorical("activation", HYPERPARAMETER_SEARCH_SPACE["activation"])
    loss_function = trial.suggest_categorical("loss_function", HYPERPARAMETER_SEARCH_SPACE["loss_function"])

    hyperparams = {
        'layer_sizes': layer_sizes,
        'learning_rate': lr,
        'epochs': epochs,
        'batch_size': batch_size,
        'batch_norm': batch_norm,
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

    builder.build_and_train(v=False)
    return builder.metrics["mae"]

# --- Run Optuna ---
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=1000, timeout=None)

    print("\nBest trial:")
    best = study.best_trial
    print(f"  Value (MAE): {best.value:.6f}")
    print("  Params:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")
