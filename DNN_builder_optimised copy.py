
import os
import optuna
import tensorflow as tf

from modules.DNN import SurrogateModelBuilder

# --------------------------------------------------------------------- #
# paths & constants
# --------------------------------------------------------------------- #
DATA_FOLDER = os.path.join("data", "COMSOL")
DATA_FILE = "results_3D_GE_Applied_Current_1MKOH_63_03_1MKOH_input_parameters_DOE_maximin_lhs_success_001.csv"
DATA_PATH = os.path.join(DATA_FOLDER, DATA_FILE)

MODELS_FOLDER = os.path.join("data", "DNN_trained_models")
LOG_FILE = os.path.join("data", "DNN_trained_models_docs.csv")

OUTPUTS = 1
TEST_SIZE = 0.2
USE_EARLY_STOPPING = True

# --------------------------------------------------------------------- #
# hyper-parameter search space
# --------------------------------------------------------------------- #
SEARCH_SPACE = {
    "n_layers": (1, 8),                     # shallower nets are enough
    "n_units_min": 16,
    "n_units_max": 512,
    "dropout": (0.0, 0.4),
    "batch_norm": [True, False],
    "learning_rate": (1e-5, 5e-2),
    "optimizer": ["adam", "nadam", "adamw", "rmsprop"],
    "batch_size": [16, 32, 64, 128, 256],
    "epochs": [40, 60, 80, 120, 150],       # 300 removed
    "activation": ["relu", "tanh", "swish", "gelu"],
    "loss": ["mse", "mae", "huber"],
    "weight_decay": (1e-6, 1e-3),
}

OPTIM_MAP = {
    "adam": tf.keras.optimizers.Adam,
    "nadam": tf.keras.optimizers.Nadam,
    "adamw": tf.keras.optimizers.AdamW,
    "rmsprop": tf.keras.optimizers.RMSprop,
}


# --------------------------------------------------------------------- #
def objective(trial: optuna.Trial) -> float:
    # layer structure ----------------------------------------------------
    n_layers = trial.suggest_int("n_layers", SEARCH_SPACE["n_layers"][0], SEARCH_SPACE["n_layers"][1])
    layer_sizes = [
        trial.suggest_int(f"units_l{i}", SEARCH_SPACE["n_units_min"], SEARCH_SPACE["n_units_max"], log=True)
        for i in range(n_layers)
    ]

    # regularisation / optimiser ----------------------------------------
    dropout_rate = trial.suggest_float("dropout", *SEARCH_SPACE["dropout"])
    batch_norm = trial.suggest_categorical("batch_norm", SEARCH_SPACE["batch_norm"])

    lr = trial.suggest_float("learning_rate", *SEARCH_SPACE["learning_rate"], log=True)
    optimizer_name = trial.suggest_categorical("optimizer", SEARCH_SPACE["optimizer"])
    weight_decay = trial.suggest_float("weight_decay", *SEARCH_SPACE["weight_decay"], log=True)

    # prefer AdamW when weight_decay > 0 --------------------------------
    if weight_decay > 0 and optimizer_name != "adamw":
        optimizer_name = "adamw"
    optimizer_cls = OPTIM_MAP[optimizer_name]

    # misc hyper-params --------------------------------------------------
    batch_size = trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"])
    epochs = trial.suggest_categorical("epochs", SEARCH_SPACE["epochs"])
    activation = trial.suggest_categorical("activation", SEARCH_SPACE["activation"])
    loss_fn = trial.suggest_categorical("loss", SEARCH_SPACE["loss"])

    # package for builder -----------------------------------------------
    hyper = dict(
        layer_sizes=layer_sizes,
        learning_rate=lr,
        epochs=epochs,
        batch_size=batch_size,
        batch_norm=batch_norm,
        test_size=TEST_SIZE,
        activation=activation,
        optimizer_class=optimizer_cls,
        loss_function=loss_fn,
        use_early_stopping=USE_EARLY_STOPPING,
        dropout_rate=dropout_rate,
        verbose=False,
        weight_decay=weight_decay,
    )

    builder = SurrogateModelBuilder(
        data_path=DATA_PATH,
        outputs=OUTPUTS,
        hyperparameters=hyper,
        models_log=LOG_FILE,
        models_folder=MODELS_FOLDER,
    )

    builder.build_and_train(v=False)
    return builder.metrics["mae"]


# --------------------------------------------------------------------- #
if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=500)

    print("\n=== BEST TRIAL ===")
    print(f"MAE : {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"{k:>15}: {v}")
