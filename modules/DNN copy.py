import os, time, datetime, joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    max_error, median_absolute_error
)

import tensorflow as tf
from tensorflow.keras import regularizers, losses, optimizers
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization
)

# ────────────────────────────────────────────────────────────────────────────
# surrogate
# ────────────────────────────────────────────────────────────────────────────
class SurrogateModelBuilder:
    """
    Build, train, evaluate and log a Keras dense-net surrogate for COMSOL data.
    """

    # ------------------------------------------------------------------ #
    # constructor
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        data_path: str,
        outputs: int,
        hyperparameters: Dict,
        output_columns: Optional[List[str]] = None,
        models_log: str = os.path.join("data", "DNN_trained_models_docs.csv"),
        models_folder: str = os.path.join("data", "DNN_trained_models"),
    ):
        # data & I/O
        self.data_path      = data_path
        self.output_columns = output_columns
        self.outputs        = outputs
        self.models_folder  = models_folder
        self.models_log     = models_log

        # hyper-parameters
        self.layer_sizes   = hyperparameters["layer_sizes"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.epochs        = hyperparameters["epochs"]
        self.batch_size    = hyperparameters["batch_size"]
        self.batch_norm    = hyperparameters["batch_norm"]
        self.test_size     = hyperparameters["test_size"]
        self.activation    = hyperparameters["activation"]
        self.optimizer_cls = hyperparameters["optimizer_class"]
        self.loss_function = hyperparameters["loss_function"]
        self.use_early_stopping = hyperparameters["use_early_stopping"]
        self.dropout_rate  = hyperparameters["dropout_rate"]
        self.verbose       = hyperparameters.get("verbose", False)
        self.weight_decay  = hyperparameters.get("weight_decay", 0.0)

        # internal state
        self.x_scaler  = StandardScaler()
        self.y_scaler  = StandardScaler()
        self.model     : Optional[tf.keras.Model] = None
        self.history   = None
        self.metrics   = {}
        self.model_name = ""
        self.trained_at = ""
        self.training_duration = 0.0

    # ------------------------------------------------------------------ #
    # data prep
    # ------------------------------------------------------------------ #
    def _prepare_data(self) -> None:
        df = pd.read_csv(self.data_path).dropna()

        # split input / output
        if self.output_columns:
            in_cols = [c for c in df.columns if c not in self.output_columns]
            out_cols = self.output_columns
        else:
            in_cols  = df.columns[:-self.outputs]
            out_cols = df.columns[-self.outputs:]

        X = df[in_cols].astype(float).values
        y = df[out_cols].astype(float).values
        self.num_features, self.num_outputs = X.shape[1], y.shape[1]

        # build simulation key – use explicit ID if present
        sim_key = (
            df["simulation_id"].astype(str).values
            if "simulation_id" in df.columns
            else df[in_cols[:12]]
                 .applymap(lambda v: f"{float(v):.6g}")
                 .agg("|".join, axis=1)
                 .values
        )

        # outer split – test set (never touched during training)
        gss_outer = GroupShuffleSplit(n_splits=1,
                                      test_size=self.test_size,
                                      random_state=42)
        train_idx, test_idx = next(gss_outer.split(X, groups=sim_key))

        # inner split – validation set (10 % of *training* groups)
        gss_inner = GroupShuffleSplit(n_splits=1,
                                      test_size=0.1,
                                      random_state=42)
        tr_idx, val_idx = next(
            gss_inner.split(X[train_idx], groups=sim_key[train_idx])
        )
        tr_idx  = train_idx[tr_idx]
        val_idx = train_idx[val_idx]

        # scale X
        self.x_scaler.fit(X[tr_idx])
        self.X_train = self.x_scaler.transform(X[tr_idx])
        self.X_val   = self.x_scaler.transform(X[val_idx])
        self.X_test  = self.x_scaler.transform(X[test_idx])

        # scale y
        self.y_scaler.fit(y[tr_idx])
        self.y_train = self.y_scaler.transform(y[tr_idx])
        self.y_val   = self.y_scaler.transform(y[val_idx])
        self.y_test  = y[test_idx]                             # keep original scale
        self.y_test_scaled = self.y_scaler.transform(y[test_idx])

    # ------------------------------------------------------------------ #
    # model
    # ------------------------------------------------------------------ #
    def _build_model(self) -> None:
        reg = regularizers.L2(self.weight_decay) if self.weight_decay else None
        model = tf.keras.Sequential(name="dense_net")

        for i, units in enumerate(self.layer_sizes, 1):
            model.add(Dense(units, activation=self.activation,
                            kernel_regularizer=reg,
                            name=f"dense_{i}"))
            if self.batch_norm:
                model.add(BatchNormalization(name=f"bn_{i}"))
            if self.dropout_rate is not None and self.dropout_rate > 0.0:
                model.add(Dropout(self.dropout_rate, name=f"dropout_{i}"))

        # output layer WITH regulariser
        model.add(Dense(self.num_outputs,
                        kernel_regularizer=reg,
                        name="dense_out"))

        opt = self.optimizer_cls(learning_rate=self.learning_rate)
        loss_map = {
            "mse": losses.MeanSquaredError(),
            "mae": losses.MeanAbsoluteError(),
            "huber": losses.Huber(),
        }
        model.compile(optimizer=opt,
                      loss=loss_map[self.loss_function],
                      metrics=["mae"])
        self.model = model

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def _train_model(self) -> None:
        cb = [
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.3, patience=5, min_lr=1e-6, verbose=0
            )
        ]
        if self.use_early_stopping:
            patience = max(5, int(self.epochs * 0.10))
            cb.append(
                tf.keras.callbacks.EarlyStopping(
                    patience=patience,
                    restore_best_weights=True,
                    verbose=0,
                )
            )

        tic = datetime.datetime.now()
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=cb,
            verbose=int(self.verbose),
        )
        toc = datetime.datetime.now()
        self.training_duration = (toc - tic).total_seconds()

    # ------------------------------------------------------------------ #
    # evaluation
    # ------------------------------------------------------------------ #
    def _evaluate_model(self) -> None:
        y_pred_scaled = self.model.predict(self.X_test, verbose=0)
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)

        eps = 1e-8
        mape = np.nanmean(
            np.abs((self.y_test - y_pred) /
                   np.where(np.abs(self.y_test) < eps, np.nan, self.y_test))
        ) * 100

        self.metrics = {
            "mae":   mean_absolute_error(self.y_test, y_pred),
            "rmse":  np.sqrt(mean_squared_error(self.y_test, y_pred)),
            "r2":    r2_score(self.y_test, y_pred),
            "mape":  mape,
            "max_error":  max_error(self.y_test, y_pred),
            "median_ae": median_absolute_error(self.y_test, y_pred),
            "inference_time_ms": (toc := datetime.datetime.now(), (toc - tic := toc).total_seconds() / len(self.X_test) * 1e3)[1],  # one-liner to avoid tmp vars
            "val_loss_best": float(np.min(self.history.history["val_loss"])),
            "best_epoch":    int(np.argmin(self.history.history["val_loss"])),
        }

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def _save_artifacts(self) -> None:
        ts = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        base = Path(self.data_path).stem
        self.model_name = f"model_{base}__{ts}"
        self.trained_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        Path(self.models_folder).mkdir(parents=True, exist_ok=True)
        self.model.save(Path(self.models_folder, f"{self.model_name}.keras"))
        joblib.dump(self.x_scaler, Path(self.models_folder, f"{self.model_name}_x_scaler.pkl"))
        joblib.dump(self.y_scaler, Path(self.models_folder, f"{self.model_name}_y_scaler.pkl"))

    # ------------------------------------------------------------------ #
    # driver
    # ------------------------------------------------------------------ #
    def build_and_train(self, v: bool = False
                        ) -> Tuple[tf.keras.Model, StandardScaler, dict,
                                   np.ndarray, np.ndarray]:
        if v:
            print("· Preparing data")
        self._prepare_data()

        if v:
            print("· Building model")
        self._build_model()

        if v:
            print("· Training")
        self._train_model()

        if v:
            print("· Evaluating")
        self._evaluate_model()

        if v:
            print("· Saving artifacts")
        self._save_artifacts()

        if v:
            print("· Logging run")
        self._log_run()

        return (self.model,
                {"x": self.x_scaler, "y": self.y_scaler},
                self.history,
                self.X_test,
                self.y_test)
