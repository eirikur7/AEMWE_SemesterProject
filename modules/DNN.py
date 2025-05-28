# -*- coding: utf-8 -*-
"""
SurrogateModelBuilder

This module defines the SurrogateModelBuilder class, a utility for constructing, training, evaluating,
and documenting deep neural network surrogate models for simulation data, specifically tailored for data
exported from COMSOL Multiphysics. It provides flexible support for custom architectures, optimizer
choices, early stopping, and extensive logging of both training metrics and model metadata.

Input data is expected to be in CSV format, with comment lines prefixed by '%'. The final comment line
must define the column headers. Input features are assumed to occupy the initial columns and output targets
the final columns, unless explicitly defined via `output_columns`.

Typical use cases include surrogate modeling, sensitivity studies, and hyperparameter optimization pipelines.
"""

import os
import time
import datetime
import joblib
import pandas as pd
import numpy as np
from os import path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    max_error, median_absolute_error
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

class SurrogateModelBuilder:
    """
    Class for building, training, evaluating, and logging deep neural network surrogate models using TensorFlow/Keras.

    Supports configurable architectures, input/output column handling, early stopping, training time tracking,
    evaluation metrics, model persistence, and experiment logging.
    """

    def __init__(self, data_path, outputs, hyperparameters,
                 output_columns = None, 
                 models_log     = os.path.join("data", "DNN_trained_models_docs.csv"),
                 models_folder  = os.path.join("data", "DNN_trained_models")):
        """
        Initialize the model builder with dataset location, expected output shape, and training hyperparameters.

        Parameters:
            data_path (str): Path to the input CSV data file.
            outputs (int): Number of output variables (if output_columns is not specified).
            hyperparameters (dict): Dictionary with keys:
                - layer_sizes (list of int): Hidden layer sizes.
                - learning_rate (float): Optimizer learning rate.
                - epochs (int): Number of training epochs.
                - batch_size (int): Training batch size.
                - test_size (float): Fraction of data to use as test set.
                - activation (str): Activation function for hidden layers.
                - optimizer_class (tf.keras.optimizers.Optimizer): Optimizer class.
                - loss_function (str): Loss function name (e.g., 'mse').
                - use_early_stopping (bool): Enable early stopping based on validation loss.
                - dropout_rate (float or None): Dropout rate (0–1) or None for no dropout.
                - verbose (bool): Verbosity flag for training.
            output_columns (list of str, optional): Names of output columns. If not provided, inferred.
            models_log (str): Path to the model log CSV.
            models_folder (str): Directory for saving models and scalers.
        """
        self.data_path = data_path
        self.outputs = outputs
        self.output_columns = output_columns
        self.models_log = models_log
        self.models_folder = models_folder

        # Extract and store hyperparameters
        self.layer_sizes = hyperparameters['layer_sizes']
        self.learning_rate = hyperparameters['learning_rate']
        self.epochs = hyperparameters['epochs']
        self.batch_size = hyperparameters['batch_size']
        self.batch_norm = hyperparameters['batch_norm']
        self.test_size = hyperparameters['test_size']
        self.activation = hyperparameters['activation']
        self.optimizer_class = hyperparameters['optimizer_class']
        self.loss_function = hyperparameters['loss_function']
        self.use_early_stopping = hyperparameters['use_early_stopping']
        self.dropout_rate = hyperparameters['dropout_rate']
        self.verbose = hyperparameters['verbose']

        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        self.model_name = None
        self.trained_at = None
        self.training_duration = None
        self.metrics = {}

    def __str__(self):
        if self.model_name:
            return (f"SurrogateModelBuilder Summary\n"
                    f"Model: {self.model_name}\n"
                    f"Trained at: {self.trained_at}\n"
                    f"Layer sizes: {self.layer_sizes}\n"
                    f"MAE: {self.metrics.get('mae', 'N/A'):.4f}, "
                    f"RMSE: {self.metrics.get('rmse', 'N/A'):.4f}, "
                    f"R^2: {self.metrics.get('r2', 'N/A'):.4f}, "
                    f"MAPE: {self.metrics.get('mape', 'N/A'):.2f}%")
        return "SurrogateModelBuilder (untrained)"

    def _load_data(self):
        """Load and preprocess data assuming a standard CSV format with headers on the first line."""
        df = pd.read_csv(self.data_path)
        df.dropna(inplace=True)

        if self.output_columns:
            input_cols = [col for col in df.columns if col not in self.output_columns]
            output_cols = self.output_columns
        else:
            input_cols = df.columns[:-self.outputs]
            output_cols = df.columns[-self.outputs:]

        X = df[input_cols].values
        y = df[output_cols].values

        X_scaled = self.scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=42)

        self.num_features = X.shape[1]
        self.num_outputs = y.shape[1]


    def _build_model(self):
        """Constructs the Keras model architecture based on the provided configuration."""
        model = Sequential()
        model.add(Input(shape=(self.X_train.shape[1],)))

        for size in self.layer_sizes:
            model.add(Dense(size, activation=self.activation))
            if self.batch_norm:
                model.add(BatchNormalization())
            if self.dropout_rate is not None:
                model.add(Dropout(self.dropout_rate))


        model.add(Dense(self.y_train.shape[1]))  # Output layer

        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=self.loss_function, metrics=['mae'])

        self.model = model

    def _train_model(self):
        """Trains the Keras model and tracks training duration and performance."""
        callbacks = []
        if self.use_early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True))

        start = datetime.datetime.now()
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_split=0.1,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=int(self.verbose))
        end = datetime.datetime.now()
        self.training_duration = (end - start).total_seconds()

    def _evaluate_model(self):
        """Evaluates the trained model and stores performance metrics."""
        start = time.time()
        y_pred = self.model.predict(self.X_test)
        end = time.time()

        self.metrics = {
            'mae': mean_absolute_error(self.y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'r2': r2_score(self.y_test, y_pred),
            'mape': np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100,
            'max_error': max_error(self.y_test, y_pred),
            'median_ae': median_absolute_error(self.y_test, y_pred),
            'inference_time_ms': ((end - start) / len(self.X_test)) * 1000,
            'val_loss_best': min(self.history.history['val_loss']),
            'best_epoch': int(np.argmin(self.history.history['val_loss']))
        }

    def _save_model(self):
        """Saves the trained model and scaler to disk in a timestamped directory."""
        now = datetime.datetime.now()
        base_name = os.path.splitext(os.path.basename(self.data_path))[0]
        self.model_name = f"model_{base_name}__{now.strftime('%y%m%d_%H%M%S')}"
        self.trained_at = now.strftime('%Y-%m-%d %H:%M:%S')

        os.makedirs(self.models_folder, exist_ok=True)
        self.model.save(path.join(self.models_folder, self.model_name + ".keras"))
        joblib.dump(self.scaler, path.join(self.models_folder, self.model_name + "_scaler.pkl"))

    def _log_model(self):
        """Appends model configuration and evaluation metrics to a persistent CSV log."""
        try:
            df_log = pd.read_csv(self.models_log)
        except FileNotFoundError:
            df_log = pd.DataFrame(columns=[
                'model_name', 'data_file', 'trained_at', 'training_duration',
                'num_layers', 'layer_sizes', 'learning_rate', 'epochs', 'batch_size',
                'test_size', 'num_features', 'num_outputs',
                'mae', 'rmse', 'r2', 'mape', 'max_error', 'median_ae',
                'val_loss_best', 'best_epoch', 'inference_time_ms',
                'activation', 'loss_function', 'optimizer', 'dropout_rate', 'early_stopping'
            ])

        df_log = pd.concat([df_log, pd.DataFrame([{
            'model_name': self.model_name,
            'data_file': os.path.basename(self.data_path),
            'trained_at': self.trained_at,
            'training_duration': self.training_duration,
            'num_layers': len(self.layer_sizes),
            'layer_sizes': self.layer_sizes,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'batch_norm': self.batch_norm,
            'test_size': self.test_size,
            'num_features': self.num_features,
            'num_outputs': self.num_outputs,
            **self.metrics,
            'activation': self.activation,
            'loss_function': self.loss_function,
            'optimizer': self.optimizer_class.__name__,
            'dropout_rate': self.dropout_rate,
            'early_stopping': self.use_early_stopping
        }])], ignore_index=True)

        df_log.to_csv(self.models_log, index=False)

    def build_and_train(self, v=False):
        """
        Orchestrates the full training pipeline: loading data, building and training the model,
        evaluating its performance, saving model artifacts, and logging results.

        Parameters:
            v (bool): If True, prints stage progress to stdout.

        Returns:
            tuple: (trained model, fitted scaler, training history, X_test, y_test)
        """
        self.verbose = v
        if v: print("[INFO] Loading data...")
        self._load_data()

        if v: print("[INFO] Building model...")
        self._build_model()

        if v: print("[INFO] Training model...")
        self._train_model()

        if v: print("[INFO] Evaluating model...")
        self._evaluate_model()

        if v: print("[INFO] Saving model...")
        self._save_model()

        if v: print("[INFO] Logging results...")
        self._log_model()

        return self.model, self.scaler, self.history, self.X_test, self.y_test
