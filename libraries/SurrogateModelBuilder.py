import pandas as pd
import numpy as np
import datetime
import joblib
import os
from os import path
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error, median_absolute_error

class SurrogateModelBuilder:
    """
    Builds, trains, evaluates, and logs a surrogate deep learning model using TensorFlow/Keras.
    Includes support for early stopping, dynamic input/output column handling, and comprehensive logging.
    """

    def __init__(self, data_path, models_log, models_folder, outputs,
                 layer_sizes, learning_rate, epochs, batch_size, test_size,
                 activation='relu', optimizer_class=Adam, loss_function='mse',
                 use_early_stopping=False, dropout_rate=None, output_columns=None):
        """Initialize the builder with model and training configuration."""
        self.data_path = data_path
        self.models_log = models_log
        self.models_folder = models_folder
        self.outputs = outputs
        self.output_columns = output_columns
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_size = test_size
        self.activation = activation
        self.optimizer_class = optimizer_class
        self.loss_function = loss_function
        self.use_early_stopping = use_early_stopping
        self.dropout_rate = dropout_rate

        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        self.model_name = None
        self.trained_at = None
        self.training_duration = None
        self.metrics = {}

    def _load_data(self):
        """Loads and preprocesses the dataset from CSV, including dynamic column handling and scaling."""
        with open(self.data_path, 'r') as file:
            lines = file.readlines()
        skip_rows = sum(1 for line in lines if line.strip().startswith('%'))
        header_line = lines[skip_rows - 1].lstrip('%').strip()
        column_names = [col.strip() for col in header_line.split(',')]

        df = pd.read_csv(self.data_path, skiprows=skip_rows, header=None)
        df.columns = column_names
        df = df.dropna()

        # Identify input/output columns
        if self.output_columns:
            input_columns = [col for col in df.columns if col not in self.output_columns]
            output_columns = self.output_columns
        else:
            input_columns = df.columns[:-self.outputs]
            output_columns = df.columns[-self.outputs:]

        X = df[input_columns].values
        y = df[output_columns].values

        X_scaled = self.scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=42
        )
        self.num_features = X.shape[1]
        self.num_outputs = y.shape[1]

    def _build_model(self):
        """Constructs the neural network model with specified architecture and dropout."""
        model = Sequential()
        model.add(Input(shape=(self.X_train.shape[1],)))
        for size in self.layer_sizes:
            model.add(Dense(size, activation=self.activation))
            if self.dropout_rate is not None:
                model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.y_train.shape[1]))  # Output layer
        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=self.loss_function, metrics=['mae'])
        self.model = model

    def _train_model(self):
        """Trains the model with optional early stopping."""
        callbacks = []
        if self.use_early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True))

        start_time = datetime.datetime.now()
        self.history = self.model.fit(
            self.X_train, self.y_train, validation_split=0.1,
            epochs=self.epochs, batch_size=self.batch_size,
            callbacks=callbacks, verbose=0
        )
        end_time = datetime.datetime.now()
        self.training_duration = (end_time - start_time).total_seconds()

    def _evaluate_model(self):
        """Evaluates the model on the test set and records various metrics."""
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
        """Saves the trained model and its scaler to disk."""
        now = datetime.datetime.now()
        base_name = os.path.splitext(os.path.basename(self.data_path))[0]
        self.model_name = f"model_{base_name}__{now.strftime('%y%m%d_%H%M%S')}"
        self.trained_at = now.strftime('%Y-%m-%d %H:%M:%S')

        os.makedirs(self.models_folder, exist_ok=True)
        self.model.save(path.join(self.models_folder, self.model_name + ".keras"))
        joblib.dump(self.scaler, path.join(self.models_folder, self.model_name + "_scaler.pkl"))

    def _log_model(self):
        """Logs model metadata and evaluation metrics to a CSV log file."""
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

    def build_and_train(self):
        """Full pipeline: load data, build, train, evaluate, save and log the model."""
        self._load_data()
        self._build_model()
        self._train_model()
        self._evaluate_model()
        self._save_model()
        self._log_model()
        return self.model, self.scaler, self.history, self.X_test, self.y_test



# import pandas as pd 
# import numpy as np 
# import matplotlib.pyplot as plt
# import seaborn as sns
# import datetime 
# import joblib 
# import os 
# from os import path 
# from sklearn.model_selection import train_test_split 
# from sklearn.preprocessing import StandardScaler 
# import tensorflow as tf 
# from tensorflow.keras.layers import Input, Dense, Dropout 
# from tensorflow.keras.models import Sequential, load_model 
# from tensorflow.keras.optimizers import Adam 
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# # Set seeds for reproducibility
# DEFAULT_RANDOM_SEED = 42
# np.random.seed(DEFAULT_RANDOM_SEED)
# tf.random.set_seed(DEFAULT_RANDOM_SEED)

# class SurrogateModelBuilder: 
#     """
#     Builds and trains a surrogate model using TensorFlow/Keras. 
#     This class handles data loading, preprocessing, 
#     model construction, training, saving, and logging.
#     """

#     DEFAULT_RANDOM_SEED = DEFAULT_RANDOM_SEED
#     DEFAULT_SCALER_TYPE = "StandardScaler"
#     DEFAULT_EARLY_STOPPING = "N/A"
#     DEFAULT_DROPOUT_RATE = "N/A"
    
#     def __init__(self, data_path, models_log, models_folder, outputs, layer_sizes,
#                  learning_rate, epochs, batch_size, test_size,
#                  activation='relu', optimizer_class=Adam, loss_function='mse',
#                  use_early_stopping=False, dropout_rate=None):
#         self.data_path = data_path
#         self.models_log = models_log
#         self.models_folder = models_folder
#         self.outputs = outputs
#         self.layer_sizes = layer_sizes
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.test_size = test_size
#         self.activation = activation
#         self.optimizer_class = optimizer_class
#         self.loss_function = loss_function
#         self.use_early_stopping = use_early_stopping
#         self.dropout_rate = dropout_rate if dropout_rate is not None else self.DEFAULT_DROPOUT_RATE

#         self.metrics = None
#         self.scaler = None
#         self.X_train = None
#         self.X_test = None
#         self.y_train = None
#         self.y_test = None
#         self.model = None
#         self.history = None
#         self.model_name = None
#         self.training_duration = None
#         self.trained_at = None
#         self.df = None

#     def _estimate_num_parameters(self):
#         """Estimate total trainable parameters using layer sizes and input/output dimensions."""
#         input_dim = self.X_train.shape[1]
#         output_dim = self.y_train.shape[1]
#         total = 0
#         prev = input_dim
#         for size in self.layer_sizes:
#             total += (prev * size) + size  # weights + biases
#             prev = size
#         total += (prev * output_dim) + output_dim
#         return total


#     def _load_data(self):
#         """
#         Loads COMSOL data from the specified CSV file, detects and skips metadata,
#         applies custom column modifications, and splits the data.
#         """
#         # Detect how many lines start with '%'
#         with open(self.data_path, 'r') as file:
#             lines = file.readlines()
#         skip_rows = 0
#         for line in lines:
#             if line.strip().startswith('%'):
#                 skip_rows += 1
#             else:
#                 break

#         # Extract header from the last metadata line
#         header_line = lines[skip_rows - 1].lstrip('%').strip()
#         column_names = [col.strip() for col in header_line.split(',')]

#         # Load CSV skipping metadata lines but using extracted header
#         self.df = pd.read_csv(self.data_path, skiprows=skip_rows, header=None)
#         self.df.columns = column_names

#         self._apply_custom_column_modifications(self.df)

#         # Split dataframe into inputs and outputs
#         input_columns = self.df.columns[:-self.outputs]
#         output_columns = self.df.columns[-self.outputs:]

#         X = self.df[input_columns].values
#         y = self.df[output_columns].values

#         self.scaler = StandardScaler()
#         X_scaled = self.scaler.fit_transform(X)

#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             X_scaled, y, test_size=self.test_size, random_state=42
#         )

#     def _build_model(self):
#         self.model = Sequential()
#         self.model.add(Input(shape=(self.X_train.shape[1],)))
#         for size in self.layer_sizes:
#             self.model.add(Dense(size, activation=self.activation))
#             if self.dropout_rate != "N/A":
#                 self.model.add(Dropout(float(self.dropout_rate)))

#         self.model.add(Dense(self.y_train.shape[1]))  # Output layer
#         optimizer = self.optimizer_class(learning_rate=self.learning_rate)
#         self.model.compile(optimizer=optimizer, loss=self.loss_function, metrics=['mae'])


#     def _train_model(self):
#         start_time = datetime.datetime.now()
#         self.history = self.model.fit(
#             self.X_train, self.y_train, validation_split=0.1,
#             epochs=self.epochs, batch_size=self.batch_size, verbose=1
#         )
#         end_time = datetime.datetime.now()
#         self.training_duration = (end_time - start_time).total_seconds()
        

#     def _save_model(self):
#         now = datetime.datetime.now()
#         data_base = os.path.splitext(os.path.basename(self.data_path))[0]
#         self.model_name = f"model_{data_base}__{now.strftime('%y%m%d_%H%M%S')}"
#         self.trained_at = now.strftime('%Y-%m-%d %H:%M:%S')

#         if not os.path.exists(self.models_folder):
#             os.makedirs(self.models_folder)

#         model_filepath = path.join(self.models_folder, self.model_name + ".keras")
#         self.model.save(model_filepath)
#         scaler_filepath = path.join(self.models_folder, self.model_name + "_scaler.pkl")
#         joblib.dump(self.scaler, scaler_filepath)
#         print(f"Model saved as: {model_filepath}")
#         print(f"Scaler saved as: {scaler_filepath}")
    
    
#     def _evaluate_model(self):
#         predictions = self.model.predict(self.X_test)
#         mae = mean_absolute_error(self.y_test, predictions)
#         rmse = mean_squared_error(self.y_test, predictions, squared=False)
#         r2 = r2_score(self.y_test, predictions)
#         self.metrics = {
#             'mae': mae,
#             'rmse': rmse,
#             'r2': r2
#         }

#     def _log_model(self):
#         try:
#             models_df = pd.read_csv(log_filepath=self.models_log)
#         except FileNotFoundError:
#             models_df = pd.DataFrame(columns=[
#                 'model_name', 'data_file', 'num_layers', 'layer_sizes',
#                 'learning_rate', 'epochs', 'batch_size', 'training_duration',
#                 'trained_at', 'mae', 'rmse', 'r2',
#                 'test_size', 'random_seed', 'activation', 'loss_function',
#                 'optimizer', 'dropout_rate', 'early_stopping',
#                 'scaler_type', 'train_size', 'num_parameters'
#             ])
#         new_model = pd.DataFrame({
#             'model_name': [self.model_name],
#             'data_file': [os.path.basename(self.data_path)],
#             'num_layers': [len(self.layer_sizes)],
#             'layer_sizes': [self.layer_sizes],
#             'learning_rate': [self.learning_rate],
#             'epochs': [self.epochs],
#             'batch_size': [self.batch_size],
#             'training_duration': [self.training_duration],
#             'trained_at': [self.trained_at],
#             'mae': [self.metrics.get('mae')],
#             'rmse': [self.metrics.get('rmse')],
#             'r2': [self.metrics.get('r2')],
#             'test_size': [self.test_size],
#             'random_seed': [self.DEFAULT_RANDOM_SEED],
#             'activation': [self.activation],
#             'loss_function': [self.loss_function],
#             'optimizer': [self.optimizer_class.__name__],
#             'dropout_rate': [self.dropout_rate],
#             'early_stopping': [self.use_early_stopping],
#             'scaler_type': [self.DEFAULT_SCALER_TYPE],
#             'train_size': [1 - self.test_size],
#             'num_parameters': [self._estimate_num_parameters()]
#         })
#         models_df = pd.concat([models_df, new_model], ignore_index=True)
#         models_df.to_csv(self.models_log, index=False)
#         print(f"Model logged in: {self.models_log}")
        

#     def build_and_train(self):
#         self._load_data()
#         self._build_model()
#         self._train_model()
#         self._save_model()
#         self._evaluate_model()
#         self._log_model()
#         return self.model, self.scaler, self.history, self.X_test, self.y_test


# if __name__ == "__main__": 
#     FOLDER_DATA = "comsol_data" 
#     FILE_NAME_DATA = "2D_002.csv" 
#     FILE_PATH_DATA = path.join(FOLDER_DATA, FILE_NAME_DATA)
#     FOLDER_MODELS = "models"
#     LOG_FILE = "models.csv"

#     layer_sizes = [6, 4, 2]
#     learning_rate = 1e-3
#     epochs = 100
#     batch_size = 32
#     test_size = 0.2
#     outputs = 1

#     builder = SurrogateModelBuilder(
#         data_path=FILE_PATH_DATA,
#         models_log=LOG_FILE,
#         models_folder=FOLDER_MODELS,
#         outputs=outputs,
#         layer_sizes=layer_sizes,
#         learning_rate=learning_rate,
#         epochs=epochs,
#         batch_size=batch_size,
#         test_size=test_size
#     )
#     model, scaler, history, X_test, y_test = builder.build_and_train()