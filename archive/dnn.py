### Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

### Load and preprocess data
df = pd.read_csv("tbl3.csv", skiprows=8)
df.columns = ['X (m)', 'Y (m)', 'c_KOH (mol/m^3)', 'W_mem (m)', 'T (K)', 'E_cell (V)', 'I_density (A/m^2)']

# Drop rows with NaN target
df = df.dropna(subset=['I_density (A/m^2)'])

# Input and output
X = df[['X (m)', 'Y (m)', 'c_KOH (mol/m^3)', 'W_mem (m)', 'T (K)', 'E_cell (V)']].values
y = df['I_density (A/m^2)'].values

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

### Define hyperparameters
NUM_LAYERS = 4
LAYER_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 32

### build the model
model = Sequential()
model.add(Dense(LAYER_SIZE, activation='relu', input_shape=(X_train.shape[1],)))
for _ in range(NUM_LAYERS - 1):
    model.add(Dense(LAYER_SIZE, activation='relu'))
model.add(Dense(1))  # Output layer for regression


### Compile the model
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])


### Train the model
history = model.fit(X_train, y_train, validation_split=0.1,
                    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)