# validation.py
# This script validates a saved deep neural network model by loading the test data,
# applying the same preprocessing (using the saved scaler), and evaluating performance
# using metrics like MSE (loss), MAE, RMSE, and R².

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from os import path
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Define paths and parameters
FOLDER_DATA = "comsol_data"
FILE_NAME = "2D_001.csv"
PATH_DATA = path.join(FOLDER_DATA, FILE_NAME)
SKIPROWS = 8

# Path to scaler and model
FOLDER_MODELS = "models"
MODEL_NAME = "model_20250407_112922"

MODEL_PATH = path.join(FOLDER_MODELS, MODEL_NAME+".keras")
SCALER_PATH = path.join(FOLDER_MODELS, MODEL_NAME+"_scaler.pkl")

def plot_pv():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from tensorflow.keras.models import load_model
    import joblib
    
    # Path to the simulated data file (tmp.csv)
    SIM_DATA_PATH = "tmp.csv"
    
    # Load simulated COMSOL data (I_density in A/cm²)
    sim_df = pd.read_csv(SIM_DATA_PATH)
    # Expecting columns: 'E_cell' and 'I_density'
    
    # Load the saved model and scaler (using your existing paths)
    scaler = joblib.load(SCALER_PATH)
    model = load_model(MODEL_PATH)
    
    # Define fixed parameters for prediction
    w_mem = 0.00006         # Membrane thickness in m
    temp = 303.15           # Temperature in K
    c_KOH = 1000            # KOH concentration in mol/m^3
    # For the width (X): the left section is 0.00037m and the membrane is w_mem; use the right edge of the membrane
    x_val = 0.00037 + w_mem  # x = 0.00043 m
    y_val = 0.005           # Use the midpoint of the 0-0.01 m domain for Y
    
    # Generate a range of E_cell values covering the range in the simulated data
    E_cell_min = sim_df['E_cell'].min()
    E_cell_max = sim_df['E_cell'].max()
    E_cell_range = np.linspace(E_cell_min, E_cell_max, num=100)
    
    # Create the input feature array for each E_cell value
    # The order of features is: [X (m), Y (m), c_KOH (mol/m^3), W_mem (m), T (K), E_cell (V)]
    num_points = len(E_cell_range)
    X_model = np.column_stack((
        np.full(num_points, x_val),
        np.full(num_points, y_val),
        np.full(num_points, c_KOH),
        np.full(num_points, w_mem),
        np.full(num_points, temp),
        E_cell_range
    ))
    
    # Scale the input features using the saved scaler
    X_model_scaled = scaler.transform(X_model)
    
    # Predict I_density using the model (output is in A/m²)
    I_density_pred = model.predict(X_model_scaled).flatten()
    
    # Convert I_density from A/m² to A/cm² by dividing by 10,000
    I_density_pred_cm2 = I_density_pred / 10000.0
    
    # Plot the simulated PV curve and the model prediction on the same graph
    plt.figure(figsize=(8, 6))
    plt.plot(sim_df['E_cell'], sim_df['I_density'], label='Simulated (COMSOL)', marker='o', linestyle='-')
    plt.plot(E_cell_range, I_density_pred_cm2, label='Model Prediction', marker='x', linestyle='--')
    plt.xlabel("E_cell (V)")
    plt.ylabel("I_density (A/cm²)")
    plt.title("PV Curve Comparison")
    plt.legend()
    plt.show()

def main():
    # Load and preprocess data
    df = pd.read_csv(PATH_DATA, skiprows=SKIPROWS)
    df.columns = ['X (m)', 'Y (m)', 'c_KOH (mol/m^3)', 'W_mem (m)', 'T (K)', 'E_cell (V)', 'I_density (A/m^2)']
    df = df.dropna(subset=['I_density (A/m^2)'])
    
    # Define inputs and target
    X = df[['X (m)', 'Y (m)', 'c_KOH (mol/m^3)', 'W_mem (m)', 'T (K)', 'E_cell (V)']].values
    y = df['I_density (A/m^2)'].values

    # Load the saved scaler and transform the data
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)

    # Split the data (use same parameters as during training)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Load the saved model
    model = load_model(MODEL_PATH)

    # Evaluate the model on the test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss (MSE): {test_loss}")
    print(f"Test MAE: {test_mae}")

    # Generate predictions on the test set
    y_pred = model.predict(X_test)

    # Compute additional metrics: RMSE and R² Score
    rmse = np.sqrt(np.mean((y_pred.flatten() - y_test) ** 2))
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse}")
    print(f"Test R^2 Score: {r2}")

if __name__ == "__main__":
    # main()
    plot_pv()