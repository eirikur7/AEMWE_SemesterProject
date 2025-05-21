import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
from os import path 
from tensorflow.keras.models import load_model 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class SurrogateModelAnalyzer:
    """
    This class take in a path to a specific model.
    It will load the model and scaler from the specified folder.
    It can evaluate the model on a given dataset and return the metrics.
    It can take in input data and predict the output.
    """
    def __init__(self, models_folder, models_log_file):
       pass
   
    def evaluate(self, X, y):
        """
        Evaluate the model on the given dataset.
        """
        pass
    
    def predict(self, X):
        """
        Predict the output for the given input data.
        """
        pass