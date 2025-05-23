
# Electrolyzer Surrogate Modeling Project

## Overview

This project centers on building surrogate models to approximate the output of COMSOL-based electrolyzer simulations. The focus is on three pillars:

1. **Design of Experiments (DOE) generation**
2. **Deep Neural Network (DNN) building and training**
3. **DNN evaluation and analysis**

The workflow enables end-to-end automation from input space definition to final model performance validation.

---

## 1. Design of Experiments (DOE)

### Purpose
DOE is used to define the simulation input space efficiently and comprehensively. It helps generate parameter sets that span the design space without redundancy or clustering.

### Capabilities
- Supports multiple DOE strategies:  
  - Full Factorial  
  - Latin Hypercube Sampling (LHS)  
  - Maximin-enhanced LHS  
  - Sobol sequences  
  - Random and Grid sampling  
  - Fractional factorial (for high-dimensional design)

- Handles units and scaling automatically via a JSON definition file.
- Supports "linked parameters" like `i0_ref_H2` and `i0_ref_O2`, which are calculated from temperature and KOH concentration using empirical equations.
- Exports both CSV and COMSOL-readable `.txt` files for use in simulations.

### Usage
1. Define parameters in `data_DOE_input_parameters/1MKOH_input_parameters.json`.
2. Run `DOE_builder.py` and choose your desired sampling method.
3. Outputs are saved to `data_DOE_output_results/` and used by the COMSOL runner.

---

## 2. DNN Build and Training

### Purpose
Train surrogate models that learn the mapping from simulation inputs (e.g., geometry, temperature, electrolyte concentration) to outputs (e.g., current density, voltage).

### Capabilities
- Flexible model construction via `SurrogateModelBuilder` in `libraries/DNN.py`.
- Input/output dimensions are inferred from the data headers dynamically.
- Configurable training hyperparameters:
  - Layer sizes, activation functions, dropout
  - Optimizers (Adam, Nadam, AdamW)
  - Loss functions (MSE, MAE, Huber)
  - Early stopping and validation split
- Grid search (`model_builder.py`) and Bayesian optimization via Optuna (`model_builder_optuna.py`).

### Usage
1. Place your COMSOL-generated dataset into `data_from_comsol/`.
2. Run either `model_builder.py` (manual search) or `model_builder_optuna.py` (automated search).
3. Trained models and scalers are saved to `data_trained_models/`.
4. Training metadata is logged in `data_models_docs.csv`.

---

## 3. DNN Evaluation and Analysis

### Purpose
Validate the trained models, compare them to COMSOL simulation outputs, and diagnose model behavior.

### Capabilities
- Reconstruct and plot PV (E_cell vs I_density) curves with simulated and predicted overlays.
- Identify best-performing models across metrics like MAE, RMSE, R², and validation loss.
- Perform statistical analysis on model errors and prediction patterns.

### Tools
- `model_PV_curves.py`: Rebuilds PV curves from trained models.
- `validate_model.py`: Loads any saved model and computes test set metrics.
- `model_prediction_analysis.ipynb`: In-depth model behavior analysis.

### Outputs
- Figures saved to `figures/`:  
  - All PV curves overlay  
  - Best models per metric  
  - Bar charts for all models’ scores
- Quantitative logs saved to `data_models_docs.csv`.

---

## File Structure

```
AEMWE_SemesterProject/
├── data_DOE_input_parameters/       # JSON config files
├── data_DOE_output_results/         # DOE results
├── data_from_comsol/                # Simulation output files
├── data_trained_models/             # Saved .keras models and .pkl scalers
├── figures/                         # Evaluation plots
├── libraries/                       # Core DOE and DNN modules
├── model_builder.py                 # Grid search
├── model_builder_optuna.py         # Optuna hyperparameter optimization
├── model_PV_curves.py              # PV curve evaluator
├── validate_model.py               # Test set evaluator
├── README.md                       # Project documentation
```

---

## Next Steps

- Expand to more complex 3D geometries and physics.
- Quantify performance gains over COMSOL runtimes.
- Investigate surrogate model robustness in extrapolation zones.
- Apply surrogate for real-time optimization and control strategies.

---

This project enables rapid, scalable experimentation and analysis for electrochemical system design using surrogate modeling techniques.
