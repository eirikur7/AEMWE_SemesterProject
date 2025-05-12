import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Original data points
T_vals = np.array([30, 50, 60, 80])
KOH_vals = np.array([1.0, 0.1, 0.01])
T_grid, KOH_grid = np.meshgrid(T_vals, KOH_vals)

# Flatten input and output
T_flat = T_grid.ravel()
KOH_flat = KOH_grid.ravel()
X_raw = np.column_stack((T_flat, KOH_flat))

# j0 values
y_an = np.array([
    [0.014, 0.027, 0.037, 0.035],
    [0.006, 0.020, 0.028, 0.030],
    [0.0003, 0.001, 0.00249, 0.0028]
]).ravel()

# ==========================
# 1. Plot: j0 vs T, colored by KOH
# ==========================
plt.figure()
for koh in KOH_vals:
    mask = KOH_flat == koh
    plt.plot(T_flat[mask], y_an[mask], 'o-', label=f'KOH = {koh} mol/L')
plt.xlabel('Temperature (°C)')
plt.ylabel('j0 (A/cm²)')
plt.title('j0 vs T at different KOH concentrations')
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# 2. Plot: j0 vs log10(KOH), colored by T
# ==========================
plt.figure()
for T in T_vals:
    mask = T_flat == T
    plt.plot(np.log10(KOH_flat[mask]), y_an[mask], 'o-', label=f'T = {T}°C')
plt.xlabel('log10(KOH conc)')
plt.ylabel('j0 (A/cm²)')
plt.title('j0 vs log10(KOH) at different Temperatures')
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# 3. Curve Fitting: Polynomial in T and log10(KOH)
# ==========================
log_KOH = np.log10(KOH_flat)
X_features = np.column_stack((T_flat, log_KOH))

# Use degree-2 polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_features)

model = LinearRegression()
model.fit(X_poly, y_an)
y_pred = model.predict(X_poly)

# Print model coefficients for inspection
feature_names = poly.get_feature_names_out(['T', 'logKOH'])
for name, coef in zip(feature_names, model.coef_):
    print(f"{name}: {coef:.5e}")
print(f"Intercept: {model.intercept_:.5e}")

# Optional: 3D surface plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(T_flat, log_KOH, y_an, c='r', label='Data')
ax.scatter(T_flat, log_KOH, y_pred, c='b', label='Fit', alpha=0.6)
ax.set_xlabel('T (°C)')
ax.set_ylabel('log10(KOH)')
ax.set_zlabel('j0 (A/cm²)')
plt.title("2D Polynomial Fit of j0(T, log10(KOH))")
plt.legend()
plt.show()
