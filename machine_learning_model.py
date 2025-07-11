import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Reading the data from the csv file
data = pd.read_csv('new_simulation_data_to_test.csv')

# Encoding non-numeric categorical columns: 'Geometry' and 'Material'
for col in ['Geometry', 'Material']:
    if col in data.columns:
        data = pd.get_dummies(data, columns=[col])

# Converting all columns to numeric values
data = data.apply(pd.to_numeric, errors='coerce')

# Target variable is MaxTemperature_C (log-transform for stability)
X = data.drop(columns=['MaxTemperature_C'])
y_raw = data['MaxTemperature_C']
y = np.log1p(y_raw)  # log(1 + x)

# Splitting the data into training and testing data (80% train, 20% test)
X_train, X_test, y_train, y_test_raw = train_test_split(X, y_raw, test_size=0.2, random_state=42)
y_train_log = np.log1p(y_train)

# Creating the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fitting the model to the training data (on log-transformed target)
model.fit(X_train, y_train_log)

# Making predictions on the test data (returning to original scale)
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # inverse of log1p

# Saving the model
model_filename = 'temperature_prediction_model.pkl'
joblib.dump(model, model_filename)

# Loading the model for future use
loaded_model = joblib.load(model_filename)

# Making predictions with the loaded model
loaded_pred_log = loaded_model.predict(X_test)
loaded_pred = np.expm1(loaded_pred_log)

#data cleaning for figure 1, remove all temp values beyond 10**8 degrees
upper_limit = 1e8 
valid_mask = (y_pred < upper_limit) & (y_test_raw < upper_limit) #masking the unwanted data and keeping the wanted data
y_pred_filtered = y_pred[valid_mask]
y_test_filtered = y_test_raw[valid_mask]


# Compute min and max for consistent axes
min_val = min(y_test_filtered.min(), y_pred_filtered.min())
max_val = max(y_test_filtered.max(), y_pred_filtered.max())

#Standard scale plot predicted temp vs actual temp
plt.figure(figsize=(8, 6))
plt.scatter(y_test_filtered, y_pred_filtered, alpha=0.5)
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
plt.xlabel('Actual Max Temperature ($^{\circ}C$)')
plt.ylabel('Predicted Max Temperature ($^{\circ}C$)')
plt.title('Predicted vs Actual Temperature ($^{\circ} C$})')
plt.grid(True, which='both')
plt.ticklabel_format(style='plain', axis='both')
plt.tight_layout()
plt.savefig('Fig1_Predicted_vs_Actual_standard_scale.png')
plt.show()

# Fit linear regression model for reference (on log-transformed target)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train_log)
coefficients = lin_model.coef_
intercept = lin_model.intercept_

# Load data again for sweep setup
df = pd.read_csv("new_simulation_data_to_test.csv")
df_encoded = pd.get_dummies(df, columns=['Geometry', 'Material'])

target = 'MaxTemperature_C'
features = df_encoded.columns.drop([target])
X = df_encoded[features]
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Refit the model for the sweep
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, np.log1p(y_train))

y_pred = np.expm1(model.predict(X_test))


# === Start Parameter Sweep ===

# Default parameters
default_values = {
    'Wavelength_m': 1e-6,
    'DomainLength_m': 0.01,
    'Sigma_s': 5.67e-8,
    'Geometry': 'Cartesian',
    'Material': 'Aluminum',
}

# Sweep ranges
power_range = np.logspace(-1, 2, 10)          # 0.1 W to 100 W
beam_radius_range = np.logspace(-6, -2, 10)   # 1 µm to 1 cm

# Material properties (density, specific heat)
material_properties = {
    "Aluminum":    {"k": 237,  "rho": 2700,  "c": 897},
    "Copper":      {"k": 401,  "rho": 8960,  "c": 385},
    "Gold":        {"k": 318,  "rho": 19300, "c": 129},
    "Iron":        {"k": 80,   "rho": 7874,  "c": 449},
    "Lead":        {"k": 35,   "rho": 11340, "c": 128},
    "Nickel":      {"k": 90.9, "rho": 8908,  "c": 444},
    "Silver":      {"k": 429,  "rho": 10490, "c": 235},
    "Steel":       {"k": 50,   "rho": 7850,  "c": 486},
    "Titanium":    {"k": 21.9, "rho": 4507,  "c": 522},
    "Tungsten":    {"k": 173,  "rho": 19300, "c": 134},
    "Zinc":        {"k": 116,  "rho": 7140,  "c": 388},
    "Magnesium":   {"k": 156,  "rho": 1740,  "c": 1023},
    "Platinum":    {"k": 71.6, "rho": 21450, "c": 133},
    "Chromium":    {"k": 93.9, "rho": 7190,  "c": 448},
    "Brass":       {"k": 109,  "rho": 8530,  "c": 380},
    "Bronze":      {"k": 60,   "rho": 8800,  "c": 380},
    "StainlessSteel": {"k": 16, "rho": 8000,  "c": 500},
}



# Physical parameters for spatial and temporal terms
R = 0.0             # radial position (m), center of beam
Z = 0.0             # depth (m)
dt = 1e-6           # time step (s)
sigma = 1e-6        # pulse width (s)
t = 0               # current time step index (start of pulse)
delta = 1e-4        # penetration depth (m)
mu_a = 1 / delta    # absorption coefficient (1/m), example

def compute_source_term(power, beam_radius, material):
    # Spatial term (Gaussian in radial direction, exponential in depth)
    spatial_term = np.exp(-2 * R**2 / beam_radius**2) * np.exp(-mu_a * Z)

    # Temporal term (Gaussian temporal profile)
    temporal_term = np.exp(-(t * dt)**2 / (2 * sigma**2))

    rho = material_properties[material]['rho']
    c = material_properties[material]['c']
    I0 = power / (np.pi * beam_radius ** 2)
    return (I0 * mu_a) / (rho * c) * spatial_term * temporal_term

expected_features = model.feature_names_in_
results = []



for power in power_range:
    for beam_radius in beam_radius_range:
        source_term = compute_source_term(power, beam_radius, default_values['Material'])

        features_dict = dict.fromkeys(expected_features, 0.0)
        features_dict['Power_W'] = power
        features_dict['BeamRadius_m'] = beam_radius
        features_dict['Wavelength_m'] = default_values['Wavelength_m']
        features_dict['DomainLength_m'] = default_values['DomainLength_m']
        features_dict['Sigma_s'] = default_values['Sigma_s']
        features_dict['InitialSourceTerm_W_per_kg'] = source_term

        geometry_col = f"Geometry_{default_values['Geometry']}"
        material_col = f"Material_{default_values['Material']}"
        if geometry_col in features_dict:
            features_dict[geometry_col] = 1.0
        if material_col in features_dict:
            features_dict[material_col] = 1.0

        features_df = pd.DataFrame([features_dict])
        pred_log = model.predict(features_df)[0]
        predicted_temp = np.expm1(pred_log)

        results.append({
            'Power_W': power,
            'BeamRadius_m': beam_radius,
            'Wavelength_m': default_values['Wavelength_m'],
            'DomainLength_m': default_values['DomainLength_m'],
            'Sigma_s': default_values['Sigma_s'],
            'Predicted_Max_Temperature_C': predicted_temp
        })

results_df = pd.DataFrame(results)
results_df.to_csv('parameter_sweep_results.csv', index=False)


# Measuring the results error
# Use the original test target and predicted values (already inverse-transformed)
y_test_actual = np.array(y_test_raw)  # true values (non-log)
y_pred_actual = np.array(y_pred)      # predicted values (non-log)

# Create a DataFrame comparing actual vs predicted
comparison_df = pd.DataFrame({
    'Actual MaxTemperature_C': y_test_actual,
    'Predicted MaxTemperature_C': y_pred_actual,
    'Error': y_pred_actual - y_test_actual,
    'Absolute Error': np.abs(y_pred_actual - y_test_actual),
    'Relative Error (%)': 100 * np.abs((y_pred_actual - y_test_actual) / y_test_actual)
})

# Print first 10 rows to check
print(comparison_df.head(10))

# Optional: Save to CSV for further analysis
comparison_df.to_csv('prediction_vs_actual.csv', index=False)

#Plotting the errorrs
from sklearn.metrics import mean_absolute_error, median_absolute_error

# Calculate additional error metrics
mae = mean_absolute_error(comparison_df['Actual MaxTemperature_C'], comparison_df['Predicted MaxTemperature_C'])
med_ae = median_absolute_error(comparison_df['Actual MaxTemperature_C'], comparison_df['Predicted MaxTemperature_C'])


# Plot residuals vs predicted values [side by side]
# Ceaning data to plot [removing any value that that is less then zero for the log fitting]
df = comparison_df[
    (comparison_df['Predicted MaxTemperature_C'] > 0) & 
    (comparison_df['Absolute Error'] > 0)
]

x = np.log10(df['Predicted MaxTemperature_C'])
y = np.log10(df['Absolute Error'])
slope, intercept = np.polyfit(x, y, 1)
line = 10**(intercept + slope * x)

# Cleaned data
min_temp = 1e4
max_temp = 1e8
min_error = 1
max_error = 1e8
#Cleaning data to zoom in between 1e4 and 1e8 degrees and between 1 to 1e8 absulute error to focus on the practical area
cleaned_df = df[
    (df['Predicted MaxTemperature_C'] >= min_temp) &
    (df['Predicted MaxTemperature_C'] <= max_temp) &
    (df['Absolute Error'] >= min_error) &
    (df['Absolute Error'] <= max_error)
]

x_cleaned = np.log10(cleaned_df['Predicted MaxTemperature_C'])
y_cleaned = np.log10(cleaned_df['Absolute Error'])
slope_cleaned, intercept_cleaned = np.polyfit(x_cleaned, y_cleaned, 1)
line_cleaned = 10**(intercept_cleaned + slope_cleaned * x_cleaned)

#Predicted temp vs absulute error log-log scale
plt.figure(figsize=(8, 6))
plt.scatter(df['Predicted MaxTemperature_C'], df['Absolute Error'], alpha=0.5, label='Data')
plt.plot(df['Predicted MaxTemperature_C'], line, color='red', label='Fit: log(Error) = {:.2f} * log(Temp) + C'.format(slope))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Predicted Max Temperature ($^{\circ}C$)')
plt.ylabel('Absolute Error')
plt.title('Absolute Error vs Predicted Max Temp (Full Range)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('Fig3_Error_vs_Temp_FullRange.png')
plt.show()
#Zoomed in practical region
plt.figure(figsize=(8, 6))
plt.scatter(cleaned_df['Predicted MaxTemperature_C'], cleaned_df['Absolute Error'], alpha=0.5, label='Cleaned Data')
plt.plot(cleaned_df['Predicted MaxTemperature_C'], line_cleaned, color='red', label='Fit: log(Error) = {:.2f} * log(Temp) + C'.format(slope_cleaned))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Predicted Max Temperature ($^{\circ}C$)')
plt.ylabel('Absolute Error')
plt.title('Absolute Error vs Predicted Max Temp (10⁴ – 10⁸ °C)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('Fig4_Error_vs_Temp_PrakRegion.png')
plt.show()








# Evaluating the model
print("Model Evaluation: ")
mse = mean_squared_error(y_test_raw, y_pred) # Intense value due to extremes
r2 = r2_score(y_test_raw, y_pred)
print(f'R^2 Score: {r2}')


# Print error (practical region)
mean_ab_error = cleaned_df['Absolute Error'].mean()
print(f"Average absolute error in the 10⁴–10⁸ °C range: {mean_ab_error:.2f}")


# Print first 10 rows to check
print(comparison_df.head(10))