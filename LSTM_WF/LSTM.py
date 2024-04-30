import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from colorama import Fore, Style
from tqdm import tqdm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import psutil
import time

process = psutil.Process()
memory_info = process.memory_info()
EPOCHS = 100

data = pd.read_csv('/Users/pranavsharma/Downloads/data.csv', encoding='ISO-8859-1')
data = data.dropna()
data = data.drop(columns=['Date Time'])
print(Fore.RED + "Data Imported. Shape: ", data.shape)

X = data[['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
       'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
       'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
       'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR (W/m²)',
       'PAR (µmol/m²/s)', 'max. PAR (µmol/m²/s)', 'Tlog (degC)', 'CO2 (ppm)']]
y = data[['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
       'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
       'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
       'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR (W/m²)',
       'PAR (µmol/m²/s)', 'max. PAR (µmol/m²/s)', 'Tlog (degC)', 'CO2 (ppm)']]

one_day = 96
days = one_day * 1
y = y.shift(-days)

X = X[:-days]
y = y[:-days]
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print("Preprocessed Data.")


model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(LSTM(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='linear'))
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy', 'mape', 'mae'])
print("Model built. Starting training...")
print(Style.RESET_ALL)
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=64, validation_split=0.2, verbose=1)
print(Fore.RED + "Model trained")
print(Style.RESET_ALL)
print(Fore.RED + "Testing Model...")


predicted_values = model.predict(X_test)
actual_values = y_test

print(Fore.GREEN + "Results(Non Auto-Regressive):")
mse = np.mean((predicted_values - actual_values) ** 2)
mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
mae = np.mean(np.abs(actual_values - predicted_values))
r2 = r2_score(actual_values, predicted_values)
smape = np.mean(np.abs(actual_values - predicted_values) / (np.abs(actual_values) + np.abs(predicted_values)) * 100)

print(Fore.GREEN + "Mean Squared Error: ", mse)
print(Fore.GREEN + "Mean Absolute Percentage Error: ", mape)
print(Fore.GREEN + "Mean Absolute Error: ", mae)
print(Fore.GREEN + "R2 Score Auto-Regressively: ", r2)
print(Fore.GREEN + "Symmetric Mean Absolute Percentage Error: ", smape)

predicted_values = []
actual_values = y_test

print(Fore.RED + "Training Model Auto-Regressively...")
print(Fore.CYAN)

one_x = np.array([X_test[0]])

for i in tqdm(range(0, len(X_test))):
    y_pred = model.predict(one_x, verbose=0)
    predicted_values.append(y_pred[0])  
    one_x = y_pred

print(Fore.RED + "Model Auto-Regressively Trained.")
print(Fore.GREEN + "Results:")
mse = np.mean((predicted_values - actual_values) ** 2)
mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
mae = np.mean(np.abs(actual_values - predicted_values))
r2 = r2_score(actual_values, predicted_values)
smape = np.mean(np.abs(actual_values - predicted_values) / (np.abs(actual_values) + np.abs(predicted_values)) * 100)

print(Fore.GREEN + "Mean Squared Error: ", mse)
print(Fore.GREEN + "Mean Absolute Percentage Error: ", mape)
print(Fore.GREEN + "Mean Absolute Error: ", mae)
print(Fore.GREEN + "R2 Score Auto-Regressively: ", r2)
print(Fore.GREEN + "Symmetric Mean Absolute Percentage Error: ", smape)


X_train_temperature = X_train[:, 1]
predicted_values_temperature = np.array(predicted_values)[:, 1]

plt.plot(X_train_temperature, color='blue')
plt.plot(range(len(X_train_temperature), len(X_train_temperature) + len(predicted_values_temperature), 1), predicted_values_temperature, color='red')
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

ax1.plot(history.history['loss'])
ax1.plot(history.history['val_loss'])
ax1.set_title('MSE Vs EPOCH for LSTM')
ax1.set_ylabel('MSE')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Validation'], loc='upper left')

ax2.plot(history.history['mape'])
ax2.plot(history.history['val_mape'])
ax2.set_title('MAPE Vs EPOCH for LSTM')
ax2.set_ylabel('MAPE')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Validation'], loc='upper left')

ax3.plot(history.history['mae'])
ax3.plot(history.history['val_mae'])
ax3.set_title('MAE Vs EPOCH for LSTM')
ax3.set_ylabel('MAE')
ax3.set_xlabel('Epoch')
ax3.legend(['Train', 'Validation'], loc='upper left')
plt.subplots_adjust(wspace=0.5)
plt.show()

###### Final Plots #########
ground_truth_values = np.array(actual_values)
forecast_prediction_values = np.array(predicted_values)
every_10th_ground_truth = ground_truth_values[:100]
every_10th_forecast_prediction = forecast_prediction_values[:100]
features = {
    'p (mbar)': 0,
    'T (degC)': 1,
    'rh (%)': 2,
    'rho (g/m**3)': 3,
    'rain (mm)': 4,
    'CO2 (ppm)': 5
}
time_array = np.arange(every_10th_ground_truth.shape[0])
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axs = axs.flatten()
for i, (key, value) in enumerate(features.items()):
    axs[i].plot(time_array, every_10th_ground_truth[:, value], 'b-', label='Ground Truth')
    axs[i].plot(time_array, every_10th_forecast_prediction[:, value], 'b--', color='orange', label='Forecast Prediction')
    axs[i].set_title(f'Feature ({key})')
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('Feature Value')
    axs[i].legend()
plt.tight_layout()
plt.show()

mse_p_mbar = np.mean((forecast_prediction_values[:, 0] - ground_truth_values[:, 0]) ** 2)
mse_T_degC = np.mean((forecast_prediction_values[:, 1] - ground_truth_values[:, 1]) ** 2)
mse_rh = np.mean((forecast_prediction_values[:, 4] - ground_truth_values[:, 4]) ** 2)
mse_rho = np.mean((forecast_prediction_values[:, 10] - ground_truth_values[:, 10]) ** 2)
mse_rain_mm = np.mean((forecast_prediction_values[:, 14] - ground_truth_values[:, 14]) ** 2)
mse_CO2_ppm = np.mean((forecast_prediction_values[:, 20] - ground_truth_values[:, 20]) ** 2)

print("MSE for p (mbar):", mse_p_mbar)
print("MSE for T (degC):", mse_T_degC)
print("MSE for rh (%):", mse_rh)
print("MSE for rho (g/m**3):", mse_rho)
print("MSE for rain (mm):", mse_rain_mm)
print("MSE for CO2 (ppm):", mse_CO2_ppm)

mse_values = {
    "p (mbar)": [],
    "T (degC)": [],
    "rh (%)": [],
    "rho (g/m**3)": [],
    "rain (mm)": [],
    "CO2 (ppm)": []
}

for epoch in range(EPOCHS):
    mse_values["p (mbar)"].append(np.mean((forecast_prediction_values[epoch, 0] - ground_truth_values[epoch, 0]) ** 2))
    mse_values["T (degC)"].append(np.mean((forecast_prediction_values[epoch, 1] - ground_truth_values[epoch, 1]) ** 2))
    mse_values["rh (%)"].append(np.mean((forecast_prediction_values[epoch, 4] - ground_truth_values[epoch, 4]) ** 2))
    mse_values["rho (g/m**3)"].append(np.mean((forecast_prediction_values[epoch, 10] - ground_truth_values[epoch, 10]) ** 2))
    mse_values["rain (mm)"].append(np.mean((forecast_prediction_values[epoch, 14] - ground_truth_values[epoch, 14]) ** 2))
    mse_values["CO2 (ppm)"].append(np.mean((forecast_prediction_values[epoch, 20] - ground_truth_values[epoch, 20]) ** 2))

epochs = np.arange(1, EPOCHS + 1)

fig, axs = plt.subplots(3, 2, figsize=(12, 10))
axs = axs.flatten()

for i, (variable, mse) in enumerate(mse_values.items()):
    ax = axs[i]
    ax.plot(epochs, mse, color='b', marker='o', linestyle='-')
    ax.set_title("MSE vs Epoch for " + variable)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True)

plt.tight_layout()
plt.show()

input_dim = X_train.shape[1]
units = [128, 256, 128, 64, 64]
num_dense_layers = 5
output_dim = y.shape[1]

lstm_params = 0
for i in range(3):
    lstm_params += (input_dim * units[i] * 4) + (units[i] * units[i] * 4) + (units[i] * 4) + (units[i] * 4)

dense_params = 0
for i in range(num_dense_layers):
    if i == 0:
        dense_params += (units[-1] * output_dim) + output_dim
    else:
        dense_params += (units[i] * units[i-1]) + units[i]

total_params = lstm_params + dense_params
print(f"Used Memory: {memory_info.rss / (1024 * 1024 * 1024):.2f} GB")
