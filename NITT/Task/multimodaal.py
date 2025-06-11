import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import random

# -------------------- STEP 1: Reproducibility --------------------
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# -------------------- STEP 2: Load and Preprocess Data --------------------
df = pd.read_csv("owid-covid-data.csv")
df = df[df['location'] == 'India']
df['date'] = pd.to_datetime(df['date'])

# Focused features
covid_df = df[['date', 'new_cases', 'new_deaths', 'new_tests', 'reproduction_rate',
               'hosp_patients', 'icu_patients', 'people_vaccinated', 
               'total_cases', 'total_deaths', 'total_tests', 'total_vaccinations',
               'stringency_index', 'population_density', 'gdp_per_capita',
               'cardiovasc_death_rate', 'diabetes_prevalence',
               'female_smokers', 'male_smokers',
               'hospital_beds_per_thousand', 'life_expectancy',
               'human_development_index', 'median_age']]

# Simulated climate data
df['temperature'] = np.random.uniform(20, 40, len(df))
df['humidity'] = np.random.uniform(30, 90, len(df))
df['precipitation'] = np.random.uniform(0, 100, len(df))
df['wind_speed'] = np.random.uniform(0, 20, len(df))

climate_df = df[['date', 'temperature', 'humidity', 'precipitation', 'wind_speed']].dropna()
data = pd.merge(covid_df, climate_df, on='date', how='inner')
data.set_index('date', inplace=True)

# -------------------- STEP 3: Weekly Aggregation --------------------
weekly = data.resample('W').mean()
weekly = weekly.ffill()

# -------------------- STEP 4: Train-test Split --------------------
train = weekly.loc[:'2023-11-30']
test = weekly.loc['2023-12-01':]

# -------------------- STEP 5: ARIMA Model --------------------
def fit_arima(series, order=(3, 3, 4)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    return forecast

y_train = train['new_cases']
y_test = test['new_cases']
arima_pred = fit_arima(y_train)

# -------------------- STEP 6: Climate Feature Input for DL --------------------
clim_train = train[['temperature', 'humidity', 'precipitation', 'wind_speed']]
clim_test = test[['temperature', 'humidity', 'precipitation', 'wind_speed']]

scaler = MinMaxScaler()
clim_train_scaled = scaler.fit_transform(clim_train)
clim_test_scaled = scaler.transform(clim_test)

X_train_cnn = clim_train_scaled.reshape((clim_train_scaled.shape[0], clim_train_scaled.shape[1], 1))
X_test_cnn = clim_test_scaled.reshape((clim_test_scaled.shape[0], clim_test_scaled.shape[1], 1))

# -------------------- STEP 7: CNN + GRU Model --------------------
model = Sequential([
    Input(shape=(4, 1)),
    Conv1D(32, kernel_size=2, activation='relu', padding='same'),
    GRU(32),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

early_stop = EarlyStopping(patience=10, restore_best_weights=True)

model.fit(X_train_cnn, y_train.values, 
          epochs=100, batch_size=4, verbose=0, 
          validation_split=0.2, callbacks=[early_stop])

# -------------------- STEP 8: Final Fusion --------------------
cnn_preds = model.predict(X_test_cnn).flatten()
final_preds = (cnn_preds + arima_pred.values) / 2

# -------------------- STEP 9: Evaluation --------------------
mae = mean_absolute_error(y_test, final_preds)
rmse = np.sqrt(mean_squared_error(y_test, final_preds))
r2 = r2_score(y_test, final_preds)

cnn_mae = mean_absolute_error(y_test, cnn_preds)
arima_mae = mean_absolute_error(y_test, arima_pred)

print(f"Best ARIMA Model Order: (3, 3, 4)")
print(f"ARIMA MAE: {arima_mae:.2f}")
print(f"CNN+GRU MAE: {cnn_mae:.2f}")
print(f"Fused MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# -------------------- STEP 10: Plot --------------------
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, label='Actual', linewidth=2)
plt.plot(y_test.index, final_preds, label='ARIMA + Climate CNN-GRU Fusion', linestyle='--')
plt.xlabel("Date")
plt.ylabel("Weekly COVID Cases")
plt.title("Prediction vs Actual - ARIMA + Climate CNN-GRU")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fusion_model_prediction_graph.png")
plt.show()
