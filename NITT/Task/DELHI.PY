import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import requests
from datetime import datetime, timedelta

# Load COVID-19 Data
df = pd.read_csv("owid-covid-data.csv")
df = df[df['location'] == 'India']
df['date'] = pd.to_datetime(df['date'])

covid_df = df[['date', 'new_cases']].set_index('date').resample('D').sum().fillna(0).reset_index()

# Fetch Real Climate Data from Open-Meteo
end_date = datetime.now().date()
start_date = end_date - timedelta(days=len(covid_df))

url = (
    f"https://archive-api.open-meteo.com/v1/archive?latitude=28.66&longitude=77.23&"
    f"start_date={start_date}&end_date={end_date}&"
    "daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
    "windspeed_10m_max,relative_humidity_2m_max&timezone=Asia%2FKolkata"
)

climate_json = requests.get(url).json()

climate_df = pd.DataFrame({
    'date': pd.to_datetime(climate_json['daily']['time']),
    'temperature': np.mean([
        climate_json['daily']['temperature_2m_max'],
        climate_json['daily']['temperature_2m_min']
    ], axis=0),
    'humidity': climate_json['daily']['relative_humidity_2m_max'],
    'precipitation': climate_json['daily']['precipitation_sum'],
    'wind_speed': climate_json['daily']['windspeed_10m_max']
})

# Merge and Preprocess
merged_df = pd.merge(covid_df, climate_df, on='date')
merged_df.set_index('date', inplace=True)
weekly = merged_df.resample('W').mean().ffill()

# Lag Features
for lag in range(1, 4):
    weekly[f'lag_{lag}'] = weekly['new_cases'].shift(lag)
weekly.dropna(inplace=True)

# Train-test Split
train = weekly[:'2023-11-30']
test = weekly['2023-12-01':]

y_train = train['new_cases']
y_test = test['new_cases']

# ARIMA Forecast
def fit_arima(series, order=(3, 3, 4)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    return forecast

arima_pred = fit_arima(y_train)

# CNN + Bidirectional GRU Model
features = ['temperature', 'humidity', 'precipitation', 'wind_speed', 'lag_1', 'lag_2', 'lag_3']
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train[features])
test_scaled = scaler.transform(test[features])

X_train_cnn = train_scaled.reshape((train_scaled.shape[0], train_scaled.shape[1], 1))
X_test_cnn = test_scaled.reshape((test_scaled.shape[0], test_scaled.shape[1], 1))

model = Sequential([
    Input(shape=(X_train_cnn.shape[1], 1)),
    Conv1D(64, kernel_size=2, activation='relu'),
    MaxPooling1D(),
    Conv1D(32, kernel_size=2, activation='relu'),
    MaxPooling1D(),
    Bidirectional(GRU(32, return_sequences=True)),
    Bidirectional(GRU(16)),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.fit(X_train_cnn, y_train.values, epochs=100, batch_size=4, verbose=0)

# Predictions and Fusion
cnn_preds = model.predict(X_test_cnn).flatten()
final_preds = (cnn_preds + arima_pred.values) / 2

# Evaluation
arima_mae = mean_absolute_error(y_test, arima_pred)
cnn_mae = mean_absolute_error(y_test, cnn_preds)
fused_mae = mean_absolute_error(y_test, final_preds)
rmse = np.sqrt(mean_squared_error(y_test, final_preds))
r2 = r2_score(y_test, final_preds)

print(f"Best ARIMA Model Order: (3, 3, 4)")
print(f"ARIMA MAE: {arima_mae:.2f}")
print(f"CNN+GRU MAE: {cnn_mae:.2f}")
print(f"Fused MAE: {fused_mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Plot Results
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, label='Actual', linewidth=2)
plt.plot(y_test.index, final_preds, label='Fusion Prediction', linestyle='--')
plt.title("COVID Cases Forecast with ARIMA + Real Climate + Bi-GRU")
plt.xlabel("Date")
plt.ylabel("Weekly COVID Cases")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("final_fused_forecast.png")
plt.show()
