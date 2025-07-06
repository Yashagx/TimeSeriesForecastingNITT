import requests
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GRU, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler

# Weather API Key
API_KEY = "bd5e378503939ddaee76f12ad7a97608"

# States to analyze
states = ["Maharashtra", "Delhi", "Tamil Nadu", "Karnataka", "West Bengal"]

# Dummy coordinates for states (latitude, longitude)
state_coords = {
    "Maharashtra": (19.7515, 75.7139),
    "Delhi": (28.6139, 77.2090),
    "Tamil Nadu": (11.1271, 78.6569),
    "Karnataka": (15.3173, 75.7139),
    "West Bengal": (22.9868, 87.8550)
}

# Load historical data (must have: infected, recovered, deaths, temp, humidity)
df = pd.read_csv("historical_climate_covid_india.csv")
df = df.ffill().fillna(0)

# Normalize only relevant features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['infected', 'recovered', 'deaths', 'temp', 'humidity']])

X = []
y = []
seq_length = 10  # number of past days to use for prediction

for i in range(len(scaled_data) - seq_length):
    X.append(scaled_data[i:i + seq_length])
    y.append(scaled_data[i + seq_length][:3])  # target: infected, recovered, deaths

X = np.array(X)
y = np.array(y)

# Define the CNN+GRU model
model = Sequential([
    Input(shape=(seq_length, 5)),  # 5 features: infected, recovered, deaths, temp, humidity
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    GRU(64),
    Dropout(0.2),
    Dense(3)  # Predict infected, recovered, deaths
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# Fetch live weather
def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url).json()
    return {
        "temp": response["main"]["temp"],
        "humidity": response["main"]["humidity"]
    }

# Predict for each state
results = []
for state in states:
    print(f"Processing {state}...")
    lat, lon = state_coords[state]
    weather = get_weather(lat, lon)

    # Use last 10 days from data and update climate features
    latest_seq = scaled_data[-seq_length:].copy()
    latest_seq[:, 3] = weather["temp"]
    latest_seq[:, 4] = weather["humidity"]

    latest_seq = np.expand_dims(latest_seq, axis=0)
    y_pred_scaled = model.predict(latest_seq)[0]
    y_pred_scaled = np.maximum(y_pred_scaled, 0)  # No negative values

    # Reverse scale prediction
    dummy_input = np.zeros((1, 5))
    dummy_input[0, :3] = y_pred_scaled
    unscaled = scaler.inverse_transform(dummy_input)[0][:3]

    results.append({
        "State": state,
        "Predicted_Infected": int(unscaled[0]),
        "Predicted_Recovered": int(unscaled[1]),
        "Predicted_Deaths": int(unscaled[2])
    })

# Save and display results
result_df = pd.DataFrame(results)
print("\n=== Predictions ===")
print(result_df)
result_df.to_csv("covid_predictions_statewise.csv", index=False)
print("\nSaved results to covid_predictions_statewise.csv")
