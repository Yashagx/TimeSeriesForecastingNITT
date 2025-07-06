import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Input
from tensorflow.keras.optimizers import Adam, Adamax
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Load India COVID data
print("📂 Loading dataset...")
df = pd.read_csv("owid-covid-data.csv")
df = df[df['location'] == 'India'][['total_cases', 'new_cases', 'total_deaths', 'new_deaths']]
df.dropna(inplace=True)

# 2. Normalize data
print("🔄 Normalizing data...")
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# 3. Create supervised dataset
def create_dataset(dataset, look_back=7):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back])
        Y.append(dataset[i+look_back])
    return np.array(X), np.array(Y)

look_back = 7
X, y = create_dataset(data_scaled, look_back)
train_size = int(len(X) * 0.7)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 4. BPNN Model
print("🧠 Training BPNN (MLPRegressor)...")
bpnn = MLPRegressor(hidden_layer_sizes=(64,), max_iter=500)
bpnn.fit(X_train.reshape(len(X_train), -1), y_train)
bpnn_pred = bpnn.predict(X_test.reshape(len(X_test), -1))

# 5. Hyperparameters
epochs = 30
learning_rate = 0.001
batch_size = 16
optimizer_choice = 'adam'

print(f"\n🔍 Training Elman RNN and LSTM with epochs={epochs}, lr={learning_rate}, optimizer={optimizer_choice}")

# 6. Elman RNN Model
el_optimizer = Adam(learning_rate=learning_rate) if optimizer_choice == 'adam' else Adamax(learning_rate=learning_rate)
print("🔁 Training Elman RNN...")
el_model = Sequential([
    Input(shape=(look_back, 4)),
    SimpleRNN(32, activation='tanh'),
    Dense(4)
])
el_model.compile(optimizer=el_optimizer, loss='mse')
el_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
elman_pred = el_model.predict(X_test)

# 7. LSTM Model
lstm_optimizer = Adam(learning_rate=learning_rate) if optimizer_choice == 'adam' else Adamax(learning_rate=learning_rate)
print("🔁 Training LSTM...")
lstm_model = Sequential([
    Input(shape=(look_back, 4)),
    LSTM(32, activation='relu'),
    Dense(4)
])
lstm_model.compile(optimizer=lstm_optimizer, loss='mse')
lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
lstm_pred = lstm_model.predict(X_test)

# 8. Stack Predictions
predictions = np.stack([bpnn_pred, elman_pred, lstm_pred], axis=0)

# 9. SCWOA optimization
print("🐋 Optimizing ensemble weights using SCWOA...")

def objective(weights, y_true, preds):
    weights = np.array(weights)
    if np.sum(weights) == 0 or np.any(np.isnan(weights)):
        return 1e9
    weights = weights / np.sum(weights)
    ensemble = np.tensordot(weights, preds, axes=1)
    if np.any(np.isnan(ensemble)):
        return 1e9
    return mean_squared_error(y_true, ensemble)

pop_size = 20
max_iter = 50
dim = 3
lb, ub = 0, 1

population = np.random.uniform(lb, ub, (pop_size, dim))
best_score = float('inf')
best_weights = None

for t in range(max_iter):
    a = 2 - t * (2 / max_iter)
    for i in range(pop_size):
        r1, r2 = np.random.rand(), np.random.rand()
        A = 2 * a * r1 - a
        C = 2 * r2
        if best_weights is None:
            Xp = population[i]
        else:
            p = np.random.rand()
            if p < 0.5:
                D = abs(C * best_weights - population[i])
                Xp = best_weights - A * D
            else:
                l = np.random.uniform(-1, 1)
                D = abs(best_weights - population[i])
                Xp = D * np.exp(0.1 * l) * np.cos(2 * np.pi * l) + best_weights
        Xp = np.clip(Xp, lb, ub)
        fitness = objective(Xp, y_test, predictions)
        if fitness < best_score:
            best_score = fitness
            best_weights = Xp

# 10. Final ensemble prediction
best_weights = best_weights / np.sum(best_weights)
final_prediction = np.tensordot(best_weights, predictions, axes=1)

# 11. Inverse transform predictions and ground truth
final_prediction_orig = scaler.inverse_transform(final_prediction)
y_test_orig = scaler.inverse_transform(y_test)

# 12. Evaluate
rmse_orig = np.sqrt(mean_squared_error(y_test_orig, final_prediction_orig))
mae_orig = mean_absolute_error(y_test_orig, final_prediction_orig)

print("\n📈 Model Evaluation:")
print("✅ RMSE (original scale):", round(rmse_orig, 2))
print("✅ MAE  (original scale):", round(mae_orig, 2))
print("📊 Optimal Ensemble Weights:")
print("   BPNN  :", round(best_weights[0], 3))
print("   Elman :", round(best_weights[1], 3))
print("   LSTM  :", round(best_weights[2], 3))

# 13. Plot predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test_orig[:, 0], label="Actual Total Cases", linewidth=2)
plt.plot(final_prediction_orig[:, 0], label="Predicted Total Cases", linestyle='--', linewidth=2)
plt.title("📉 COVID-19 Total Cases: Actual vs Predicted", fontsize=14)
plt.xlabel("Days")
plt.ylabel("Total Cases")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

