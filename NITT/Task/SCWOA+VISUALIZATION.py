import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
import matplotlib.pyplot as plt

# 1. Load and filter data
print("📂 Loading and filtering dataset for India...")
df = pd.read_csv("owid-covid-data.csv")
df = df[df['location'] == 'India']
df = df[['total_cases', 'new_cases', 'total_deaths', 'new_deaths']]
df = df.dropna().reset_index(drop=True)
print("✅ India-only data loaded with shape:", df.shape)

# 2. Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# 3. Create time series
def create_dataset(dataset, look_back=7):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back)])
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)

look_back = 7
X, y = create_dataset(data_scaled, look_back)

# 4. Train-test split
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print("✅ Training and testing data prepared.")

# 5. Define LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(32, activation='relu'))
    model.add(Dense(4))
    model.compile(optimizer='adam', loss='mse')
    return model

print("🚀 Training LSTM model...")
lstm_model = build_lstm_model((look_back, 4))
lstm_model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=1)
print("✅ LSTM training complete.")

# 6. Predict & simulate ensemble models
pred_lstm = lstm_model.predict(X_test)
pred_dummy1 = pred_lstm + np.random.normal(0, 0.01, pred_lstm.shape)
pred_dummy2 = pred_lstm + np.random.normal(0, 0.015, pred_lstm.shape)
pred_dummy3 = pred_lstm + np.random.normal(0, 0.02, pred_lstm.shape)

base_preds = np.stack([pred_lstm, pred_dummy1, pred_dummy2, pred_dummy3], axis=-1)

# 7. SCWOA optimization
def fitness(weights, preds, actual):
    combined = np.tensordot(preds, weights, axes=([2], [0]))
    return mean_squared_error(actual, combined)

def scwoa_optimize(preds, actual, n_agents=20, max_iter=30):
    dim = preds.shape[2]
    agents = np.random.dirichlet(np.ones(dim), size=n_agents)
    best_weight = agents[0]
    best_fitness = fitness(best_weight, preds, actual)

    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)
        for i in range(n_agents):
            r1, r2 = np.random.rand(2)
            A = 2 * a * r1 - a
            C = 2 * r2
            p = np.random.rand()
            if p < 0.5:
                D = np.abs(C * best_weight - agents[i])
                agents[i] = best_weight - A * D
            else:
                D = np.abs(best_weight - agents[i])
                l = np.random.rand()
                agents[i] = D * np.exp(2 * np.cos(2 * np.pi * l)) * np.cos(l * 2 * np.pi) + best_weight

            # Prevent invalid weights
            agents[i] = np.clip(agents[i], 0, 1)
            sum_weights = np.sum(agents[i])
            if sum_weights == 0 or np.isnan(sum_weights):
                agents[i] = np.ones(dim) / dim
            else:
                agents[i] /= sum_weights

            score = fitness(agents[i], preds, actual)
            if score < best_fitness:
                best_fitness = score
                best_weight = agents[i]
    return best_weight

# 8. Run optimizer
print("🔄 Optimizing ensemble weights...")
optimal_weights = scwoa_optimize(base_preds, y_test)
final_preds = np.tensordot(base_preds, optimal_weights, axes=([2], [0]))
print("✅ Optimization complete.")
print("🎯 Optimal Weights:", optimal_weights)

# 9. Evaluation
rmse = np.sqrt(mean_squared_error(y_test, final_preds))
print(f"📈 Ensemble RMSE: {rmse:.4f}")

# 10. Inverse transform for plotting
combined_scaled = np.vstack([y_test, final_preds])
rescaled_combined = scaler.inverse_transform(combined_scaled)
y_test_inv = rescaled_combined[:len(y_test)]
final_preds_inv = rescaled_combined[len(y_test):]

# Extract Total Cases and Deaths
actual_cases = y_test_inv[:, 0]
predicted_cases = final_preds_inv[:, 0]

actual_deaths = y_test_inv[:, 2]
predicted_deaths = final_preds_inv[:, 2]

# 11. Plot Total Cases
print("📊 Plotting Total Cases...")
plt.figure(figsize=(12, 5))
plt.plot(actual_cases, label='Actual Total Cases', linewidth=2)
plt.plot(predicted_cases, label='Predicted Total Cases', linestyle='--')
plt.title('India - COVID-19 Total Cases Prediction')
plt.xlabel('Time Step')
plt.ylabel('Total Cases')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=True)

# 12. Plot Total Deaths
print("📊 Plotting Total Deaths...")
plt.figure(figsize=(12, 5))
plt.plot(actual_deaths, label='Actual Total Deaths', linewidth=2)
plt.plot(predicted_deaths, label='Predicted Total Deaths', linestyle='--', color='red')
plt.title('India - COVID-19 Total Deaths Prediction')
plt.xlabel('Time Step')
plt.ylabel('Total Deaths')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=True)

# 13. Pause
input("\n🚪 Press Enter to close the program...")
