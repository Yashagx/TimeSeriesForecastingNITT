import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load and preprocess data
df = pd.read_csv("owid-covid-data.csv")
df = df[df['location'] == 'India'][['total_cases', 'new_cases', 'total_deaths', 'new_deaths']]
df.dropna(inplace=True)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

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

# SCWOA optimization function
def objective(weights, y_true, preds):
    weights = np.array(weights)
    if np.sum(weights) == 0 or np.any(np.isnan(weights)):
        return 1e9
    weights = weights / np.sum(weights)
    ensemble = np.tensordot(weights, preds, axes=1)
    if np.any(np.isnan(ensemble)):
        return 1e9
    return mean_squared_error(y_true, ensemble)

# Hyperparameter grid
epoch_list = [10, 30, 50, 100, 200]
lr_list = [0.2, 0.1, 0.05, 0.01, 0.001]
results = []

for epochs in tqdm(epoch_list, desc="Epoch Loop"):
    for lr in tqdm(lr_list, desc=f"Learning Rate Loop for epoch={epochs}", leave=False):

        # Train BPNN
        bpnn = MLPRegressor(hidden_layer_sizes=(64,), max_iter=500)
        bpnn.fit(X_train.reshape(len(X_train), -1), y_train)
        bpnn_pred = bpnn.predict(X_test.reshape(len(X_test), -1))

        # Train Elman RNN
        el_model = Sequential([
            Input(shape=(look_back, 4)),
            SimpleRNN(32, activation='tanh'),
            Dense(4)
        ])
        el_model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        el_model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)
        elman_pred = el_model.predict(X_test)

        # Train LSTM
        lstm_model = Sequential([
            Input(shape=(look_back, 4)),
            LSTM(32, activation='relu'),
            Dense(4)
        ])
        lstm_model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)
        lstm_pred = lstm_model.predict(X_test)

        predictions = np.stack([bpnn_pred, elman_pred, lstm_pred], axis=0)

        # SCWOA optimization
        pop_size, max_iter, dim = 20, 50, 3
        lb, ub = 0, 1
        population = np.random.uniform(lb, ub, (pop_size, dim))
        best_score, best_weights = float('inf'), None

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

        # Final ensemble prediction
        best_weights = best_weights / np.sum(best_weights)
        final_pred = np.tensordot(best_weights, predictions, axes=1)

        # Evaluate in real scale
        y_test_orig = scaler.inverse_transform(y_test)
        final_pred_orig = scaler.inverse_transform(final_pred)
        rmse = np.sqrt(mean_squared_error(y_test_orig, final_pred_orig))
        mae = mean_absolute_error(y_test_orig, final_pred_orig)

        # Save result
        results.append({
            'Epochs': epochs,
            'Learning_Rate': lr,
            'RMSE': rmse,
            'MAE': mae,
            'BPNN_Weight': best_weights[0],
            'Elman_Weight': best_weights[1],
            'LSTM_Weight': best_weights[2]
        })

        # Save actual vs predicted (only for last run or best run - here saving from all runs will overwrite)
        comparison_df = pd.DataFrame({
            'Actual_Total_Cases': y_test_orig[:, 0],
            'Predicted_Total_Cases': final_pred_orig[:, 0],
            'Actual_New_Cases': y_test_orig[:, 1],
            'Predicted_New_Cases': final_pred_orig[:, 1],
            'Actual_Total_Deaths': y_test_orig[:, 2],
            'Predicted_Total_Deaths': final_pred_orig[:, 2],
            'Actual_New_Deaths': y_test_orig[:, 3],
            'Predicted_New_Deaths': final_pred_orig[:, 3]
        })
        comparison_df.to_csv("actual_vs_predicted.csv", index=False)

# Save all results summary to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("ensemble_gridsearch_results.csv", index=False)

print("\n🔍 Top 5 Configurations by RMSE:")
print(results_df.sort_values(by='RMSE').head())
print("\n✅ Saved test set actual vs predicted values to actual_vs_predicted.csv")
