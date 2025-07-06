import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Input
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==== Step 1: Load & Resample to Weekly ====
df = pd.read_csv("owid-covid-data.csv", parse_dates=['date'])
df = df[df['location'] == 'India'][['date', 'total_cases', 'new_cases', 'total_deaths']]
df.dropna(inplace=True)
df.set_index('date', inplace=True)

weekly_df = pd.DataFrame()
weekly_df['total_cases'] = df['total_cases'].resample('W').last()
weekly_df['new_cases'] = df['new_cases'].resample('W').sum()
weekly_df['total_deaths'] = df['total_deaths'].resample('W').last()
weekly_df.dropna(inplace=True)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(weekly_df)

def create_dataset(dataset, look_back=4):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back])
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)

def objective(weights, y_true, preds):
    weights = np.array(weights)
    if np.sum(weights) == 0 or np.any(np.isnan(weights)):
        return 1e9
    weights = weights / np.sum(weights)
    ensemble = np.tensordot(weights, preds, axes=1)
    if np.any(np.isnan(ensemble)):
        return 1e9
    return mean_squared_error(y_true, ensemble)

def scwoa_optimize(preds, y_true, dim=2, pop_size=20, max_iter=50):
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
            fitness = objective(Xp, y_true, preds)
            if fitness < best_score:
                best_score = fitness
                best_weights = Xp
    return best_weights / np.sum(best_weights)

# ==== Step 3: Create output folders ====
output_dir = "prediction_outputs"
os.makedirs(output_dir, exist_ok=True)

summary_data = []

# ==== Step 4: Evaluate for Each Target ====
target_indices = {'total_cases': 0, 'new_cases': 1, 'total_deaths': 2}
look_back = 4
epoch = 30
lr = 0.01

for target_name, idx in target_indices.items():
    print(f"\n🔍 Target: {target_name}")

    X, y_all = create_dataset(data_scaled, look_back)
    y = y_all[:, idx]

    train_size = int(len(X) * 0.7)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # ==== BPNN ====
    bpnn = MLPRegressor(hidden_layer_sizes=(64,), max_iter=500)
    bpnn.fit(X_train.reshape(len(X_train), -1), y_train)
    bpnn_pred = bpnn.predict(X_test.reshape(len(X_test), -1))

    # ==== BPNN + Elman ====
    elman = Sequential([
        Input(shape=(look_back, 3)),
        SimpleRNN(32, activation='tanh'),
        Dense(1)
    ])
    elman.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    elman.fit(X_train, y_train, epochs=epoch, batch_size=8, verbose=0)
    elman_pred = elman.predict(X_test).flatten()

    preds_elman = np.stack([bpnn_pred, elman_pred], axis=0)
    best_weights_elman = scwoa_optimize(preds_elman, y_test)
    final_pred_elman = np.tensordot(best_weights_elman, preds_elman, axes=1)

    dummy = np.zeros((len(y_test), data_scaled.shape[1]))
    dummy[:, idx] = y_test
    actual_orig = scaler.inverse_transform(dummy)[:, idx]
    dummy[:, idx] = final_pred_elman
    pred_orig_elman = scaler.inverse_transform(dummy)[:, idx]

    df_elman = pd.DataFrame({'Actual': actual_orig, 'Predicted': pred_orig_elman})
    path_elman = os.path.join(output_dir, f"weekly_actual_vs_predicted_{target_name}_bpnn_elman.csv")
    df_elman.to_csv(path_elman, index=False)

    rmse_elman = np.sqrt(mean_squared_error(actual_orig, pred_orig_elman))
    mae_elman = mean_absolute_error(actual_orig, pred_orig_elman)
    summary_data.append({
        'Target': target_name,
        'Model': 'BPNN + Elman',
        'RMSE': rmse_elman,
        'MAE': mae_elman,
        'Weight_BPNN': best_weights_elman[0],
        'Weight_RNN': best_weights_elman[1]
    })
    print(f"✅ BPNN + Elman → {path_elman}")

    # ==== BPNN + LSTM ====
    lstm = Sequential([
        Input(shape=(look_back, 3)),
        LSTM(32, activation='relu'),
        Dense(1)
    ])
    lstm.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    lstm.fit(X_train, y_train, epochs=epoch, batch_size=8, verbose=0)
    lstm_pred = lstm.predict(X_test).flatten()

    preds_lstm = np.stack([bpnn_pred, lstm_pred], axis=0)
    best_weights_lstm = scwoa_optimize(preds_lstm, y_test)
    final_pred_lstm = np.tensordot(best_weights_lstm, preds_lstm, axes=1)

    dummy[:, idx] = final_pred_lstm
    pred_orig_lstm = scaler.inverse_transform(dummy)[:, idx]

    df_lstm = pd.DataFrame({'Actual': actual_orig, 'Predicted': pred_orig_lstm})
    path_lstm = os.path.join(output_dir, f"weekly_actual_vs_predicted_{target_name}_bpnn_lstm.csv")
    df_lstm.to_csv(path_lstm, index=False)

    rmse_lstm = np.sqrt(mean_squared_error(actual_orig, pred_orig_lstm))
    mae_lstm = mean_absolute_error(actual_orig, pred_orig_lstm)
    summary_data.append({
        'Target': target_name,
        'Model': 'BPNN + LSTM',
        'RMSE': rmse_lstm,
        'MAE': mae_lstm,
        'Weight_BPNN': best_weights_lstm[0],
        'Weight_RNN': best_weights_lstm[1]
    })
    print(f"✅ BPNN + LSTM → {path_lstm}")

# ==== Step 5: Save Summary ====
summary_df = pd.DataFrame(summary_data)
summary_path = os.path.join(output_dir, "model_summary_results.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n📊 Summary saved to: {summary_path}")
