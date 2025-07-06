import requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

def fetch_clinical_data():
    url = "https://api.rootnet.in/covid19-in/stats/history"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("API request failed.")

    data = response.json().get('data', [])
    records = []

    for entry in data:
        summary = entry.get('summary', {})
        indian = summary.get('confirmedCasesIndian')
        foreign = summary.get('confirmedCasesForeign')
        total = summary.get('total')
        if total is None and indian is not None and foreign is not None:
            total = indian + foreign
        if total is None:
            continue
        try:
            date_obj = datetime.strptime(entry['day'], '%Y-%m-%d').date()
        except:
            continue
        records.append({'date': date_obj, 'total_cases': total})

    if not records:
        raise ValueError("No valid data found.")
    df = pd.DataFrame(records)
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def compute_new_cases(df):
    df['new_cases'] = df['total_cases'].diff().fillna(0).astype(int)
    return df

def predict_next_week(df):
    # Use last available data date as cutoff
    cutoff_date = df['date'].max()

    # Filter up to cutoff
    df = df[df['date'] <= cutoff_date].copy()

    if len(df) < 14:
        raise ValueError("Not enough data before cutoff date.")

    recent_df = df[-14:]
    X = np.arange(len(recent_df)).reshape(-1, 1)
    y = recent_df['total_cases'].values
    model = LinearRegression().fit(X, y)

    next_index = [[len(recent_df)]]
    predicted_total = model.predict(next_index)[0]
    predicted_new = predicted_total - recent_df['total_cases'].iloc[-1]

    return (
        df['date'].iloc[0],               # Data start
        df['date'].iloc[-1],              # Data end
        recent_df['date'].iloc[0],        # Training start
        recent_df['date'].iloc[-1],       # Training end
        cutoff_date + timedelta(days=7),  # Prediction end
        int(predicted_total),
        int(predicted_new),
        recent_df
    )

if __name__ == "__main__":
    print("📊 Fetching and processing historical clinical COVID-19 data...\n")
    df = fetch_clinical_data()
    df = compute_new_cases(df)

    data_start, data_end, train_start, train_end, predict_to, pred_total, pred_new, train_df = predict_next_week(df)

    print(f"✅ Data fetched from {data_start} to {data_end}")
    print(f"✅ Used data till {train_end} to train model (last 14 days: {train_start} to {train_end})")
    print(f"🔮 Predicting total cases up to: {predict_to}")

    print("\n=== Last 14 Days Used for Training ===")
    print(train_df.to_string(index=False))

    print("\n=== Prediction ===")
    print(f"Predicted Total Cases by {predict_to}: {pred_total}")
    print(f"Predicted New Cases This Week: {pred_new}")

    train_df.to_csv("clinical_data_until_prediction.csv", index=False)
    print("\n📁 Saved training data to 'clinical_data_until_prediction.csv'")
