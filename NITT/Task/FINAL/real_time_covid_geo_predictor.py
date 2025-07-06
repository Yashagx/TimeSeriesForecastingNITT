# real_time_covid_geo_predictor.py — Enhanced Multimodal Fusion Model with XGBoost

import pandas as pd, numpy as np, sqlite3
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from geopy.geocoders import Nominatim
import feedparser, plotly.express as px
from datetime import datetime
import requests, torch
from xgboost import XGBRegressor
from textblob import TextBlob

# ====================
# INIT
# ====================
geolocator = Nominatim(user_agent="covid_mapper")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================
# TEXT EMBEDDER
# ====================
class TextEmbedder:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

    def embed(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            out = self.model(**{k: v.to(device) for k, v in inputs.items()})
        return out.last_hidden_state[:, 0, :].cpu().numpy()

# ====================
# UTILS
# ====================
def geocode(lat, lon):
    try:
        loc = geolocator.reverse((lat, lon), timeout=10)
        return loc.raw['address'].get('state', loc.raw['address'].get('county', 'Unknown'))
    except:
        return 'Unknown'

def normalize_state_name(name):
    mapping = {
        'NCT of Delhi': 'Delhi',
        'Jammu and Kashmir': 'Jammu & Kashmir',
        'Andaman and Nicobar Islands': 'Andaman & Nicobar Islands',
        'Dadra and Nagar Haveli and Daman and Diu': 'Dadra and Nagar Haveli',
    }
    return mapping.get(name, name)

def fetch_truth_by_state():
    url = "https://api.rootnet.in/covid19-in/stats/latest"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from rootnet. Status: {response.status_code}")

    data = response.json()['data']['regional']
    df = pd.DataFrame(data)
    df['state'] = df['loc'].apply(normalize_state_name)
    df = df.rename(columns={
        'confirmedCasesIndian': 'cases_today',
        'discharged': 'recovered_today',
        'deaths': 'deaths_today'
    })
    df['cases_yest'] = df['cases_today'] * 0.95
    df['deaths_yest'] = df['deaths_today'] * 0.95
    df['recovered_yest'] = df['recovered_today'] * 0.95

    df['cases_today'] = df['cases_today'].clip(0, 10000)
    df['deaths_today'] = df['deaths_today'].clip(0, 500)
    df['recovered_today'] = df['recovered_today'].clip(0, 10000)
    return df[['state', 'cases_today', 'cases_yest', 'deaths_today', 'deaths_yest', 'recovered_today', 'recovered_yest']]

def get_news():
    rows = []
    feed = feedparser.parse("https://news.google.com/rss/search?q=covid+india")
    for e in feed.entries:
        try:
            loc = geolocator.geocode(e.title, timeout=10)
            if loc:
                sentiment = TextBlob(e.title).sentiment.polarity
                rows.append({'text': e.title, 'lat': loc.latitude, 'lon': loc.longitude, 'sentiment': sentiment})
        except:
            continue
    if not rows:
        print("[INFO] Using fallback test data.")
        rows = [
            {'text': 'COVID surge in Delhi hospitals', 'lat': 28.6139, 'lon': 77.2090, 'sentiment': -0.5},
            {'text': 'Rising infections in Maharashtra schools', 'lat': 19.7515, 'lon': 75.7139, 'sentiment': -0.4},
            {'text': 'New variants spreading in Kerala', 'lat': 10.8505, 'lon': 76.2711, 'sentiment': -0.3},
            {'text': 'Cases spike in Tamil Nadu after festival', 'lat': 13.0827, 'lon': 80.2707, 'sentiment': -0.6},
            {'text': 'COVID wave in Uttar Pradesh rural areas', 'lat': 26.8467, 'lon': 80.9462, 'sentiment': -0.7},
        ]
    return pd.DataFrame(rows, columns=['text', 'lat', 'lon', 'sentiment'])

def save_results(df, name="covid_results"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"{name}_{ts}.csv", index=False)
    conn = sqlite3.connect("covid_data_india.db")
    df.to_sql(name, conn, if_exists="replace", index=False)
    conn.close()

# ====================
# PIPELINE
# ====================
def train_and_predict():
    embedder = TextEmbedder()
    truth = fetch_truth_by_state()
    news = get_news()

    df = news.dropna(subset=['lat', 'lon']).reset_index(drop=True)

    def safe_geocode(row):
        try:
            return geocode(float(row['lat']), float(row['lon']))
        except:
            return 'Unknown'

    df['state'] = df.apply(safe_geocode, axis=1)
    df = df[df['state'] != 'Unknown']
    df['state'] = df['state'].apply(normalize_state_name)

    df = df.merge(truth, on='state', how='left').dropna(subset=['cases_today'])
    df[['cases_yest','deaths_yest','recovered_yest']] = df[['cases_yest','deaths_yest','recovered_yest']].fillna(0)

    if df.empty:
        raise Exception("No valid data after geocoding. Cannot proceed.")

    le = LabelEncoder()
    df['sid'] = le.fit_transform(df['state'])

    x_text = embedder.embed(df['text'].tolist())
    x_lag = df[['cases_yest','deaths_yest','recovered_yest']].values
    x_sent = df[['sentiment']].values
    x_state = df['sid'].values.reshape(-1, 1)

    X = np.concatenate([x_text, x_lag, x_sent, x_state], axis=1)
    y = df[['cases_today','deaths_today','recovered_today']].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_cases = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model_deaths = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model_recovered = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    model_cases.fit(X_scaled, y[:, 0])
    model_deaths.fit(X_scaled, y[:, 1])
    model_recovered.fit(X_scaled, y[:, 2])

    pred_cases = model_cases.predict(X_scaled)
    pred_deaths = model_deaths.predict(X_scaled)
    pred_recovered = model_recovered.predict(X_scaled)

    df[['pred_cases','pred_deaths','pred_recovered']] = np.stack(
        [pred_cases, pred_deaths, pred_recovered], axis=1)

    agg = df.groupby('state')[['pred_cases','pred_deaths','pred_recovered']].mean().reset_index()
    save_results(agg, "covid_predictions_india")

    fig = px.bar(
        agg.sort_values("pred_cases", ascending=False),
        x="state",
        y="pred_cases",
        color="pred_cases",
        title="Predicted COVID Cases by Indian State (XGBoost with Sentiment)",
        labels={"pred_cases": "Predicted Cases", "state": "State"},
        height=500
    )
    fig.show()

if __name__ == "__main__":
    train_and_predict()