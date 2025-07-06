import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# API Key for OpenWeatherMap
API_KEY = "bd5e378503939ddaee76f12ad7a97608"

# States and their coordinates
states = ["Maharashtra", "Delhi", "Tamil Nadu", "Karnataka", "West Bengal"]
state_coords = {
    "Maharashtra": (19.7515, 75.7139),
    "Delhi": (28.6139, 77.2090),
    "Tamil Nadu": (11.1271, 78.6569),
    "Karnataka": (15.3173, 75.7139),
    "West Bengal": (22.9868, 87.8550)
}

# Load historical COVID-19 data (ensure this CSV exists and is formatted correctly)
# Required columns: date, state, infected, recovered, deaths
covid_df = pd.read_csv("historical_covid_data_india.csv", parse_dates=["date"])

# Clean date format
covid_df["date"] = pd.to_datetime(covid_df["date"]).dt.date

# Define date range (based on COVID data)
start_date = covid_df["date"].min()
end_date = covid_df["date"].max()

# Function to fetch historical weather for a given location and date (uses OpenWeatherMap)
def get_weather(lat, lon, date):
    # Convert date to UNIX timestamp (midday)
    dt = int(time.mktime(datetime.combine(date, datetime.min.time()).timetuple()))
    url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine"
    params = {
        "lat": lat,
        "lon": lon,
        "dt": dt,
        "appid": API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        hourly = data.get("hourly", [])
        if not hourly:
            return None

        # Average values over 24 hours
        temps = [h["temp"] for h in hourly]
        hums = [h["humidity"] for h in hourly]
        precips = [h.get("rain", {}).get("1h", 0.0) for h in hourly]

        return {
            "temp": sum(temps) / len(temps),
            "humidity": sum(hums) / len(hums),
            "precip": sum(precips) / len(precips)
        }
    except Exception as e:
        print(f"Error fetching weather for {lat}, {lon} on {date}: {e}")
        return None

# Prepare merged data
merged_data = []

print("Processing merged COVID + climate dataset...")

for state in states:
    lat, lon = state_coords[state]

    # Filter COVID data for this state
    state_covid = covid_df[covid_df["state"] == state].copy()

    for _, row in state_covid.iterrows():
        date = row["date"]

        # Fetch weather for the date
        weather = get_weather(lat, lon, date)
        if not weather:
            continue

        merged_data.append({
            "date": date,
            "state": state,
            "infected": row["infected"],
            "recovered": row["recovered"],
            "deaths": row["deaths"],
            "temp": weather["temp"],
            "humidity": weather["humidity"],
            "precip": weather["precip"]
        })

        print(f"{state} - {date}: {weather}")
        time.sleep(1)  # Avoid hitting API rate limit

# Convert to DataFrame and save
merged_df = pd.DataFrame(merged_data)
merged_df.to_csv("historical_climate_covid_india.csv", index=False)
print("\n✅ Saved merged dataset to historical_climate_covid_india.csv")
