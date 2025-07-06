import requests
import re

# Insert your NewsAPI key here
NEWS_API_KEY = "your_newsapi_key_here"

states = ["Maharashtra", "Delhi", "Tamil Nadu", "Karnataka", "West Bengal"]

def fetch_news_covid_data(state):
    url = f"https://newsapi.org/v2/everything?q=COVID+{state}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])

        infected = recovered = deaths = 0

        for article in articles[:5]:  # Check top 5 articles
            text = (article.get("title", "") or "") + " " + (article.get("description", "") or "")
            text = text.replace(",", "")  # remove commas for easier parsing

            # Show what we're working with
            print(f"[{state}] Checking article: {text[:100]}...")

            # Enhanced regex patterns
            matches = re.findall(r"(\d+)\s*(cases|infected|positive|recovered|deaths|dead)", text, re.IGNORECASE)

            for value, label in matches:
                value = int(value)
                label = label.lower()

                if label in ["cases", "infected", "positive"]:
                    infected = max(infected, value)
                elif "recover" in label:
                    recovered = max(recovered, value)
                elif "death" in label or "dead" in label:
                    deaths = max(deaths, value)

        return infected, recovered, deaths

    except Exception as e:
        print(f"[ERROR] {state}: {e}")
        return 0, 0, 0

# Print only news-based numbers
print("=== News-Based COVID-19 Data (Improved) ===")
for state in states:
    infected, recovered, deaths = fetch_news_covid_data(state)
    print(f"{state} → Infected: {infected}, Recovered: {recovered}, Deaths: {deaths}")
