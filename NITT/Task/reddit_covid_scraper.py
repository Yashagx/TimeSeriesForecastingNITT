import praw
import pandas as pd
from datetime import datetime

# Use your actual Reddit credentials
reddit = praw.Reddit(
    client_id="2epQGb_f5H4SQLgb1jtlzw",
    client_secret="5W-cM8FHz3a6Y_FECKmEHF61PRi7jg",
    user_agent="CovidDataScraper by u/Zestyclose-Fig4731"
)

# Keywords and Subreddits
keywords = ["covid", "coronavirus", "covid-19", "pandemic", "omicron", "vaccine", "lockdown"]
subreddits = ["coronavirus", "COVID19", "IndiaCoronavirus", "worldnews", "news"]

# Collecting posts
posts = []

for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    for keyword in keywords:
        for submission in subreddit.search(keyword, sort="new", limit=100):
            posts.append({
                "title": submission.title,
                "selftext": submission.selftext[:500],
                "subreddit": subreddit_name,
                "author": str(submission.author),
                "created_utc": datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                "url": submission.url,
                "score": submission.score,
                "num_comments": submission.num_comments
            })

# Save to CSV
df = pd.DataFrame(posts)
df.to_csv("reddit_covid_data_latest.csv", index=False)

print(f"✅ Extracted {len(df)} posts to 'reddit_covid_data_latest.csv'")
