import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from transformers import pipeline
import requests
import pytz

# Your NewsAPI Key
API_KEY = 'b511d67695ba44248857916965e5126a'  # Your actual API key
BASE_URL = 'https://newsapi.org/v2/top-headlines'

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Initialize summarization pipeline
summarizer = pipeline("summarization", model="t5-small")

# Function to get the time in EST
def get_est_time():
    # Get the current UTC time and convert it to EST time zone
    utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)  # Get UTC time
    est = pytz.timezone('US/Eastern')  # Eastern Standard Time zone
    est_now = utc_now.astimezone(est)  # Convert to EST
    return est_now

# Function to get news for a specific date
def get_news_data(country='us', page_size=10, date=None):
    if date is None:
        date = datetime.today().strftime('%Y-%m-%d')
    params = {
        'apiKey': API_KEY,
        'country': country,  # Fetch articles from the US
        'pageSize': page_size,  # Number of articles to fetch (max 100)
    }
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
        # Filter articles by date
        articles = [article for article in data['articles'] if article['publishedAt'][:10] == date]
        return articles
    else:
        st.error(f"Error fetching news data: {response.status_code} - {response.text}")
        return None

# Function to analyze sentiment
def analyze_sentiment(description):
    sentiment = sentiment_analyzer(description)
    return sentiment[0]['label']

# Function to summarize the description with adjusted length
def summarize_article(description):
    input_length = len(description.split())
    max_len = min(100, input_length)  # Ensure max_length is not greater than input_length
    min_len = min(30, max_len - 1)  # Ensure min_length is less than max_length
    if input_length < 10:
        max_len = input_length
        min_len = max_len
    summary = summarizer(description, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]['summary_text']

# Function to scrape news and perform sentiment analysis and summarization
def scrape_news(date=None):
    if date is None:
        date = datetime.today().strftime('%Y-%m-%d')
    news_data = get_news_data(country='us', page_size=10, date=date)
    
    if news_data:
        articles = []
        for article in news_data:
            title = article['title']
            description = article['description']
            date = article['publishedAt'][:10]  # Extract the date (YYYY-MM-DD)
            if description:
                sentiment = analyze_sentiment(description)  # Sentiment analysis
                summary = summarize_article(description)  # Summarization
                articles.append([title, description, sentiment, summary, date])
            else:
                articles.append([title, "No description available", "N/A", "N/A", date])
        return pd.DataFrame(articles, columns=["Title", "Description", "Sentiment", "Summary", "Date"])

# Get yesterday's date in EST
est_now = get_est_time()
yesterday = (est_now - timedelta(1)).strftime('%Y-%m-%d')

# Timer for countdown to next update (set to 12:01 AM every day)
def time_remaining():
    next_update_time = datetime.combine(est_now.date(), datetime.min.time(), tzinfo=est_now.tzinfo) + timedelta(days=1, minutes=1)  # Next day at 12:01 AM
    remaining_time = next_update_time - est_now  # Now both are aware datetimes

    # Extract hours, minutes, and seconds
    hours, remainder = divmod(remaining_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# Streamlit UI
st.set_page_config(page_title="News Scraper", page_icon="📖", layout="wide")

# Title and Description
st.title('📰 News Scraper')
st.write("This is a web scraper that fetches news articles, performs sentiment analysis, and summarizes the descriptions.")

# Container for displaying the button and output
with st.container():
    st.subheader(f"Click the button to scrape news for {yesterday}:")
    if st.button(f"Scrape News for {yesterday}"):
        articles = scrape_news(date=yesterday)  # Use yesterday's date
        if articles is not None:
            st.dataframe(articles)  # Display the articles in a more readable format
        else:
            st.write("No news data found for the selected date.")

# Timer display (positioned at the top right of the page)
remaining = time_remaining()
st.markdown(f"<h3 style='text-align:right;'>Time remaining until next update: {remaining}</h3>", unsafe_allow_html=True)

# Custom CSS for styling
st.markdown("""
    <style>
        .reportview-container {
            padding-top: 50px;
        }
        h3 {
            font-size: 20px;
            font-weight: bold;
            color: #007bff;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            font-size: 18px;
            border-radius: 5px;
            padding: 15px;
            width: 50%;
            margin-top: 20px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        table {
            width: 100%;
            margin-top: 30px;
            border-collapse: collapse;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        table, th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #f1f1f1;
            font-weight: bold;
            color: #555;
        }
        td {
            background-color: #fff;
            color: #444;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
    </style>
""", unsafe_allow_html=True)
