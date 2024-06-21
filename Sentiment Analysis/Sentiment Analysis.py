import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text.
    Returns 'Positive', 'Negative', or 'Neutral'.
    """
    sentiment_score = sia.polarity_scores(text)
    compound_score = sentiment_score['compound']
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Sample data: replace this with your actual data source
reviews = [
    "The food was absolutely wonderful, from preparation to presentation, very pleasing.",
    "I did not enjoy the meal at all, it was too salty and the service was slow.",
    "It was an average experience, nothing special to mention."
]

# Analyze sentiment for each review
results = []

for review in reviews:
    sentiment = analyze_sentiment(review)
    results.append({
        'Review': review,
        'Sentiment': sentiment
    })

# Create a DataFrame for better visualization
df = pd.DataFrame(results)
print(df)
