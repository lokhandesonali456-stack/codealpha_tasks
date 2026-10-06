import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.sentiment import SentimentIntensityAnalyzer


# Download VADER lexicon
nltk.download("vader_lexicon")


# ----------------------------------------
# Load dataset
# ----------------------------------------

df = pd.read_csv("reviews.csv")

print("Dataset loaded successfully!")
print("\nFirst 5 records:")
print(df.head())


# ----------------------------------------
# Create sentiment analyzer
# ----------------------------------------

sia = SentimentIntensityAnalyzer()


# ----------------------------------------
# Function for sentiment prediction
# ----------------------------------------

def predict_sentiment(text):

    score = sia.polarity_scores(text)

    compound_score = score["compound"]

    if compound_score >= 0.05:
        return "positive"

    elif compound_score <= -0.05:
        return "negative"

    else:
        return "neutral"


# ----------------------------------------
# Apply sentiment analysis
# ----------------------------------------

df["Predicted_Sentiment"] = df["review"].apply(
    predict_sentiment
)


# ----------------------------------------
# Display results
# ----------------------------------------

print("\nSentiment Analysis Results:")
print(df[["review", "Predicted_Sentiment"]])


# ----------------------------------------
# Compare actual and predicted sentiment
# ----------------------------------------

print("\nActual Sentiment:")
print(df["sentiment"].value_counts())

print("\nPredicted Sentiment:")
print(df["Predicted_Sentiment"].value_counts())


# ----------------------------------------
# Create sentiment count graph
# ----------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Predicted_Sentiment"
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig("sentiment_distribution.png")

plt.show()


# ----------------------------------------
# Save results
# ----------------------------------------

df.to_csv(
    "sentiment_results.csv",
    index=False
)

print("\nResults saved to sentiment_results.csv")