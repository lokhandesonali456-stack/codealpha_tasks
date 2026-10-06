from flask import Flask, render_template, request
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk


# Download VADER
nltk.download("vader_lexicon")


app = Flask(__name__)

sia = SentimentIntensityAnalyzer()


def predict_sentiment(text):

    score = sia.polarity_scores(text)

    compound = score["compound"]

    if compound >= 0.05:
        return "Positive"

    elif compound <= -0.05:
        return "Negative"

    else:
        return "Neutral"


@app.route("/", methods=["GET", "POST"])
def home():

    sentiment = None
    review = ""

    if request.method == "POST":

        review = request.form["review"]

        sentiment = predict_sentiment(review)

    return render_template(
        "index.html",
        sentiment=sentiment,
        review=review
    )


if __name__ == "__main__":
    app.run(debug=True)