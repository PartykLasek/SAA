import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)

    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def main():
    print("Sentiment Analysis Application")
    user_input = input("Enter a sentence or phrase: ")

    sentiment = analyze_sentiment(user_input)

    print(f"Sentiment: {sentiment}")

if __name__ == '__main__':
    main()
