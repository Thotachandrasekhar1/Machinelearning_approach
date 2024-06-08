import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize NLTK sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define function to predict sentiment
def predict_sentiment(text):
    # Get sentiment scores
    scores = sia.polarity_scores(text)
    
    # Classify sentiment based on compound score
    if scores['compound'] >= 0.05:
        return 'Positive', '😃'  # Positive
    elif scores['compound'] <= -0.05:
        return 'Negative', '😞'  # Negative
    else:
        return 'Neutral', '😐'  # Neutral

# Streamlit app
def main():
    # Set title and description
    st.title("Sentiment Analysis using Machine Learning")
    st.write("This is a simple Streamlit app for sentiment analysis using a machine learning approach.")
    
    # Input text area
    text = st.text_area("Enter text to analyze sentiment:")
    
    # Button to predict sentiment
    if st.button("Predict Sentiment"):
        if text:
            # Predict sentiment
            sentiment_text, sentiment_emoji = predict_sentiment(text)
            st.write(f"Sentiment: {sentiment_text} {sentiment_emoji}")
        else:
            st.warning("Please enter some text.")

# Run the app
if __name__ == "__main__":
    main()
