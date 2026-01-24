import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if not present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def load_and_clean_data(filepath):
    """Loads data and performs initial cleaning."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Convert Date (from "November 2025" to datetime)
    df['stay_date_dt'] = pd.to_datetime(df['stay_date'], format='%B %Y', errors='coerce')
    df = df.dropna(subset=['stay_date_dt'])
    
    # 2. Extract Seasonality
    df['month'] = df['stay_date_dt'].dt.month
    df['season'] = df['month'].apply(lambda x: 
        'Winter' if x in [12, 1, 2] else 
        'Spring' if x in [3, 4, 5] else 
        'Summer' if x in [6, 7, 8] else 'Autumn')

    # 3. Local Tourist Flag (Poland vs. Rest)
    df['is_local'] = df['reviewer_country'].apply(lambda x: 1 if str(x).strip() == 'Poland' else 0)
    
    # Fill missing text to prevent VADER errors
    df['pos_text'] = df['pos_text'].fillna("")
    df['neg_text'] = df['neg_text'].fillna("")
    
    print(f"Data clean. Rows: {len(df)}")
    return df

def apply_sentiment_analysis(df):
    """Applies VADER to score the emotional tone of reviews."""
    print("Running VADER Sentiment Analysis...")
    sia = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        if not text or len(str(text)) < 3: return 0.0
        return sia.polarity_scores(str(text))['compound']

    # Analyze positive and negative text sections separately
    df['pos_sentiment_score'] = df['pos_text'].apply(get_sentiment)
    df['neg_sentiment_score'] = df['neg_text'].apply(get_sentiment)

    # Calculate average text sentiment
    df['text_sentiment_avg'] = (df['pos_sentiment_score'] + df['neg_sentiment_score']) / 2
    
    return df

def generate_insights(df):
    """Generates a simple summary in the console for verification."""
    print("\n--- INSIGHTS PREVIEW ---")
    
    # 1. Local vs Foreign Rating
    local_avg = df[df['is_local'] == 1]['rating_score'].mean()
    foreign_avg = df[df['is_local'] == 0]['rating_score'].mean()
    print(f"Avg Rating - Locals: {local_avg:.2f}")
    print(f"Avg Rating - Foreign: {foreign_avg:.2f}")
    
    # 2. Detect Anomalies (High Score + Negative Text)
    anomalies = df[(df['rating_score'] > 9) & (df['neg_sentiment_score'] < -0.5)]
    if not anomalies.empty:
        print(f"\nFound {len(anomalies)} anomalies (High score, negative text):")
        print(anomalies[['rating_score', 'neg_text']].head(2))
    
    # 3. Correlation
    corr = df[['rating_score', 'text_sentiment_avg']].corr().iloc[0,1]
    print(f"\nCorrelation rating vs. sentiment: {corr:.2f}")
    print("(Closer to 1.0 means text matches stars)")

def main():
    input_file = "output/wroclaw_hotels_analysis.csv"
    output_file = "output/booking_reviews_processed.csv"
    
    df = load_and_clean_data(input_file)
    df = apply_sentiment_analysis(df)
    generate_insights(df)
    
    df.to_csv(output_file, index=False)
    print(f"\nSuccess! Processed data saved to {output_file}")

if __name__ == "__main__":
    main()