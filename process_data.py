import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

# Download VADER lexicon if not present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- CATEGORY KEYWORDS ---
CATEGORIES = {
    'cat_cleanliness': ['clean', 'dirty', 'smell', 'mold', 'mould', 'stain', 'hygiene', 'filthy', 'spotless'],
    'cat_location': ['location', 'center', 'centre', 'walking', 'station', 'distance', 'central', 'close', 'nearby'],
    'cat_staff': ['staff', 'friendly', 'helpful', 'rude', 'reception', 'service', 'receptionist', 'polite'],
    'cat_facilities': ['parking', 'kitchen', 'broken', 'furniture', 'equipment', 'bed', 'shower', 'bathroom', 'towel', 'heating', 'ac', 'air conditioning'],
    'cat_value': ['price', 'money', 'expensive', 'cheap', 'value', 'worth', 'overpriced', 'affordable'],
    'cat_noise': ['noise', 'noisy', 'loud', 'quiet', 'party', 'sound', 'silent', 'peaceful']
}


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
    """Applies VADER to score the emotional tone of reviews with misclassification detection."""
    print("Running VADER Sentiment Analysis...")
    sia = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        if not text or len(str(text)) < 3: 
            return 0.0
        return sia.polarity_scores(str(text))['compound']

    # Analyze positive and negative text sections separately
    df['pos_sentiment_score'] = df['pos_text'].apply(get_sentiment)
    df['neg_sentiment_score'] = df['neg_text'].apply(get_sentiment)

    # Calculate average text sentiment (legacy - kept for backward compatibility)
    df['text_sentiment_avg'] = (df['pos_sentiment_score'] + df['neg_sentiment_score']) / 2
    
    # --- NEW: Misclassification Detection ---
    # Flag when positive text appears in negative field (and vice versa)
    df['neg_text_is_positive'] = df['neg_sentiment_score'] > 0.3
    df['pos_text_is_negative'] = df['pos_sentiment_score'] < -0.3
    
    # --- NEW: True Combined Sentiment ---
    # Analyze both texts together as a single string for more accurate overall sentiment
    df['combined_text'] = df['pos_text'].astype(str) + ' ' + df['neg_text'].astype(str)
    df['true_sentiment'] = df['combined_text'].apply(get_sentiment)
    
    return df


def apply_category_detection(df):
    """Detects presence of category keywords in review text."""
    print("Detecting category mentions...")
    
    # Combine pos and neg text for category detection
    combined = (df['pos_text'].astype(str) + ' ' + df['neg_text'].astype(str)).str.lower()
    
    for category, keywords in CATEGORIES.items():
        # Create regex pattern to match any keyword
        pattern = '|'.join([re.escape(kw) for kw in keywords])
        df[category] = combined.str.contains(pattern, regex=True, na=False)
    
    return df


def generate_insights(df):
    """Generates a comprehensive summary in the console for verification."""
    print("\n" + "="*60)
    print("📊 INSIGHTS REPORT")
    print("="*60)
    
    # 1. Local vs Foreign Rating
    local_avg = df[df['is_local'] == 1]['rating_score'].mean()
    foreign_avg = df[df['is_local'] == 0]['rating_score'].mean()
    print(f"\n🌍 LOCAL VS FOREIGN:")
    print(f"   Avg Rating - Locals:  {local_avg:.2f}")
    print(f"   Avg Rating - Foreign: {foreign_avg:.2f}")
    
    # 2. Misclassification Report
    misclassified_neg = df['neg_text_is_positive'].sum()
    misclassified_pos = df['pos_text_is_negative'].sum()
    total = len(df)
    print(f"\n⚠️  MISCLASSIFICATION REPORT:")
    print(f"   Positive feedback in 'negative' section: {misclassified_neg} ({misclassified_neg/total*100:.1f}%)")
    print(f"   Negative feedback in 'positive' section: {misclassified_pos} ({misclassified_pos/total*100:.1f}%)")
    
    # 3. Correlation Comparison
    corr_legacy = df[['rating_score', 'text_sentiment_avg']].corr().iloc[0,1]
    corr_true = df[['rating_score', 'true_sentiment']].corr().iloc[0,1]
    print(f"\n📈 CORRELATION (Rating vs. Sentiment):")
    print(f"   Legacy (avg of pos/neg):   {corr_legacy:.3f}")
    print(f"   True (combined text):      {corr_true:.3f}")
    improvement = corr_true - corr_legacy
    if improvement > 0:
        print(f"   ✅ Improvement: +{improvement:.3f}")
    
    # 4. Category Frequency
    print(f"\n📁 CATEGORY MENTIONS:")
    for category in CATEGORIES.keys():
        count = df[category].sum()
        pct = count / total * 100
        label = category.replace('cat_', '').capitalize()
        print(f"   {label:12}: {count:4} reviews ({pct:5.1f}%)")
    
    # 5. Anomaly Detection (High Score + Negative Text)
    anomalies_classic = df[(df['rating_score'] > 9) & (df['neg_sentiment_score'] < -0.5)]
    anomalies_misclass = df[(df['rating_score'] > 9) & (df['neg_text_is_positive'])]
    print(f"\n🔍 ANOMALIES DETECTED:")
    print(f"   High score (>9) + very negative text: {len(anomalies_classic)}")
    print(f"   High score (>9) + misclassified positive: {len(anomalies_misclass)}")
    
    print("\n" + "="*60)


def main():
    input_file = "output/wroclaw_hotels_analysis.csv"
    output_file = "output/booking_reviews_processed.csv"
    
    df = load_and_clean_data(input_file)
    df = apply_sentiment_analysis(df)
    df = apply_category_detection(df)
    generate_insights(df)
    
    df.to_csv(output_file, index=False)
    print(f"\n✅ Success! Processed data saved to {output_file}")


if __name__ == "__main__":
    main()