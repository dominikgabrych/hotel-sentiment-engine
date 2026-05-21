import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import os

# Ensure reproducible langdetect results
DetectorFactory.seed = 0

def is_english(text):
    if not isinstance(text, str) or not text.strip():
        return True # Empty text doesn't violate English rule
    
    try:
        lang = detect(text)
        return lang == 'en'
    except LangDetectException:
        # If text is too short or lacks letters (e.g., emojis or "10/10"), keep it
        return True

def clean_reviews(input_path, output_path):
    print(f"Loading {input_path}...")
    
    if not os.path.exists(input_path):
        print(f"File {input_path} not found.")
        return
        
    df = pd.read_csv(input_path)
    initial_count = len(df)
    
    # 1. Filter by Date (Jan 2024 - May 2026)
    # The stay_date is usually "Month YYYY", e.g., "May 2026"
    print("Filtering by date (Jan 2024 - May 2026)...")
    df['parsed_date'] = pd.to_datetime(df['stay_date'], format='%B %Y', errors='coerce')
    
    # Filter valid dates in range
    date_mask = (df['parsed_date'] >= '2024-01-01') & (df['parsed_date'] <= '2026-05-31')
    
    # If date couldn't be parsed, we probably want to drop it, or keep it depending on requirements.
    # We will keep only the ones that match the mask.
    df = df[date_mask].copy()
    print(f"  Rows remaining after date filter: {len(df)} (Removed {initial_count - len(df)})")
    
    # 2. Drop duplicates based on content, NOT source_id (to avoid hash collisions for anonymous reviewers)
    print("Dropping duplicates based on actual content...")
    df = df.drop_duplicates(subset=['hotel_name', 'stay_date', 'rating_score', 'pos_text', 'neg_text'])
    print(f"  Rows remaining: {len(df)}")
    
    # 3. Drop rows with missing ratings
    print("Dropping missing ratings...")
    df = df.dropna(subset=['rating_score'])
    print(f"  Rows remaining: {len(df)}")
    
    # --- SAVE FOR STATS (PB1-PB3) ---
    # We keep non-English and short reviews for stats because the numerical/categorical data is still valid!
    stats_output = output_path.replace('.csv', '_stats.csv')
    os.makedirs(os.path.dirname(stats_output), exist_ok=True)
    df.to_csv(stats_output, index=False, encoding='utf-8')
    print(f"\n[✓] Saved {len(df)} reviews for Statistical Analysis (PB1-PB3) to {stats_output}")
    
    # --- FILTERS FOR NLP (PB4) ---
    print("\n--- Applying NLP Filters ---")
    
    # 4. Remove ultra-short reviews (<= 2 words combined)
    print("Removing ultra-short reviews (<= 2 words)...")
    df['combined_text'] = df['pos_text'].fillna('') + " " + df['neg_text'].fillna('')
    df['combined_text'] = df['combined_text'].str.strip()
    
    df['word_count'] = df['combined_text'].apply(lambda x: len(x.split()))
    short_mask = df['word_count'] > 2
    df = df[short_mask].copy()
    print(f"  Rows remaining after ultra-short filter: {len(df)}")
    
    # 5. Filter out non-English reviews
    print("Filtering out non-English reviews...")
    eng_mask = df['combined_text'].apply(is_english)
    df = df[eng_mask].copy()
    print(f"  Rows remaining after language filter: {len(df)}")
    
    # Drop temporary columns
    df = df.drop(columns=['parsed_date', 'combined_text', 'word_count'])
    
    # Save for NLP
    nlp_output = output_path.replace('.csv', '_nlp.csv')
    df.to_csv(nlp_output, index=False, encoding='utf-8')
    print(f"[✓] Saved {len(df)} reviews for NLP/ABSA (PB4) to {nlp_output}")

if __name__ == "__main__":
    raw_path = "data/01_raw/raw_reviews.csv"
    cleaned_base_path = "data/02_cleaned/cleaned_reviews.csv"
    clean_reviews(raw_path, cleaned_base_path)
