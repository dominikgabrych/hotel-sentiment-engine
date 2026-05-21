"""
preprocess.py — Data Preprocessing Pipeline

Combines raw review data from both scraping passes, deduplicates,
and exports two analysis-ready datasets:

  1. data/02_cleaned/reviews_stats.csv
     Full cleaned dataset for statistical analysis (PB1, PB2, PB3).
     Retains all languages and short reviews — only numerical and
     categorical fields are required for statistical tests.

  2. data/03_processed/absa_input.csv
     Filtered dataset for Aspect-Based Sentiment Analysis (PB4).
     Contains only English, substantive negative-text entries from
     all reviews (not just low-scoring ones), ready for PyABSA.

Usage:
  python preprocess.py
"""

import os
import warnings

import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from langdetect import DetectorFactory

warnings.filterwarnings("ignore")
DetectorFactory.seed = 0  # Reproducible language detection


# ─────────────────────────── helpers ─────────────────────────────────


def is_english(text: str) -> bool:
    """Returns True if text is in English or is empty/undetectable."""
    if not isinstance(text, str) or not text.strip():
        return True
    try:
        return detect(text) == "en"
    except LangDetectException:
        return True


def load_and_combine() -> pd.DataFrame:
    """Loads both raw scraping passes and merges them into one DataFrame."""
    df_pass1 = pd.read_csv("data/01_raw/raw_reviews.csv")
    df_pass2 = pd.read_csv("data/01_raw/negative_reviews_raw.csv")

    print(f"Pass 1 (general reviews): {len(df_pass1):,} records")
    print(f"Pass 2 (negative reviews): {len(df_pass2):,} records")

    combined = pd.concat([df_pass1, df_pass2], ignore_index=True)
    print(f"Combined total:            {len(combined):,} records\n")
    return combined


def apply_base_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Applies filters common to both output datasets."""
    # Parse stay_date for filtering (e.g. "May 2026" → datetime)
    df["parsed_date"] = pd.to_datetime(df["stay_date"], format="%B %Y", errors="coerce")

    print("Filtering by date range (January 2024 – May 2026)...")
    date_mask = (df["parsed_date"] >= "2024-01-01") & (df["parsed_date"] <= "2026-05-31")
    df = df[date_mask].copy()
    print(f"  Remaining: {len(df):,}")

    print("Dropping rows with missing rating scores...")
    df = df.dropna(subset=["rating_score"])

    print("Removing duplicate reviews based on content...")
    df = df.drop_duplicates(subset=["hotel_name", "stay_date", "rating_score", "pos_text", "neg_text"])
    print(f"  Remaining after deduplication: {len(df):,}\n")

    return df


def save_stats_dataset(df: pd.DataFrame) -> None:
    """
    Saves the dataset for statistical analysis (PB1–PB3).
    No language or length filters are applied here — the numerical
    and categorical fields are valid regardless of review language.
    """
    output_path = "data/02_cleaned/reviews_stats.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.drop(columns=["parsed_date"]).to_csv(output_path, index=False, encoding="utf-8")
    print(f"[✓] Statistical dataset saved → {output_path}  ({len(df):,} reviews)\n")


def save_absa_dataset(df: pd.DataFrame) -> None:
    """
    Saves the dataset for Aspect-Based Sentiment Analysis (PB4).
    Extracts the negative-text field from all reviews (not only low-scoring
    ones) and applies English-language and minimum-length filters.
    """
    print("--- Building ABSA dataset (PB4) ---")

    # Keep only rows that contain a substantive negative comment
    df_nlp = df[df["neg_text"].notna()].copy()
    df_nlp["neg_text"] = df_nlp["neg_text"].str.strip()
    df_nlp = df_nlp[df_nlp["neg_text"] != ""]

    # Remove complaints too short for meaningful aspect extraction
    df_nlp["word_count"] = df_nlp["neg_text"].apply(lambda x: len(str(x).split()))
    df_nlp = df_nlp[df_nlp["word_count"] > 2].copy()
    print(f"  After removing ultra-short complaints (≤ 2 words): {len(df_nlp):,}")

    # Retain only English-language complaints
    df_nlp = df_nlp[df_nlp["neg_text"].apply(is_english)].copy()
    print(f"  After language filter (English only):               {len(df_nlp):,}")

    output_path = "data/03_processed/absa_input.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_nlp.drop(columns=["parsed_date", "word_count"]).to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[✓] ABSA dataset saved → {output_path}  ({len(df_nlp):,} reviews)")


# ─────────────────────────── entry point ─────────────────────────────


def main() -> None:
    print("=" * 60)
    print("Booking.com Review Preprocessing Pipeline")
    print("=" * 60 + "\n")

    df = load_and_combine()
    df = apply_base_filters(df)

    save_stats_dataset(df)
    save_absa_dataset(df)


if __name__ == "__main__":
    main()
