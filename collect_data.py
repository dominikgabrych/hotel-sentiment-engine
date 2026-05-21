"""
collect_data.py — Booking.com Review Scraper Orchestrator

Usage:
  python collect_data.py --pass1             # Collect ~150 general reviews per hotel
  python collect_data.py --pass2             # Collect all negative reviews (Fair/Poor/Very Poor)
  python collect_data.py --pass1 --pass2     # Run both passes
  python collect_data.py --pass1 --test      # Test mode: only scrape the first hotel
  python collect_data.py --pass1 --target 200  # Collect 200 reviews per hotel

Output files:
  data/01_raw/raw_reviews.csv           → Pass 1 results
  data/01_raw/negative_reviews_raw.csv  → Pass 2 results
  data/01_raw/review_distribution.csv   → Score distribution metadata per hotel
"""

import os
import csv
import re
import argparse
import json
from typing import List, Tuple, Dict
from dataclasses import asdict

from playwright.sync_api import sync_playwright
from booking import BookingScraper, HotelReview

# ─────────────────────────── utilities ───────────────────────────────


def parse_target_hotels(filepath: str) -> List[Tuple[str, int]]:
    """
    Parses target_hotels.txt and returns a list of (url, stars) tuples.

    Detects star rating from section headers:
      "3 gwiazdkowe 11 hoteli:" → stars = 3
      "4 gwiazdkowe 6 hoteli:"  → stars = 4
      "5 gwiazdkowe 3 hotele:"  → stars = 5
    """
    hotels = []
    current_stars = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Detect section header e.g. "3 gwiazdkowe 11 hoteli:"
            star_match = re.match(r"^(\d)\s+gwiazdkow", line, re.IGNORECASE)
            if star_match:
                current_stars = int(star_match.group(1))
                print(f"  Detected section: {current_stars}★")
                continue

            if line.startswith("https://") and current_stars is not None:
                hotels.append((line, current_stars))

    return hotels


def ensure_output_dir():
    """Creates the necessary directories if they don't exist."""
    os.makedirs("data/01_raw", exist_ok=True)


def save_reviews_to_csv(reviews: List[HotelReview], filepath: str):
    """
    Appends reviews to a CSV file (checkpoint mode).

    Creates the file with a header on the first call. Subsequent calls append rows
    without re-writing the header.
    """
    if not reviews:
        return

    file_exists = os.path.exists(filepath)
    fieldnames = list(asdict(reviews[0]).keys())

    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for review in reviews:
            writer.writerow(asdict(review))


def save_distribution_row(
    filepath: str, hotel_name: str, hotel_stars: int, counts: Dict[str, any]
):
    """
    Appends one row to the review_distribution.csv metadata file.
    """
    file_exists = os.path.exists(filepath)
    fieldnames = [
        "hotel_name",
        "hotel_stars",
        "total",
        "wonderful_9plus",
        "good_7to9",
        "fair_5to7",
        "poor_3to5",
        "very_poor_1to3",
        "negative_total",
        "type_families",
        "type_couples",
        "type_group_of_friends",
        "type_solo_travellers",
        "type_business_travellers",
        "languages_json",
    ]

    negative_total = (
        counts.get("fair_5to7", 0)
        + counts.get("poor_3to5", 0)
        + counts.get("very_poor_1to3", 0)
    )

    row = {
        "hotel_name": hotel_name,
        "hotel_stars": hotel_stars,
        "total": counts.get("total", 0),
        "wonderful_9plus": counts.get("wonderful_9plus", 0),
        "good_7to9": counts.get("good_7to9", 0),
        "fair_5to7": counts.get("fair_5to7", 0),
        "poor_3to5": counts.get("poor_3to5", 0),
        "very_poor_1to3": counts.get("very_poor_1to3", 0),
        "negative_total": negative_total,
        "type_families": counts.get("type_families", 0),
        "type_couples": counts.get("type_couples", 0),
        "type_group_of_friends": counts.get("type_group_of_friends", 0),
        "type_solo_travellers": counts.get("type_solo_travellers", 0),
        "type_business_travellers": counts.get("type_business_travellers", 0),
        "languages_json": json.dumps(counts.get("languages", {})),
    }

    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def clear_file(filepath: str):
    """Removes a file if it exists (called at start of each pass)."""
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"  Cleared existing file: {filepath}")


# ────────────────────────── pass runners ──────────────────────────────


def run_pass1(
    hotels: List[Tuple[str, int]],
    output_file: str,
    distribution_file: str,
    target_per_hotel: int = 150,
    headless: bool = True,
):
    """
    Pass 1: General reviews sorted by Newest First.
    """
    print("\n" + "=" * 60)
    print("PASS 1: General Reviews (PB1, PB2, PB3)")
    print(f"  Target: {target_per_hotel} reviews per hotel")
    print(f"  Output: {output_file}")
    print("=" * 60)

    clear_file(output_file)
    clear_file(distribution_file)

    total_collected = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="en-GB",
        )
        context.add_cookies([
            {"name": "bkng_lang", "value": "en-gb", "domain": ".booking.com", "path": "/"},
            {"name": "bkng_sso_lang", "value": "en-gb", "domain": ".booking.com", "path": "/"},
            {"name": "b_lang", "value": "en-gb", "domain": ".booking.com", "path": "/"}
        ])
        page = context.new_page()
        scraper = BookingScraper(page)

        for i, (url, stars) in enumerate(hotels, 1):
            print(f"\n[{i}/{len(hotels)}] ★{stars} hotel — Starting...")
            try:
                reviews, counts = scraper.scrape_hotel_general(
                    url, hotel_stars=stars, target_count=target_per_hotel
                )
                save_reviews_to_csv(reviews, output_file)

                if reviews:
                    save_distribution_row(
                        distribution_file,
                        hotel_name=reviews[0].hotel_name,
                        hotel_stars=stars,
                        counts=counts,
                    )

                total_collected += len(reviews)
                print(
                    f"  ✓ Checkpoint saved | Hotel total: {len(reviews)} | Running total: {total_collected}"
                )

            except Exception as e:
                print(f"  ✗ ERROR on hotel {i}: {e}")

        browser.close()


def run_pass2(
    hotels: List[Tuple[str, int]],
    output_file: str,
    headless: bool = True,
):
    """
    Pass 2: Targeted collection of negative reviews for PB4 (ABSA).
    """
    print("\n" + "=" * 60)
    print("PASS 2: Negative Reviews for PB4 (rating_score < 7.5)")
    print(f"  Output: {output_file}")
    print("=" * 60)

    clear_file(output_file)

    total_collected = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="en-GB",
        )
        context.add_cookies([
            {"name": "bkng_lang", "value": "en-gb", "domain": ".booking.com", "path": "/"},
            {"name": "bkng_sso_lang", "value": "en-gb", "domain": ".booking.com", "path": "/"},
            {"name": "b_lang", "value": "en-gb", "domain": ".booking.com", "path": "/"}
        ])
        page = context.new_page()
        scraper = BookingScraper(page)

        for i, (url, stars) in enumerate(hotels, 1):
            print(
                f"\n[{i}/{len(hotels)}] ★{stars} hotel — Scraping negative reviews..."
            )
            try:
                reviews = scraper.scrape_hotel_negative(url, hotel_stars=stars)
                save_reviews_to_csv(reviews, output_file)

                total_collected += len(reviews)
                print(
                    f"  ✓ Checkpoint saved | Hotel total: {len(reviews)} | Running total: {total_collected}"
                )

            except Exception as e:
                print(f"  ✗ ERROR on hotel {i}: {e}")

        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Booking.com hotel review scraper")
    parser.add_argument("--pass1", action="store_true")
    parser.add_argument("--pass2", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--target", type=int, default=150)
    args = parser.parse_args()

    if not args.pass1 and not args.pass2:
        args.pass1 = True
        args.pass2 = True

    hotels = parse_target_hotels("target_hotels.txt")
    if args.test:
        hotels = hotels[:1]
        args.pass1 = True
        args.pass2 = False  # Skip pass 2 entirely during test so we don't pull 600 reviews

    ensure_output_dir()
    PASS1_OUTPUT = "data/01_raw/raw_reviews.csv"
    PASS2_OUTPUT = "data/01_raw/negative_reviews_raw.csv"
    DISTRIBUTION_FILE = "data/01_raw/review_distribution.csv"

    if args.pass1:
        run_pass1(
            hotels,
            PASS1_OUTPUT,
            DISTRIBUTION_FILE,
            target_per_hotel=args.target,
            headless=not args.headed,
        )

    if args.pass2:
        run_pass2(hotels, PASS2_OUTPUT, headless=not args.headed)
