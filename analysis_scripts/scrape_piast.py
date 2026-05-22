"""
scrape_piast.py — One-shot supplementary scraper for Hotel Piast (missing 20th hotel).

Runs Pass 1 (general), Pass 2 (negative), and Pass 3 (business) for Hotel Piast only.
APPENDS to existing CSV files — does NOT overwrite or clear them.

Run from the project root:
    python analysis_scripts/scrape_piast.py
"""

import os
import sys
import time
from typing import List

# Make sure project root imports work when called from root dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from playwright.sync_api import sync_playwright
from booking import BookingScraper, HotelReview
from analysis_scripts.recover_business_travelers import BusinessScraper
from analysis_scripts.collect_data import save_reviews_to_csv

# ─── Hotel to scrape (replacement for Hotel Piast which had bot-protection issues) ────────
# ibis Wrocław Centrum — 3★ international chain, good diversity vs boutique hotels in dataset
HOTEL_URL   = (
    "https://www.booking.com/hotel/pl/ibis-wroclaw-centrum.en-gb.html"
    "?aid=304142&label=gen173nr-10CAEoggI46AdIM1gEaLYBiAEBmAEzuAEXyAEM"
    "2AED6AEB-AEBiAIBqAIBuAKhiLLQBsACAdICJGMyOTU4MGE3LTNiOTMtNDM3OC04"
    "ZjNiLTAyOTQ4NDlhMjdkZtgCAeACAQ"
    "&sid=daf7eb85b2419aa34769a848cf0964a9"
    "&dest_id=1635737&dest_type=hotel#tab-reviews"
)
HOTEL_STARS = 3
HOTEL_NAME  = "ibis Wrocław Centrum"

PASS1_OUTPUT = "data/01_raw/raw_reviews.csv"
PASS2_OUTPUT = "data/01_raw/negative_reviews_raw.csv"
PASS3_OUTPUT = "data/01_raw/business_reviews_raw.csv"

PASS1_TARGET = 200   # extra buffer; preprocess.py deduplicates
PASS3_TARGET = 100   # business reviews


# ─── context factory ──────────────────────────────────────────────────────────

def _new_browser_scraper(p, scraper_class=BookingScraper):
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        locale="en-GB",
    )
    context.add_cookies([
        {"name": "bkng_lang",     "value": "en-gb", "domain": ".booking.com", "path": "/"},
        {"name": "bkng_sso_lang", "value": "en-gb", "domain": ".booking.com", "path": "/"},
        {"name": "b_lang",        "value": "en-gb", "domain": ".booking.com", "path": "/"},
    ])
    page = context.new_page()
    return browser, scraper_class(page)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"{HOTEL_NAME} — supplementary scrape (Pass 1 + 2 + 3)")
    print("=" * 60)

    os.makedirs("data/01_raw", exist_ok=True)

    with sync_playwright() as p:

        # ── Pass 1: general reviews ──────────────────────────────────────────
        print("\n[Pass 1] General reviews (target: {})...".format(PASS1_TARGET))
        browser, scraper = _new_browser_scraper(p, BookingScraper)
        try:
            reviews, _ = scraper.scrape_hotel_general(
                HOTEL_URL, hotel_stars=HOTEL_STARS, target_count=PASS1_TARGET
            )
            save_reviews_to_csv(reviews, PASS1_OUTPUT)
            print(f"  ✓ Pass 1 done — {len(reviews)} reviews appended to {PASS1_OUTPUT}")
        except Exception as e:
            print(f"  ✗ Pass 1 failed: {e}")
        finally:
            browser.close()

        time.sleep(4)

        # ── Pass 2: negative reviews ──────────────────────────────────────────
        print("\n[Pass 2] Negative reviews...")
        browser, scraper = _new_browser_scraper(p, BookingScraper)
        try:
            reviews = scraper.scrape_hotel_negative(HOTEL_URL, hotel_stars=HOTEL_STARS)
            save_reviews_to_csv(reviews, PASS2_OUTPUT)
            print(f"  ✓ Pass 2 done — {len(reviews)} reviews appended to {PASS2_OUTPUT}")
        except Exception as e:
            print(f"  ✗ Pass 2 failed: {e}")
        finally:
            browser.close()

        time.sleep(4)

        # ── Pass 3: business reviews ──────────────────────────────────────────
        print(f"\n[Pass 3] Business reviews (target: {PASS3_TARGET})...")
        browser, scraper = _new_browser_scraper(p, BusinessScraper)
        try:
            reviews = scraper.scrape_hotel_business(
                HOTEL_URL, hotel_stars=HOTEL_STARS, target_count=PASS3_TARGET
            )
            save_reviews_to_csv(reviews, PASS3_OUTPUT)
            print(f"  ✓ Pass 3 done — {len(reviews)} reviews appended to {PASS3_OUTPUT}")
        except Exception as e:
            print(f"  ✗ Pass 3 failed: {e}")
        finally:
            browser.close()

    print("\n" + "=" * 60)
    print(f"{HOTEL_NAME} scraping complete.")
    print("Next step: python analysis_scripts/preprocess.py")
    print("           python analysis_scripts/analyse_statistics.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
