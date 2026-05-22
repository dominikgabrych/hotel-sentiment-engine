"""
recover_business_travelers.py
Pass 3 script to extract "Business travellers" disguised as Solo travellers.
"""
import os
import time
import argparse
from typing import List, Tuple
from playwright.sync_api import sync_playwright

from booking import BookingScraper, HotelReview
from collect_data import parse_target_hotels, save_reviews_to_csv, clear_file


class BusinessScraper(BookingScraper):
    def scrape_hotel_business(self, url: str, hotel_stars: int, target_count: int = 150) -> List[HotelReview]:
        hotel_name = self._navigate_to_reviews(url)
        
        # Select Sort Order -> Newest First
        self._set_sort_order("NEWEST_FIRST")
        
        # Filter for Business Travellers
        try:
            customer_select = self.page.locator('select[data-testid="customerType"]')
            if customer_select.count() > 0:
                customer_select.select_option("BUSINESS_TRAVELLERS")
                self.page.wait_for_load_state("networkidle", timeout=10000)
                time.sleep(1.5)  # Let it render
                print(f"   -> Customer Type set to: BUSINESS_TRAVELLERS")
            else:
                print("   WARNING: Customer Type dropdown not found. Skipping.")
                return []
        except Exception as e:
            print(f"   WARNING: Could not set Customer Type: {e}")
            return []
            
        # Paginate and collect
        reviews = self._paginate_and_collect(hotel_name, target_count=target_count)
        
        for r in reviews:
            r.hotel_stars = hotel_stars
            r.traveler_type = "Business"  # Force overwrite since Booking says 'Solo'
            
        print(f"   ✓ Pass 3 (Business) complete: {len(reviews)} business reviews collected.")
        return reviews


def run_pass3(hotels: List[Tuple[str, int]], output_file: str, target_per_hotel: int = 200):
    print("\n" + "=" * 60)
    print("PASS 3: Recovering 'Business travellers'")
    print(f"  Target: {target_per_hotel} reviews per hotel")
    print(f"  Output: {output_file}")
    print("=" * 60)

    clear_file(output_file)
    total_collected = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="en-GB"
        )
        # Avoid local language redirects
        context.add_cookies([
            {"name": "bkng_lang", "value": "en-gb", "domain": ".booking.com", "path": "/"},
            {"name": "bkng_sso_lang", "value": "en-gb", "domain": ".booking.com", "path": "/"},
            {"name": "b_lang", "value": "en-gb", "domain": ".booking.com", "path": "/"}
        ])
        page = context.new_page()
        scraper = BusinessScraper(page)

        for i, (url, stars) in enumerate(hotels, 1):
            print(f"\n[{i}/{len(hotels)}] ★{stars} hotel — Scraping Business...")
            try:
                reviews = scraper.scrape_hotel_business(url, hotel_stars=stars, target_count=target_per_hotel)
                save_reviews_to_csv(reviews, output_file)
                total_collected += len(reviews)
                print(f"  ✓ Checkpoint saved | Running total: {total_collected}")
            except Exception as e:
                print(f"  ✗ ERROR on hotel {i}: {e}")

        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    hotels = parse_target_hotels("target_hotels.txt")
    if args.test:
        hotels = hotels[:1]

    os.makedirs("data/01_raw", exist_ok=True)
    PASS3_OUTPUT = "data/01_raw/business_reviews_raw.csv"
    run_pass3(hotels, PASS3_OUTPUT, target_per_hotel=100)
