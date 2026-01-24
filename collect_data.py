import os
import csv
from typing import List
from dataclasses import asdict
from playwright.sync_api import sync_playwright
from booking import BookingScraper, HotelReview

def save_reviews_to_csv(reviews: List[HotelReview], filename: str):
    """Saves a list of HotelReview objects to a CSV file."""
    if not reviews:
        print("No reviews to save.")
        return
        
    # Ensure output directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    first_row = asdict(reviews[0])
    headers = first_row.keys()

    print(f"Saving {len(reviews)} reviews to: {filename}...")
    
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for review in reviews:
            writer.writerow(asdict(review))
            
    print("Save successful.")

if __name__ == "__main__":
    hotels_to_scrape = [
        {
            "url": "https://www.booking.com/hotel/pl/deluxe-apartments-by-the-railway-station.en-gb.html?aid=304142&label=gen173nr-10CAEoggI46AdICVgEaLYBiAEBmAEzuAEXyAEM2AED6AEB-AEBiAIBqAIBuAL-mbnLBsACAdICJDM0MjM3ZjhkLTU2NmYtNDY5Mi04Nzk4LWJkNTM4YzRjYjExNdgCAeACAQ&sid=fd292eafc57e85a289f6338a6f7e19ef&dest_id=-537080&dest_type=city&dist=0&group_adults=2&group_children=0&hapos=120&hpos=20&nflt=ht_id%3D204&no_rooms=1&req_adults=2&req_children=0&room1=A%2CA&sb_price_type=total&sr_order=bayesian_review_score&srepoch=1769190511&srpvid=1c5c6a762fc02597b37e7101f73ab8d7&type=total&ucfs=1&#tab-reviews",
            "name": "Deluxe Apartments by The Railway Station Wroclaw",
            "stars": 3,
            "district": "Krzyki"
        },
        {
            "url": "https://www.booking.com/hotel/pl/hpparkplazawroclaw.en-gb.html?aid=304142&label=gen173nr-10CAEoggI46AdICVgEaLYBiAEBmAEzuAEXyAEM2AED6AEB-AEBiAIBqAIBuAL-mbnLBsACAdICJDM0MjM3ZjhkLTU2NmYtNDY5Mi04Nzk4LWJkNTM4YzRjYjExNdgCAeACAQ&sid=fd292eafc57e85a289f6338a6f7e19ef&dest_id=-537080&dest_type=city&dist=0&group_adults=2&group_children=0&hapos=75&hpos=25&nflt=ht_id%3D204&no_rooms=1&req_adults=2&req_children=0&room1=A%2CA&sb_price_type=total&sr_order=bayesian_review_score&srepoch=1769190506&srpvid=1c5c6a762fc02597b37e7101f73ab8d7&type=total&ucfs=1&#tab-reviews",
            "name": "HP Park Plaza",
            "stars": 4,
            "district": "Śródmieście"
        },
        {
            "url": "https://www.booking.com/hotel/pl/granary.en-gb.html?aid=304142&label=gen173nr-10CAEoggI46AdICVgEaLYBiAEBmAEzuAEXyAEM2AED6AEB-AEBiAIBqAIBuAL-mbnLBsACAdICJDM0MjM3ZjhkLTU2NmYtNDY5Mi04Nzk4LWJkNTM4YzRjYjExNdgCAeACAQ&sid=fd292eafc57e85a289f6338a6f7e19ef&dest_id=-537080&dest_type=city&dist=0&group_adults=2&group_children=0&hapos=83&hpos=8&nflt=ht_id%3D204&no_rooms=1&req_adults=2&req_children=0&room1=A%2CA&sb_price_type=total&sr_order=bayesian_review_score&srepoch=1769190508&srpvid=1c5c6a762fc02597b37e7101f73ab8d7&type=total&ucfs=1&#tab-reviews",
            "name": "Great Polonia The Granary La Suite Hotel",
            "stars": 5,
            "district": "Stare Miasto"
        },
    ]
    all_reviews = []

    with sync_playwright() as p:
        # headless=False to see the browser action
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        scraper = BookingScraper(page)
        
        for hotel_config in hotels_to_scrape:
            print(f"Starting: {hotel_config['name']}")
            
            reviews = scraper.scrape_hotel(hotel_config['url'], max_pages=15)

            for review in reviews:
                review.hotel_stars = hotel_config['stars']
                review.hotel_district = hotel_config['district']
            
            all_reviews.extend(reviews)

        browser.close()

    save_reviews_to_csv(all_reviews, "output/wroclaw_hotels_analysis.csv")