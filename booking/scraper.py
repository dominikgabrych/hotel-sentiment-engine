import hashlib
import time
import random
import re
from typing import List, Dict
from playwright.sync_api import Page, Locator
from .models import HotelReview

class BookingScraper:
    """
    Engine responsible for scraping reviews from a specific Booking.com hotel page.
    """
    def __init__(self, page: Page):
        self.page = page

    def _get_text_safe(self, parent: Locator, selector: str) -> str:
        """Helper: Safely extracts inner text from an element if it exists."""
        el = parent.locator(selector)
        return el.inner_text().strip() if el.count() > 0 else ""

    @staticmethod
    def _generate_id(hotel_name: str, reviewer: str, title: str) -> str:
        """Helper: Generates a unique MD5 hash based on review content."""
        raw = f"{hotel_name}|{reviewer}|{title}"
        return hashlib.md5(raw.encode('utf-8')).hexdigest()

    def _parse_stay_info(self, card: Locator) -> Dict:
        """Helper: Extracts metadata like room type, stay date, and duration."""
        room = self._get_text_safe(card, '[data-testid="review-room-name"]')
        date = self._get_text_safe(card, '[data-testid="review-stay-date"]')
        traveler = self._get_text_safe(card, '[data-testid="review-traveler-type"]')
        nights_txt = self._get_text_safe(card, '[data-testid="review-num-nights"]')
        
        duration = 1
        if nights_txt:
            try:
                # Example text: "1 night" -> takes "1"
                duration = int(nights_txt.split(' ')[0])
            except ValueError:
                pass

        return {
            "room_type": room,
            "stay_date": date,
            "traveler_type": traveler,
            "stay_duration": duration
        }

    def _extract_score_from_text(self, text: str) -> float:
        """Helper: Parses score string (e.g., 'Scored 10.0') into float."""
        if not text:
            return 0.0
        # Handle different decimal separators (e.g., '9,5' -> '9.5')
        text = text.replace(',', '.')
        match = re.search(r"\d+(\.\d+)?", text)
        if match:
            return float(match.group())
        return 0.0

    def _extract_from_single_page(self, hotel_name: str) -> List[HotelReview]:
        """Scrapes all review cards currently visible on the page."""
        data = []
        cards = self.page.locator('div[data-testid="review-card"]')
        count = cards.count()
        print(f"   -> Scraping {count} reviews from current page...")

        for i in range(count):
            card = cards.nth(i)
            
            title = self._get_text_safe(card, '[data-testid="review-title"]')
            pos = self._get_text_safe(card, '[data-testid="review-positive-text"]')
            neg = self._get_text_safe(card, '[data-testid="review-negative-text"]')
            name = self._get_text_safe(card, '[data-testid="review-author-name"]')
            rating_str = self._get_text_safe(card, '[data-testid="review-score"]')
            rating = self._extract_score_from_text(rating_str)
            stay_info = self._parse_stay_info(card)
            country = "Unknown"
            flag_locator = card.locator('img[src*="flags"]')
            if flag_locator.count() > 0:
                country_alt = flag_locator.first.get_attribute("alt")
                if country_alt:
                    country = country_alt.strip()
            rid = self._generate_id(hotel_name, name, title)

            review = HotelReview(
                source_id=rid,
                hotel_name=hotel_name,
                rating_score=rating,
                title=title,
                pos_text=pos,
                neg_text=neg,
                stay_duration=stay_info['stay_duration'],
                stay_date=stay_info['stay_date'],
                room_type=stay_info['room_type'],
                traveler_type=stay_info['traveler_type'],
                reviewer_country=country,
            )
            data.append(review)
        
        return data

    def scrape_hotel(self, url: str, max_pages: int = 5) -> List[HotelReview]:
        """
        Main manager loop. Navigates pages and collects data.
        If max_pages is None, it scrapes until 'Next' button is disabled.
        """
        all_reviews = []
        print(f"Opening: {url}")
        self.page.goto(url)
        
        # Wait for dynamic content to load
        self.page.wait_for_selector('div[data-testid="review-card"]', timeout=10000)
        
        # Try to extract hotel name from header
        header_locator = self.page.locator('h2').filter(has_text="Guest reviews for").first
        if header_locator.count() > 0:
            full_header_text = header_locator.inner_text() 
            hotel_name = full_header_text.replace("Guest reviews for ", "").strip()
        else:
            print("WARNING: Hotel name header not found. Using fallback.")
            hotel_name = "Unknown Hotel"

        page_num = 1
        while page_num <= max_pages:
            print(f"--- Page {page_num} ---")
            
            batch = self._extract_from_single_page(hotel_name)
            all_reviews.extend(batch)
            
            # Navigation logic
            next_button = self.page.locator('button[aria-label="Next page"]')
            
            if next_button.count() == 0:
                print("'Next' button not found")
                break
            
            if next_button.is_disabled():
                print("'Next' button is disabled, ending process.")
                break

            print("Clicking 'Next'...")
            next_button.click()
            
            time.sleep(random.uniform(2, 4))
            self.page.wait_for_load_state("networkidle") 
            
            page_num += 1

        print(f"Done! Collected {len(all_reviews)} reviews in total.")
        return all_reviews