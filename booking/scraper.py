import hashlib
import time
import random
import re
from typing import List, Dict, Optional
from playwright.sync_api import Page, Locator
from .models import HotelReview

# --- Constants ---
SORT_NEWEST_FIRST = "NEWEST_FIRST"

# Score filter values → human-readable labels
NEGATIVE_SCORE_FILTERS = [
    ("REVIEW_ADJ_AVERAGE_PASSABLE", "Fair 5-7"),
    ("REVIEW_ADJ_POOR", "Poor 3-5"),
    ("REVIEW_ADJ_VERY_POOR", "Very Poor 1-3"),
]

# Maps option value → short key for distribution CSV
SCORE_FILTER_KEYS = {
    "ALL": "total",
    "REVIEW_ADJ_SUPERB": "wonderful_9plus",
    "REVIEW_ADJ_GOOD": "good_7to9",
    "REVIEW_ADJ_AVERAGE_PASSABLE": "fair_5to7",
    "REVIEW_ADJ_POOR": "poor_3to5",
    "REVIEW_ADJ_VERY_POOR": "very_poor_1to3",
}


class BookingScraper:
    """
    Scrapes hotel reviews from Booking.com using Playwright.

    Two public methods:
      - scrape_hotel_general()  → Pass 1: newest reviews, target_count per hotel
      - scrape_hotel_negative() → Pass 2: all Fair + Poor + Very Poor reviews per hotel
    """

    def __init__(self, page: Page):
        self.page = page

    # ─────────────────────────── helpers ────────────────────────────

    @staticmethod
    def _generate_id(hotel_name: str, reviewer: str, title: str) -> str:
        """Generates a unique MD5 hash based on review content (dedup key)."""
        raw = f"{hotel_name}|{reviewer}|{title}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _get_text_safe(self, parent: Locator, selector: str) -> str:
        """Safely extracts inner text from a child element, or returns empty string."""
        el = parent.locator(selector)
        return el.inner_text().strip() if el.count() > 0 else ""

    def _extract_score_from_text(self, text: str) -> float:
        """Parses a score string like 'Scored 8.5' or '9,2' into a float."""
        if not text:
            return 0.0
        text = text.replace(",", ".")
        match = re.search(r"\d+(\.\d+)?", text)
        return float(match.group()) if match else 0.0

    # ─────────────────────── page interactions ───────────────────────

    def _dismiss_cookie_banner(self):
        """
        Dismisses the Booking.com OneTrust cookie consent banner.

        Uses JavaScript click to bypass overlay interception issues.
        Falls back to removing the element from DOM entirely.
        """
        try:
            accepted = self.page.evaluate("""
                () => {
                    const btn = document.getElementById('onetrust-accept-btn-handler');
                    if (btn) { btn.click(); return true; }
                    return false;
                }
            """)
            if accepted:
                time.sleep(0.5)
                print("   -> Cookie banner dismissed.")
                return
        except Exception:
            pass

        # Fallback: remove the entire banner element from DOM
        try:
            removed = self.page.evaluate("""
                () => {
                    const sdk = document.getElementById('onetrust-consent-sdk');
                    if (sdk) { sdk.remove(); return true; }
                    return false;
                }
            """)
            if removed:
                print("   -> Cookie banner removed from DOM (fallback).")
        except Exception:
            pass

    def _set_sort_order(self, value: str = SORT_NEWEST_FIRST):
        """Selects the review sort order (e.g. NEWEST_FIRST)."""
        try:
            sort_select = self.page.locator(
                'select[data-testid="reviews-sorter-component"]'
            )
            if sort_select.count() > 0:
                sort_select.select_option(value)
                self.page.wait_for_load_state("networkidle", timeout=10000)
                print(f"   -> Sort order set to: {value}")
            else:
                print("   WARNING: Sort dropdown not found.")
        except Exception as e:
            print(f"   WARNING: Could not set sort order: {e}")

    def _set_score_filter(self, value: str):
        """Selects the review score range filter (e.g. REVIEW_ADJ_POOR)."""
        try:
            filter_select = self.page.locator('select[data-testid="scoreRange"]')
            if filter_select.count() > 0:
                filter_select.select_option(value)
                self.page.wait_for_load_state("networkidle", timeout=10000)
                time.sleep(random.uniform(1.0, 2.0))  # Extra wait for re-render
                print(f"   -> Score filter set to: {value}")
            else:
                print("   WARNING: Score filter dropdown not found.")
        except Exception as e:
            print(f"   WARNING: Could not set score filter: {e}")

    def get_review_counts(self) -> Dict[str, any]:
        """
        Reads the number of reviews per score category, customer type, and language.
        Returns a dict with keys from SCORE_FILTER_KEYS mapping and dynamically collected keys.
        """
        counts = {"languages": {}}

        # 1. Score Range
        try:
            filter_select = self.page.locator('select[data-testid="scoreRange"]')
            if filter_select.count() > 0:
                options = filter_select.locator("option").all()
                for opt in options:
                    key_raw = opt.get_attribute("value")
                    text = opt.inner_text().strip()
                    # Parse number in parentheses: "Fair: 5-7 (326)" → 326
                    match = re.search(r"\((\d+)\)", text)
                    if match and key_raw and key_raw in SCORE_FILTER_KEYS:
                        friendly_key = SCORE_FILTER_KEYS[key_raw]
                        counts[friendly_key] = int(match.group(1))
                        counts[f"_raw_{key_raw}"] = int(match.group(1))
        except Exception as e:
            print(f"   WARNING: Could not read review counts: {e}")

        # 2. Customer Type
        try:
            customer_select = self.page.locator('select[data-testid="customerType"]')
            if customer_select.count() > 0:
                options = customer_select.locator("option").all()
                for opt in options:
                    key_raw = opt.get_attribute("value")
                    text = opt.inner_text().strip()
                    match = re.search(r"\((\d+)\)", text)
                    if match and key_raw and key_raw != "ALL":
                        counts[f"type_{key_raw.lower()}"] = int(match.group(1))
        except Exception as e:
            print(f"   WARNING: Could not read customer type counts: {e}")

        # 3. Languages
        try:
            lang_select = self.page.locator('select[data-testid="languages"]')
            if lang_select.count() > 0:
                options = lang_select.locator("option").all()
                for opt in options:
                    key_raw = opt.get_attribute("value")
                    text = opt.inner_text().strip()
                    match = re.search(r"\((\d+)\)", text)
                    if (
                        match and key_raw and key_raw != "0"
                    ):  # 0 is ALL in language select
                        counts["languages"][key_raw] = int(match.group(1))
        except Exception as e:
            print(f"   WARNING: Could not read language counts: {e}")

        return counts

    # ─────────────────────── core scraping ───────────────────────────

    def _extract_from_single_page(self, hotel_name: str) -> List[HotelReview]:
        """Scrapes all review cards currently visible on the page."""
        data = []
        cards = self.page.locator('div[data-testid="review-card"]')
        count = cards.count()
        print(f"   -> Found {count} review cards on this page.")

        for i in range(count):
            card = cards.nth(i)

            title = self._get_text_safe(card, '[data-testid="review-title"]')
            pos = self._get_text_safe(card, '[data-testid="review-positive-text"]')
            neg = self._get_text_safe(card, '[data-testid="review-negative-text"]')
            name = self._get_text_safe(card, '[data-testid="review-author-name"]')
            rating_str = self._get_text_safe(card, '[data-testid="review-score"]')
            stay_date = self._get_text_safe(card, '[data-testid="review-stay-date"]')
            traveler = self._get_text_safe(card, '[data-testid="review-traveler-type"]')
            rating = self._extract_score_from_text(rating_str)

            country = "Unknown"
            flag_loc = card.locator('img[src*="flags"]')
            if flag_loc.count() > 0:
                alt = flag_loc.first.get_attribute("alt")
                if alt:
                    country = alt.strip()

            rid = self._generate_id(hotel_name, name, title)

            review = HotelReview(
                source_id=rid,
                hotel_name=hotel_name,
                stay_date=stay_date,
                hotel_stars=0,  # Assigned by caller after collection
                traveler_type=traveler,
                reviewer_country=country,
                rating_score=rating,
                title=title,
                pos_text=pos,
                neg_text=neg,
            )
            data.append(review)

        return data

    def _paginate_and_collect(
        self, hotel_name: str, target_count: Optional[int] = None
    ) -> List[HotelReview]:
        """
        Paginates through review pages and collects reviews.

        Args:
            hotel_name:   Used for ID generation and logging.
            target_count: Stop after reaching this many reviews. None = paginate to end.
        """
        all_reviews = []
        page_num = 1

        while True:
            print(f"   --- Page {page_num} ---")
            batch = self._extract_from_single_page(hotel_name)
            all_reviews.extend(batch)

            if target_count and len(all_reviews) >= target_count:
                print(
                    f"   -> Target of {target_count} reviews reached ({len(all_reviews)} collected)."
                )
                break

            next_btn = self.page.locator('button[aria-label="Next page"]')

            if next_btn.count() == 0:
                print("   -> No 'Next' button found. End of reviews.")
                break

            if next_btn.is_disabled():
                print("   -> 'Next' button is disabled. End of reviews.")
                break

            print("   -> Clicking 'Next'...")
            self._dismiss_cookie_banner()  # Banner may appear lazily after page interactions
            next_btn.click()
            time.sleep(random.uniform(2.0, 4.0))
            self.page.wait_for_load_state("networkidle")
            page_num += 1

        return all_reviews

    def _navigate_to_reviews(self, url: str) -> str:
        """
        Navigates to the hotel URL and waits for review cards to appear.

        Order of operations:
          1. Navigate and wait for full page load (networkidle)
          2. Dismiss cookie banner immediately (before it blocks any clicks)
          3. Scroll down to trigger lazy-loaded review content
          4. Wait for review cards with extended timeout
        Returns the extracted hotel name from the page h2 header.
        """
        print(f"   Opening URL...")
        self.page.goto(url, wait_until="domcontentloaded", timeout=30000)

        # Dismiss cookie banner FIRST — it may block all further clicks
        self._dismiss_cookie_banner()

        # The #tab-reviews hash does NOT auto-activate the tab in headless mode.
        # Wait for the Reviews nav tab to appear in DOM, then click it.
        try:
            self.page.wait_for_selector(
                '[data-testid="Property-Header-Nav-Tab-Trigger-reviews"]', timeout=10000
            )
            reviews_tab = self.page.locator(
                '[data-testid="Property-Header-Nav-Tab-Trigger-reviews"]'
            )
            reviews_tab.click()
            print("   -> Clicked Reviews tab.")
            time.sleep(2.0)
        except Exception as e:
            print(
                f"   WARNING: Reviews tab not found or click failed ({e}). Trying text fallback..."
            )
            try:
                self.page.locator("text=Guest reviews").first.click()
                time.sleep(2.0)
                print("   -> Clicked 'Guest reviews' via text fallback.")
            except Exception as e2:
                print(f"   WARNING: Text fallback also failed: {e2}")

        # Wait for review cards with generous timeout
        try:
            self.page.wait_for_selector('div[data-testid="review-card"]', timeout=20000)
            print("   -> Review cards loaded.")
        except Exception:
            print(
                "   WARNING: Review cards did not load within timeout. Will attempt to scrape anyway."
            )

        # Extract hotel name
        header = self.page.locator("h2").filter(has_text="Guest reviews for").first
        if header.count() > 0:
            hotel_name = header.inner_text().replace("Guest reviews for ", "").strip()
        else:
            # Fallback: use page title
            title = self.page.title()
            hotel_name = (
                title.split(",")[0].strip()
                if "," in title
                else title.split("|")[0].strip()
            )
            print(
                f"   WARNING: Hotel name header not found. Using title fallback: '{hotel_name}'"
            )

        return hotel_name

    # ─────────────────────── public API ──────────────────────────────

    def scrape_hotel_general(
        self, url: str, hotel_stars: int, target_count: int = 150
    ) -> tuple[List[HotelReview], Dict[str, int]]:
        """
        Pass 1 — Collects general reviews sorted by Newest First.

        Returns:
            (reviews, counts) where counts is the score distribution dict.
        """
        hotel_name = self._navigate_to_reviews(url)

        # Read distribution before any filtering
        counts = self.get_review_counts()
        total = counts.get("total", "?")
        negative_total = (
            counts.get("fair_5to7", 0)
            + counts.get("poor_3to5", 0)
            + counts.get("very_poor_1to3", 0)
        )
        print(
            f"   Hotel: '{hotel_name}' | Total reviews: {total} | Negative (<7.5): ~{negative_total}"
        )

        self._set_sort_order(SORT_NEWEST_FIRST)

        reviews = self._paginate_and_collect(hotel_name, target_count=target_count)

        for r in reviews:
            r.hotel_stars = hotel_stars

        print(f"   ✓ Pass 1 complete: {len(reviews)} reviews collected.")
        return reviews, counts

    def scrape_hotel_negative(self, url: str, hotel_stars: int) -> List[HotelReview]:
        """
        Pass 2 — Collects all Fair + Poor + Very Poor reviews for PB4 (ABSA).

        Iterates through each negative score category separately to exhaustion.
        Deduplicates by source_id across categories.
        """
        hotel_name = self._navigate_to_reviews(url)

        counts = self.get_review_counts()
        print(f"   Hotel: '{hotel_name}' — Negative review breakdown:")
        for filter_val, label in NEGATIVE_SCORE_FILTERS:
            raw_key = f"_raw_{filter_val}"
            c = counts.get(raw_key, 0)
            print(f"      {label}: {c} available")

        self._set_sort_order(SORT_NEWEST_FIRST)

        all_negative: List[HotelReview] = []
        seen_ids: set = set()

        for filter_val, label in NEGATIVE_SCORE_FILTERS:
            raw_key = f"_raw_{filter_val}"
            available = counts.get(raw_key, 0)

            if available == 0:
                print(f"\n   -> Skipping {label} (0 reviews available).")
                continue

            print(f"\n   -> Scraping {label} ({available} reviews available)...")
            self._set_score_filter(filter_val)

            batch = self._paginate_and_collect(hotel_name, target_count=None)

            # Deduplicate across categories
            new_reviews = [r for r in batch if r.source_id not in seen_ids]
            seen_ids.update(r.source_id for r in new_reviews)
            all_negative.extend(new_reviews)

            print(f"   -> {len(new_reviews)} unique reviews added from {label}.")

        for r in all_negative:
            r.hotel_stars = hotel_stars

        print(f"\n   ✓ Pass 2 complete: {len(all_negative)} negative reviews total.")
        return all_negative
