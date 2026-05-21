from dataclasses import dataclass


@dataclass
class HotelReview:
    """
    Data model representing a single hotel review from Booking.com.
    Specification source: AGENT_CONTEXT.md (Section 4 — Model Danych).
    """

    # Identification
    source_id: str  # MD5 hash (hotel_name|reviewer_name|title) — dedup key
    hotel_name: str  # Extracted automatically from page h2 header

    # Date of stay — used for filtering Jan 2024 – May 2026 in preprocessing
    stay_date: str  # Format: "Month YYYY" e.g. "November 2024"

    # Independent variables — for PB1, PB2, PB3
    hotel_stars: (
        int  # Star category: 3, 4, or 5 (assigned from target_hotels.txt sections)
    )
    traveler_type: str  # Business, Couple, Family, Solo, Other
    reviewer_country: str  # Reviewer country (extracted from flag image alt text)

    # Dependent variable — for PB1, PB2, PB3, PB4
    rating_score: (
        float  # Overall score — continuous scale 1–10 (guest's autonomous decision)
    )

    # Qualitative fields for NLP — for PB4 (ABSA)
    title: str  # Review headline
    pos_text: str  # Positive section ("What I liked")
    neg_text: str  # Negative section ("What I didn't like") — primary ABSA target
