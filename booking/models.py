from dataclasses import dataclass

@dataclass
class HotelReview:
    """
    Data model representing a single hotel review from Booking.com.
    """
    source_id: str
    hotel_name: str
    rating_score: float
    title: str
    pos_text: str
    neg_text: str
    stay_duration: int
    stay_date: str
    room_type: str
    traveler_type: str
    hotel_stars: int = 0      
    hotel_district: str = ""
    reviewer_country: str = "" 