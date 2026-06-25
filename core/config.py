"""
Core configuration and shared data models
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


def get_groq_key() -> str:
    key = os.getenv("GROQ_API_KEY","")
    if not key:
        raise ValueError("GROQ_API_KEY not set. Add to .env file.")
    return key

def get_scrapedo_token() -> str:
    return os.getenv("SCRAPEDO_TOKEN", "")

def get_gov_key() -> str:
    return os.getenv("DATA_GOV_IN_KEY", "demo")


@dataclass
class BusinessRecord:
    business_name: str = ""
    address: str = ""
    phone: str = ""
    email: str = ""
    website: str = ""
    working_hours: str = ""
    rating: str = ""
    review_count: str = ""
    services: List[str] = field(default_factory=list)
    specialties: List[str] = field(default_factory=list)
    license_information: str = ""
    certifications: List[str] = field(default_factory=list)
    awards: List[str] = field(default_factory=list)
    social_profiles: List[str] = field(default_factory=list)
    images_urls: List[str] = field(default_factory=list)
    source_urls: Dict[str, str] = field(default_factory=dict)
    verification_score: float = 0.0
    reliability_score: float = 0.0
    sources_count: int = 0
    conflicts: List[str] = field(default_factory=list)
    phone_sources: List[Dict] = field(default_factory=list)
    email_sources: List[Dict] = field(default_factory=list)
    source_url: str = ""

    def to_dict(self) -> dict:
        return {
            "business_name": self.business_name, "address": self.address,
            "phone": self.phone, "email": self.email, "website": self.website,
            "working_hours": self.working_hours, "rating": self.rating,
            "review_count": self.review_count, "services": self.services,
            "specialties": self.specialties,
            "license_information": self.license_information,
            "certifications": self.certifications, "awards": self.awards,
            "social_profiles": self.social_profiles,
            "images_urls": self.images_urls, "source_urls": self.source_urls,
            "verification_score": self.verification_score,
            "reliability_score": self.reliability_score,
            "sources_count": self.sources_count,
            "conflicts": self.conflicts, "source_url": self.source_url,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BusinessRecord":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class ResearchReport:
    query: str = ""
    category: str = ""
    location: str = ""
    businesses_found: int = 0
    businesses_verified: int = 0
    duplicates_removed: int = 0
    sources_searched: int = 0
    duration_seconds: float = 0.0
    businesses: List[dict] = field(default_factory=list)
    data_quality: Dict[str, str] = field(default_factory=dict)
    ai_summary: str = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class PipelineEvent(BaseModel):
    event_type: str
    phase: int = 0
    phase_name: str = ""
    message: str = ""
    data: Optional[dict] = None
    progress: float = 0.0


SCRAPER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

ALL_SOURCES = [
    "duckduckgo","yelp","yellowpages","justdial","sulekha",
    "linkedin","facebook","openstreetmap","overpass",
    "nominatim","data_gov_in","nhfr","wikidata",
]
