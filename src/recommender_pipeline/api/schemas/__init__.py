# recommender_pipeline/api/__init__.py
"""API module for music recommendation system"""

# recommender_pipeline/api/schemas/__init__.py
"""Pydantic schemas for API"""
from .recommendation import (
    RatingSubmission,
    RatingResponse,
    RecommendationRequest,
    RecommendationResponse,
    ArtistRecommendation,
    ErrorResponse,
)

__all__ = [
    "RatingSubmission",
    "RatingResponse",
    "RecommendationRequest",
    "RecommendationResponse",
    "ArtistRecommendation",
    "ErrorResponse",
]

# recommender_pipeline/api/routers/__init__.py
"""API routers"""
from . import recommendation

__all__ = ["recommendation"]

# recommender_pipeline/api/services/__init__.py
"""Business logic services"""
from .rating_storage import RatingStorageService
# from .recommendation import RecommendationAPIService

__all__ = ["RatingStorageService", "RecommendationAPIService"]
