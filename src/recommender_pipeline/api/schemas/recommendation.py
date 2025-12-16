"""
Pydantic schemas for recommendation API
"""

from typing import Dict, List
from pydantic import BaseModel, Field, field_validator


class ArtistRating(BaseModel):
    """Single artist rating"""

    artist_id: int = Field(..., description="Artist ID from the survey")
    rank: int = Field(
        ...,
        ge=1,
        le=10,
        description="Rank from 1 (most preferred) to 10 (least preferred)",
    )


class RatingSubmission(BaseModel):
    """User rating submission for survey artists"""

    user_id: str = Field(
        ..., min_length=1, max_length=100, description="Unique user identifier"
    )
    ratings: List[ArtistRating] = Field(
        ..., min_items=1, max_items=10, description="List of artist ratings"
    )

    @field_validator("ratings")
    @classmethod
    def validate_ratings(cls, v: List[ArtistRating]) -> List[ArtistRating]:
        """Validate ratings list"""
        # Check for duplicate artist IDs
        artist_ids = [r.artist_id for r in v]
        if len(artist_ids) != len(set(artist_ids)):
            raise ValueError("Duplicate artist IDs found in ratings")

        # Check for duplicate ranks
        ranks = [r.rank for r in v]
        if len(ranks) != len(set(ranks)):
            raise ValueError(
                "Duplicate ranks found. Each artist must have a unique rank"
            )

        return v

    def to_dict(self) -> Dict[int, int]:
        """Convert to dictionary format {artist_id: rank}"""
        return {rating.artist_id: rating.rank for rating in self.ratings}


class RatingResponse(BaseModel):
    """Response after rating submission"""

    success: bool
    message: str
    user_id: str
    ratings_count: int


class RecommendationRequest(BaseModel):
    """Request for personalized recommendations"""

    user_id: str = Field(
        ..., min_length=1, max_length=100, description="Unique user identifier"
    )
    top_k: int = Field(
        default=10, ge=1, le=50, description="Number of recommendations to return"
    )


class ArtistRecommendation(BaseModel):
    """Single artist recommendation"""

    rank: int = Field(..., description="Recommendation rank")
    artist_id: int = Field(..., description="Artist ID")
    artist_name: str = Field(..., description="Artist name")
    score: float = Field(..., description="Recommendation score")


class RecommendationResponse(BaseModel):
    """Response with personalized recommendations"""

    success: bool
    message: str
    user_id: str
    recommendations: List[ArtistRecommendation]


class ErrorResponse(BaseModel):
    """Standard error response"""

    success: bool = False
    error: str
    detail: str = None
