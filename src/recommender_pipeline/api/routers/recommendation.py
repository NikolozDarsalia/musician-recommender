"""
API routes for recommendation endpoints
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict
import logging

from recommender_pipeline.api.schemas.recommendation import (
    RatingSubmission,
    RatingResponse,
    RecommendationRequest,
    RecommendationResponse,
    ArtistRecommendation,
)

from recommender_pipeline.api.services.recommendation import RecommendationAPIService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["recommendations"])

# Initialize service (in production, this would be dependency injection)
recommendation_service = RecommendationAPIService()


@router.post(
    "/ratings",
    response_model=RatingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit user ratings",
    description="Submit ratings for survey artists. Ratings will be saved for generating recommendations.",
)
async def submit_ratings(rating_data: RatingSubmission) -> RatingResponse:
    """
    API 1: Submit user ratings for artists

    Args:
        rating_data: User ratings submission

    Returns:
        RatingResponse with success status

    Raises:
        HTTPException: If validation fails or storage error occurs
    """
    try:
        # Convert to dictionary format
        ratings_dict = rating_data.to_dict()

        # Save ratings through service
        success, message = recommendation_service.save_user_ratings(
            user_id=rating_data.user_id, ratings=ratings_dict
        )

        if not success:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

        return RatingResponse(
            success=True,
            message=message,
            user_id=rating_data.user_id,
            ratings_count=len(ratings_dict),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in submit_ratings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )


@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
    summary="Get personalized recommendations",
    description="Generate personalized artist recommendations based on previously submitted ratings.",
)
async def get_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """
    API 2: Get personalized recommendations for a user

    Args:
        request: Recommendation request with user_id and top_k

    Returns:
        RecommendationResponse with personalized recommendations

    Raises:
        HTTPException: If user has no ratings or generation fails
    """
    try:
        # Generate recommendations through service
        success, recommendations, error_msg = (
            recommendation_service.generate_recommendations(
                user_id=request.user_id, top_k=request.top_k
            )
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND
                if "No ratings found" in error_msg
                else status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg,
            )

        # Convert to response schema
        recommendation_list = [
            ArtistRecommendation(
                rank=rec["rank"],
                artist_id=rec["artist_id"],
                artist_name=rec["artist_name"],
                score=rec["score"],
            )
            for rec in recommendations
        ]

        return RecommendationResponse(
            success=True,
            message=f"Generated {len(recommendation_list)} recommendations",
            user_id=request.user_id,
            recommendations=recommendation_list,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )


@router.get(
    "/survey-artists",
    response_model=Dict,
    summary="Get survey artists",
    description="Get the list of artists available in the rating survey.",
)
async def get_survey_artists() -> Dict:
    """
    Get the list of artists for the rating survey

    Returns:
        Dictionary of survey artists with metadata
    """
    try:
        survey_artists = recommendation_service.get_survey_artists()

        # Format response
        formatted_artists = {
            str(artist_id): {"artist_id": artist_id, "name": name, "genre": genre}
            for artist_id, (name, genre) in survey_artists.items()
        }

        return {
            "success": True,
            "count": len(formatted_artists),
            "artists": formatted_artists,
        }

    except Exception as e:
        logger.error(f"Error getting survey artists: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve survey artists: {str(e)}",
        )
