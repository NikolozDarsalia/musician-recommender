"""
Service layer for recommendation operations
"""

from typing import Dict, List, Optional
import logging
from pathlib import Path
from recommender_pipeline.inference.recommendation_service import RecommendationService
from recommender_pipeline.api.schemas.rating_storage import RatingStorageService

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[4]
MODEL_DIR = BASE_DIR / "notebooks" / "models"
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
MATRIX_BUILDER_PATH = MODEL_DIR / "matrix_builder.pkl"
ITEM_FEATURES_PATH = MODEL_DIR / "artist_features.pkl"
ITEM_FEATURES_BUILDER_PATH = MODEL_DIR / "features_builder.pkl"
ARTIST_METADATA_PATH = DATA_DIR / "artist_matching_for_api.parquet"


class RecommendationAPIService:
    """Service for handling recommendation logic"""

    # Weight mapping: rank → weight
    RANK_TO_WEIGHT = {
        1: 1.0,  # Most preferred
        2: 0.9,
        3: 0.8,
        4: 0.7,
        5: 0.6,
        6: 0.5,
        7: 0.4,
        8: 0.3,
        9: 0.2,
        10: 0.1,  # Least preferred
    }

    def __init__(
        self,
        model_path=str(MODEL_PATH),
        matrix_builder_path=str(MATRIX_BUILDER_PATH),
        item_features_path=str(ITEM_FEATURES_PATH),
        item_features_builder_path=str(ITEM_FEATURES_BUILDER_PATH),
        artist_metadata_path=str(ARTIST_METADATA_PATH),
        storage_dir: str = "recommender_pipeline/inference/user_ratings",
    ):
        """
        Initialize recommendation service

        Args:
            model_path: Path to trained LightFM model
            matrix_builder_path: Path to matrix builder
            item_features_path: Path to item features
            item_features_builder_path: Path to features builder
            artist_metadata_path: Path to artist metadata
            storage_dir: Directory for storing user ratings
        """
        try:
            self.recommendation_service = RecommendationService(
                model_path=model_path,
                matrix_builder_path=matrix_builder_path,
                item_features_path=item_features_path,
                item_features_builder_path=item_features_builder_path,
                artist_metadata_path=artist_metadata_path,
                rank_to_weight=self.RANK_TO_WEIGHT,
            )

            self.storage_service = RatingStorageService(storage_dir=storage_dir)

            self.survey_artists = self.recommendation_service.get_survey_artists()

            logger.info("RecommendationAPIService initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RecommendationAPIService: {str(e)}")
            raise

    def get_survey_artists(self) -> Dict:
        """Get the list of survey artists"""
        return self.survey_artists

    def validate_artist_ids(
        self, ratings: Dict[int, int]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that all artist IDs exist in survey

        Args:
            ratings: Dictionary mapping artist_id to rank

        Returns:
            Tuple of (is_valid, error_message)
        """
        valid_artist_ids = set(self.survey_artists.keys())
        submitted_ids = set(ratings.keys())

        invalid_ids = submitted_ids - valid_artist_ids

        if invalid_ids:
            return False, f"Invalid artist IDs: {sorted(invalid_ids)}"

        return True, None

    def save_user_ratings(
        self, user_id: str, ratings: Dict[int, int]
    ) -> tuple[bool, str]:
        """
        Validate and save user ratings

        Args:
            user_id: Unique user identifier
            ratings: Dictionary mapping artist_id to rank

        Returns:
            Tuple of (success, message)
        """
        # Validate artist IDs
        is_valid, error_msg = self.validate_artist_ids(ratings)
        if not is_valid:
            return False, error_msg

        # Save ratings
        success = self.storage_service.save_ratings(user_id, ratings)

        if success:
            return True, f"Successfully saved {len(ratings)} ratings for user {user_id}"
        else:
            return False, "Failed to save ratings due to storage error"

    def generate_recommendations(
        self, user_id: str, top_k: int = 10
    ) -> tuple[bool, Optional[List[Dict]], Optional[str]]:
        """
        Generate recommendations for a user based on their saved ratings

        Args:
            user_id: Unique user identifier
            top_k: Number of recommendations to return

        Returns:
            Tuple of (success, recommendations_list, error_message)
        """
        # Check if user has ratings
        if not self.storage_service.user_has_ratings(user_id):
            return (
                False,
                None,
                f"No ratings found for user {user_id}. Please submit ratings first.",
            )

        # Load user ratings
        ratings = self.storage_service.load_ratings(user_id)

        if ratings is None:
            return False, None, "Failed to load user ratings"

        if len(ratings) == 0:
            return False, None, "User ratings are empty"

        try:
            # Generate recommendations using the recommendation service
            recommendations = self.recommendation_service.recommend_from_survey(
                survey_responses=ratings, k=top_k
            )

            return True, recommendations, None

        except Exception as e:
            logger.error(
                f"Failed to generate recommendations for user {user_id}: {str(e)}"
            )
            return False, None, f"Failed to generate recommendations: {str(e)}"
