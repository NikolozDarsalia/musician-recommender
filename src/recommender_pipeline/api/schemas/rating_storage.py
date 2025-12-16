"""
Service for storing and retrieving user ratings
"""

import pickle
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RatingStorageService:
    """Handles persistence of user ratings as pickle files"""

    def __init__(
        self, storage_dir: str = "recommender_pipeline/inference/user_ratings"
    ):
        """
        Initialize rating storage service

        Args:
            storage_dir: Directory to store user rating files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Rating storage initialized at: {self.storage_dir}")

    def _get_user_file_path(self, user_id: str) -> Path:
        """Get the file path for a user's ratings"""
        # Sanitize user_id to prevent path traversal
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ("-", "_"))
        return self.storage_dir / f"{safe_user_id}_ratings.pkl"

    def save_ratings(self, user_id: str, ratings: Dict[int, int]) -> bool:
        """
        Save user ratings to a pickle file

        Args:
            user_id: Unique user identifier
            ratings: Dictionary mapping artist_id to rank

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self._get_user_file_path(user_id)

            with open(file_path, "wb") as f:
                pickle.dump(ratings, f)

            logger.info(f"Saved ratings for user {user_id}: {len(ratings)} artists")
            return True

        except Exception as e:
            logger.error(f"Failed to save ratings for user {user_id}: {str(e)}")
            return False

    def load_ratings(self, user_id: str) -> Optional[Dict[int, int]]:
        """
        Load user ratings from pickle file

        Args:
            user_id: Unique user identifier

        Returns:
            Dictionary mapping artist_id to rank, or None if not found
        """
        try:
            file_path = self._get_user_file_path(user_id)

            if not file_path.exists():
                logger.warning(f"No ratings found for user {user_id}")
                return None

            with open(file_path, "rb") as f:
                ratings = pickle.load(f)

            logger.info(f"Loaded ratings for user {user_id}: {len(ratings)} artists")
            return ratings

        except Exception as e:
            logger.error(f"Failed to load ratings for user {user_id}: {str(e)}")
            return None

    def user_has_ratings(self, user_id: str) -> bool:
        """
        Check if a user has submitted ratings

        Args:
            user_id: Unique user identifier

        Returns:
            True if user has ratings, False otherwise
        """
        file_path = self._get_user_file_path(user_id)
        return file_path.exists()

    def delete_ratings(self, user_id: str) -> bool:
        """
        Delete user ratings

        Args:
            user_id: Unique user identifier

        Returns:
            True if successful or file didn't exist, False on error
        """
        try:
            file_path = self._get_user_file_path(user_id)

            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted ratings for user {user_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to delete ratings for user {user_id}: {str(e)}")
            return False
