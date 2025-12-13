from typing import Dict, Set, Optional, Tuple
import pandas as pd


class ArtistIDMapper:
    """
    Manages artist ID mappings between different data sources and creates unified IDs.
    
    This class handles the creation of a unified artist ID system by:
    1. Using matched artist data to link primary and secondary data sources
    2. Creating new unified IDs for unmatched secondary source artists
    3. Providing lookup functionality between different ID systems
    
    The mapper ensures that:
    - Matched artists get the same unified ID across data sources
    - Unmatched secondary source artists get new unique unified IDs
    - All ID mappings are consistent and reversible
    """
    
    def __init__(self, start_id: int = 1):
        """
        Initialize the ID mapper.
        
        Args:
            start_id: Starting ID for new unified IDs (default: 1)
        """
        self.start_id = start_id
        self.next_available_id = start_id
        
        # Mapping dictionaries
        self.primary_to_unified: Dict[int, int] = {}
        self.secondary_to_unified: Dict[str, int] = {}  # Secondary can use string or int IDs
        self.unified_to_primary: Dict[int, int] = {}
        self.unified_to_secondary: Dict[int, str] = {}
        
        # Track which IDs are used
        self.used_unified_ids: Set[int] = set()
        
    def fit(self, 
            matched_artists_df: pd.DataFrame,
            primary_artists_df: pd.DataFrame,
            secondary_artists_df: pd.DataFrame,
            primary_id_col: str = 'artistID',
            secondary_id_col: str = 'artist_name',
            matched_primary_col: str = 'artistID',
            matched_secondary_col: str = 'artist_name') -> 'ArtistIDMapper':
        """
        Create unified ID mappings from matched and unmatched artist data.
        
        Args:
            matched_artists_df: DataFrame with matched primary-secondary artist pairs
            primary_artists_df: DataFrame with all primary source artists
            secondary_artists_df: DataFrame with all secondary source artists  
            primary_id_col: Column name for primary source artist ID
            secondary_id_col: Column name for secondary source artist ID
            matched_primary_col: Column name for primary ID in matched data
            matched_secondary_col: Column name for secondary ID in matched data
            
        Returns:
            self for method chaining
        """
        # Reset mappings
        self._reset_mappings()
        
        # Step 1: Process matched artists - they get the primary ID as unified ID
        self._process_matched_artists(matched_artists_df, matched_primary_col, matched_secondary_col)
        
        # Step 2: Process unmatched primary artists
        self._process_unmatched_primary(primary_artists_df, primary_id_col)
        
        # Step 3: Process unmatched secondary artists
        self._process_unmatched_secondary(secondary_artists_df, secondary_id_col)
        
        return self
    
    def _reset_mappings(self):
        """Reset all internal mappings."""
        self.next_available_id = self.start_id
        self.primary_to_unified.clear()
        self.secondary_to_unified.clear()
        self.unified_to_primary.clear()
        self.unified_to_secondary.clear()
        self.used_unified_ids.clear()
    
    def _process_matched_artists(self, matched_df: pd.DataFrame, 
                                primary_col: str, secondary_col: str):
        """Process matched artists - assign new global unified IDs."""
        for _, row in matched_df.iterrows():
            primary_id = row[primary_col]
            secondary_id = row[secondary_col]
            
            # Create new global unified ID for matched artists
            unified_id = self._get_next_available_id()
            
            # Create bidirectional mappings
            self.primary_to_unified[primary_id] = unified_id
            self.secondary_to_unified[secondary_id] = unified_id
            self.unified_to_primary[unified_id] = primary_id
            self.unified_to_secondary[unified_id] = secondary_id
    
    def _process_unmatched_primary(self, primary_df: pd.DataFrame, id_col: str):
        """Process unmatched primary source artists."""
        for _, row in primary_df.iterrows():
            primary_id = row[id_col]
            
            # Skip if already processed in matched artists
            if primary_id in self.primary_to_unified:
                continue
                
            # Create new global unified ID
            unified_id = self._get_next_available_id()
            
            # Create mappings
            self.primary_to_unified[primary_id] = unified_id
            self.unified_to_primary[unified_id] = primary_id
    
    def _process_unmatched_secondary(self, secondary_df: pd.DataFrame, id_col: str):
        """Process unmatched secondary source artists."""
        for _, row in secondary_df.iterrows():
            secondary_id = row[id_col]
            
            # Skip if already processed in matched artists
            if secondary_id in self.secondary_to_unified:
                continue
                
            # Create new global unified ID
            unified_id = self._get_next_available_id()
            
            # Create mappings
            self.secondary_to_unified[secondary_id] = unified_id
            self.unified_to_secondary[unified_id] = secondary_id
    
    def _get_next_available_id(self) -> int:
        """Get the next available unified ID."""
        current_id = self.next_available_id
        self.used_unified_ids.add(current_id)
        self.next_available_id += 1
        return current_id
    
    def get_unified_id(self, source_id, source: str = 'auto') -> Optional[int]:
        """
        Get unified ID for a given source ID.
        
        Args:
            source_id: The original ID from primary or secondary source
            source: 'primary', 'secondary', or 'auto' to detect
            
        Returns:
            Unified ID or None if not found
        """
        if source == 'auto':
            # Try both mappings
            if isinstance(source_id, int) and source_id in self.primary_to_unified:
                return self.primary_to_unified[source_id]
            elif str(source_id) in self.secondary_to_unified:
                return self.secondary_to_unified[str(source_id)]
            return None
        elif source == 'primary':
            return self.primary_to_unified.get(source_id)
        elif source == 'secondary':
            return self.secondary_to_unified.get(str(source_id))
        else:
            raise ValueError(f"Invalid source: {source}. Must be 'primary', 'secondary', or 'auto'")
    
    def get_source_id(self, unified_id: int, target_source: str) -> Optional:
        """
        Get original source ID for a unified ID.
        
        Args:
            unified_id: The unified ID
            target_source: 'primary' or 'secondary'
            
        Returns:
            Original source ID or None if not found
        """
        if target_source == 'primary':
            return self.unified_to_primary.get(unified_id)
        elif target_source == 'secondary':
            return self.unified_to_secondary.get(unified_id)
        else:
            raise ValueError(f"Invalid target_source: {target_source}. Must be 'primary' or 'secondary'")
    
    def transform_dataframe(self, df: pd.DataFrame, 
                           id_col: str, 
                           source: str,
                           unified_id_col: str = 'unified_artist_id') -> pd.DataFrame:
        """
        Add unified IDs to a dataframe.
        
        Args:
            df: Input dataframe
            id_col: Column containing source IDs
            source: 'primary' or 'secondary'
            unified_id_col: Name for new unified ID column
            
        Returns:
            DataFrame with added unified ID column
        """
        result_df = df.copy()
        result_df[unified_id_col] = result_df[id_col].apply(
            lambda x: self.get_unified_id(x, source)
        )
        return result_df
    
    def get_mapping_summary(self) -> Dict:
        """
        Get summary statistics about the mappings.
        
        Returns:
            Dictionary with mapping statistics
        """
        return {
            'total_unified_ids': len(self.used_unified_ids),
            'primary_artists': len(self.primary_to_unified),
            'secondary_artists': len(self.secondary_to_unified),
            'matched_artists': len(set(self.unified_to_primary.keys()) & 
                                 set(self.unified_to_secondary.keys())),
            'unmatched_primary': len(self.unified_to_primary) - 
                               len(set(self.unified_to_primary.keys()) & 
                                   set(self.unified_to_secondary.keys())),
            'unmatched_secondary': len(self.unified_to_secondary) - 
                                len(set(self.unified_to_primary.keys()) & 
                                    set(self.unified_to_secondary.keys())),
            'id_range': (min(self.used_unified_ids), max(self.used_unified_ids)) 
                       if self.used_unified_ids else (None, None)
        }
    
    def export_mappings(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Export mapping tables as DataFrames.
        
        Returns:
            Tuple of (primary_mappings_df, secondary_mappings_df)
        """
        primary_df = pd.DataFrame([
            {'primary_id': primary_id, 'unified_id': unified_id}
            for primary_id, unified_id in self.primary_to_unified.items()
        ])
        
        secondary_df = pd.DataFrame([
            {'secondary_id': secondary_id, 'unified_id': unified_id}
            for secondary_id, unified_id in self.secondary_to_unified.items()
        ])
        
        return primary_df, secondary_df