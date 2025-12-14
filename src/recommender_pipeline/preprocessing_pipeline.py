from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from ..interfaces.base_preprocessor import BasePreprocessor
from ..interfaces.base_pipeline import BasePipeline


class PreprocessingPipeline(BasePipeline, BasePreprocessor):
    """
    A pipeline for chaining multiple preprocessing steps together.
    
    Combines multiple preprocessors (like FillMissingSpotify and FeatureScaler)
    into a single pipeline that can be fit and transformed as one unit.
    
    Parameters
    ----------
    steps : List[Tuple[str, BasePreprocessor]]
        List of (name, preprocessor) tuples that define the pipeline steps.
        Each preprocessor must implement fit() and transform() methods.
    
    memory : str or object, optional
        Used to cache fitted transformers. Can be a string path or joblib.Memory object.
        
    verbose : bool, default=False
        If True, print the step names and execution times.
    """
    
    def __init__(
        self, 
        steps: List[Tuple[str, BasePreprocessor]], 
        memory: Optional[Any] = None,
        verbose: bool = False
    ):
        super().__init__()
        self.steps = steps
        self.memory = memory  # Keep for future caching implementation
        self.verbose = verbose
        
        # Store step names and preprocessors for easy access
        self.named_steps = dict(steps)
        self.step_names = [name for name, _ in steps]
    
    def fit(self, X: pd.DataFrame, y=None) -> "PreprocessingPipeline":
        """
        Fit all preprocessors in the pipeline sequentially.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to fit the pipeline on.
        y : ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : PreprocessingPipeline
            Returns self for chaining.
        """
        if self.verbose:
            print(f"Fitting preprocessing pipeline with {len(self.steps)} steps...")
        
        # Fit each step sequentially, transforming data for the next step
        X_transformed = X.copy()
        for name, preprocessor in self.steps:
            if self.verbose:
                print(f"  Fitting step '{name}'...")
            
            preprocessor.fit(X_transformed, y)
            X_transformed = preprocessor.transform(X_transformed)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data through all pipeline steps sequentially.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.
            
        Returns
        -------
        pd.DataFrame
            Transformed data after applying all pipeline steps.
        """
        if self.verbose:
            print("Applying preprocessing pipeline transformation...")
        
        # Transform through each step sequentially
        X_transformed = X.copy()
        for name, preprocessor in self.steps:
            if self.verbose:
                print(f"  Applying step '{name}'...")
            
            X_transformed = preprocessor.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit the pipeline and transform the data in one step.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to fit and transform.
        y : ignored
            Not used, present for API consistency.
            
        Returns
        -------
        pd.DataFrame
            Transformed data after fitting and applying all pipeline steps.
        """
        return self.fit(X, y).transform(X)
    
    def get_step(self, name: str) -> BasePreprocessor:
        """
        Get a specific preprocessor step by name.
        
        Parameters
        ----------
        name : str
            Name of the pipeline step to retrieve.
            
        Returns
        -------
        BasePreprocessor
            The preprocessor object for the specified step.
        """
        if name not in self.named_steps:
            raise KeyError(f"Step '{name}' not found in pipeline. Available steps: {self.step_names}")
        
        return self.named_steps[name]
    
    def set_params(self, **params) -> "PreprocessingPipeline":
        """
        Set parameters for specific pipeline steps.
        
        Parameters should be specified as 'step_name__parameter_name'.
        
        Parameters
        ----------
        **params : dict
            Dictionary of parameters to set for pipeline steps.
            
        Returns
        -------
        self : PreprocessingPipeline
            Returns self for chaining.
        """
        for param_name, param_value in params.items():
            if '__' in param_name:
                step_name, attr_name = param_name.split('__', 1)
                if step_name in self.named_steps:
                    setattr(self.named_steps[step_name], attr_name, param_value)
                else:
                    raise ValueError(f"Step '{step_name}' not found in pipeline. Available steps: {self.step_names}")
            else:
                # Set parameter on this pipeline object
                setattr(self, param_name, param_value)
        
        return self
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for all pipeline steps.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters for all nested objects.
            
        Returns
        -------
        dict
            Dictionary of all pipeline parameters.
        """
        params = {
            'steps': self.steps,
            'memory': self.memory,
            'verbose': self.verbose
        }
        
        if deep:
            # Add parameters from each step
            for step_name, preprocessor in self.steps:
                if hasattr(preprocessor, '__dict__'):
                    for attr_name, attr_value in preprocessor.__dict__.items():
                        if not attr_name.startswith('_'):
                            params[f'{step_name}__{attr_name}'] = attr_value
        
        return params
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline structure and steps.
        
        Returns
        -------
        dict
            Dictionary with pipeline information.
        """
        step_info = []
        for name, preprocessor in self.steps:
            step_info.append({
                'name': name,
                'class': type(preprocessor).__name__,
                'parameters': getattr(preprocessor, '__dict__', {})
            })
        
        return {
            'n_steps': len(self.steps),
            'step_names': self.step_names,
            'steps_info': step_info,
            'memory': self.memory,
            'verbose': self.verbose
        }
    
    def __repr__(self) -> str:
        """String representation of the pipeline."""
        return f"PreprocessingPipeline(steps={self.step_names})"
    
    # Implementation of BasePipeline abstract methods
    
    def load_data(self, paths_dict: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Load data from specified paths for preprocessing.
        
        Parameters
        ----------
        paths_dict : dict
            Dictionary with keys as data types and values as file paths.
            Expected keys: 'features', 'interactions', 'items_meta', etc.
            
        Returns
        -------
        dict
            Dictionary of loaded DataFrames with same keys as input.
            
        Examples
        --------
        >>> pipeline = PreprocessingPipeline(steps)
        >>> data = pipeline.load_data({
        ...     'features': 'artist_audio_features.parquet',
        ...     'interactions': 'user_artists.dat'
        ... })
        """
        loaded_data = {}
        
        for data_type, file_path in paths_dict.items():
            try:
                if file_path.endswith('.parquet'):
                    loaded_data[data_type] = pd.read_parquet(file_path)
                elif file_path.endswith('.csv'):
                    loaded_data[data_type] = pd.read_csv(file_path)
                elif file_path.endswith('.dat'):
                    # Assume tab-separated for .dat files (common in LastFM data)
                    loaded_data[data_type] = pd.read_csv(file_path, sep='\t')
                else:
                    # Default to CSV
                    loaded_data[data_type] = pd.read_csv(file_path)
                    
                if self.verbose:
                    print(f"Loaded {data_type} from {file_path}: {loaded_data[data_type].shape}")
                    
            except Exception as e:
                print(f"Error loading {data_type} from {file_path}: {e}")
                raise
        
        return loaded_data
    
    def run(self, data: Optional[pd.DataFrame] = None, paths_dict: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        Parameters
        ----------
        data : pd.DataFrame, optional
            Input data to process. If None, must provide paths_dict.
            
        paths_dict : dict, optional
            Dictionary of file paths to load data from. If None, must provide data.
            
        Returns
        -------
        pd.DataFrame
            Processed data after applying all pipeline steps.
            
        Examples
        --------
        >>> # Run with pre-loaded data
        >>> result = pipeline.run(data=spotify_features)
        
        >>> # Run with file paths
        >>> result = pipeline.run(paths_dict={'features': 'features.parquet'})
        """
        if data is None and paths_dict is None:
            raise ValueError("Must provide either data or paths_dict")
        
        if data is None:
            loaded_data = self.load_data(paths_dict)
            # Assume we want to process the 'features' data
            data = loaded_data.get('features', list(loaded_data.values())[0])
        
        if self.verbose:
            print("Running preprocessing pipeline...")
            print(f"Input data shape: {data.shape}")
        
        # Apply the pipeline
        processed_data = self.fit_transform(data)
        
        if self.verbose:
            print(f"Output data shape: {processed_data.shape}")
            print("Preprocessing pipeline completed!")
        
        return processed_data
    
    def add_step(self, step: Tuple[str, BasePreprocessor], position: Optional[int] = None) -> "PreprocessingPipeline":
        """
        Add a new preprocessing step to the pipeline.
        
        Parameters
        ----------
        step : tuple
            Tuple of (step_name, preprocessor) to add.
            
        position : int, optional
            Position to insert the step. If None, appends to the end.
            
        Returns
        -------
        self : PreprocessingPipeline
            Returns self for chaining.
        """
        step_name, preprocessor = step
        
        if position is None:
            # Append to end
            self.steps.append(step)
        else:
            # Insert at specific position
            self.steps.insert(position, step)
        
        # Update pipeline components
        self._update_pipeline()
        
        if self.verbose:
            print(f"Added step '{step_name}' at position {position or len(self.steps)-1}")
        
        return self
    
    def remove_step(self, step_name: str) -> "PreprocessingPipeline":
        """
        Remove a preprocessing step from the pipeline.
        
        Parameters
        ----------
        step_name : str
            Name of the step to remove.
            
        Returns
        -------
        self : PreprocessingPipeline
            Returns self for chaining.
        """
        # Find and remove the step
        original_length = len(self.steps)
        self.steps = [(name, preprocessor) for name, preprocessor in self.steps if name != step_name]
        
        if len(self.steps) == original_length:
            raise ValueError(f"Step '{step_name}' not found in pipeline. Available steps: {self.step_names}")
        
        # Update pipeline components
        self._update_pipeline()
        
        if self.verbose:
            print(f"Removed step '{step_name}' from pipeline")
        
        return self
    
    def replace_step(self, step_name: str, new_step: Tuple[str, BasePreprocessor]) -> "PreprocessingPipeline":
        """
        Replace an existing preprocessing step with a new one.
        
        Parameters
        ----------
        step_name : str
            Name of the step to replace.
            
        new_step : tuple
            Tuple of (new_step_name, new_preprocessor) to replace with.
            
        Returns
        -------
        self : PreprocessingPipeline
            Returns self for chaining.
        """
        new_step_name, new_preprocessor = new_step
        
        # Find and replace the step
        step_found = False
        for i, (name, preprocessor) in enumerate(self.steps):
            if name == step_name:
                self.steps[i] = new_step
                step_found = True
                break
        
        if not step_found:
            raise ValueError(f"Step '{step_name}' not found in pipeline. Available steps: {self.step_names}")
        
        # Update pipeline components
        self._update_pipeline()
        
        if self.verbose:
            print(f"Replaced step '{step_name}' with '{new_step_name}'")
        
        return self
    
    def _update_pipeline(self):
        """Update internal pipeline components after step modifications."""
        self.named_steps = dict(self.steps)
        self.step_names = [name for name, _ in self.steps]


def create_spotify_preprocessing_pipeline(
    fill_missing_params: Optional[Dict] = None,
    feature_scaler_params: Optional[Dict] = None,
    verbose: bool = False
) -> PreprocessingPipeline:
    """
    Factory function to create a standard Spotify feature preprocessing pipeline.
    
    Creates a pipeline with FillMissingSpotify followed by FeatureScaler.
    
    Parameters
    ----------
    fill_missing_params : dict, optional
        Parameters for FillMissingSpotify. If None, uses default parameters.
        
    feature_scaler_params : dict, optional
        Parameters for FeatureScaler. If None, uses default parameters.
        
    verbose : bool, default=False
        If True, print step information during execution.
        
    Returns
    -------
    PreprocessingPipeline
        Configured pipeline with missing value filling and feature scaling.
        
    Examples
    --------
    >>> # Create default pipeline
    >>> pipeline = create_spotify_preprocessing_pipeline()
    >>> processed_data = pipeline.fit_transform(spotify_features)
    
    >>> # Create custom pipeline
    >>> pipeline = create_spotify_preprocessing_pipeline(
    ...     fill_missing_params={'strategy': 'mixed', 'genre_col': 'track_genre_top'},
    ...     feature_scaler_params={'method': 'standard', 'clip_outliers': True}
    ... )
    """
    from .spotify_missing_filler import FillMissingSpotify
    from .feature_scaler import FeatureScaler
    
    # Set default parameters
    if fill_missing_params is None:
        fill_missing_params = {
            'strategy': 'mixed',
            'genre_col': 'track_genre_top'
        }
    
    if feature_scaler_params is None:
        feature_scaler_params = {
            'method': 'standard',
            'exclude_cols': ['artist_name', 'track_genre_top', 'track_genre_n_unique'],
            'log_transform_cols': ['tempo_mean', 'tempo_max', 'duration_ms_mean']
        }
    
    # Create preprocessors
    missing_filler = FillMissingSpotify(**fill_missing_params)
    feature_scaler = FeatureScaler(**feature_scaler_params)
    
    # Create pipeline
    steps = [
        ('fill_missing', missing_filler),
        ('scale_features', feature_scaler)
    ]
    
    return PreprocessingPipeline(steps=steps, verbose=verbose)


def create_custom_pipeline(*preprocessors, names: Optional[List[str]] = None, verbose: bool = False) -> PreprocessingPipeline:
    """
    Create a custom preprocessing pipeline from a list of preprocessors.
    
    Parameters
    ----------
    *preprocessors : BasePreprocessor
        Variable number of preprocessor objects to chain together.
        
    names : List[str], optional
        Custom names for the pipeline steps. If None, uses class names.
        
    verbose : bool, default=False
        If True, print step information during execution.
        
    Returns
    -------
    PreprocessingPipeline
        Pipeline with the specified preprocessors.
        
    Examples
    --------
    >>> from .spotify_missing_filler import FillMissingSpotify
    >>> from .feature_scaler import FeatureScaler
    >>> 
    >>> filler = FillMissingSpotify(strategy='genre_mean')
    >>> scaler = FeatureScaler(method='minmax')
    >>> 
    >>> pipeline = create_custom_pipeline(filler, scaler, verbose=True)
    """
    if names is None:
        names = [type(preprocessor).__name__.lower() for preprocessor in preprocessors]
    
    if len(names) != len(preprocessors):
        raise ValueError(f"Number of names ({len(names)}) must match number of preprocessors ({len(preprocessors)})")
    
    steps = list(zip(names, preprocessors))
    return PreprocessingPipeline(steps=steps, verbose=verbose)