"""
Unit tests for the EDA module.

This module contains comprehensive tests for the EDA class in the
recommender_pipeline.exploration.eda module.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import warnings
warnings.filterwarnings('ignore')

from recommender_pipeline.exploration.eda import EDA


class TestEDA:
    """Test class for EDA functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = {
            'numeric_col1': np.random.normal(100, 15, 1000),
            'numeric_col2': np.random.exponential(2, 1000),
            'numeric_col3': np.random.uniform(0, 100, 1000),
            'categorical_col1': np.random.choice(['A', 'B', 'C', 'D'], 1000),
            'categorical_col2': np.random.choice(['X', 'Y', 'Z'], 1000),
            'target_col': np.random.normal(50, 10, 1000)
        }
        
        # Add some missing values
        data_df = pd.DataFrame(data)
        data_df.loc[np.random.choice(data_df.index, 50), 'numeric_col1'] = np.nan
        data_df.loc[np.random.choice(data_df.index, 30), 'categorical_col1'] = np.nan
        
        return data_df
    
    @pytest.fixture
    def sample_data_no_missing(self):
        """Create sample data without missing values."""
        np.random.seed(42)
        data = {
            'numeric_col1': np.random.normal(100, 15, 100),
            'numeric_col2': np.random.exponential(2, 100),
            'categorical_col1': np.random.choice(['A', 'B', 'C'], 100),
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_data_only_categorical(self):
        """Create sample data with only categorical columns."""
        np.random.seed(42)
        data = {
            'categorical_col1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical_col2': np.random.choice(['X', 'Y', 'Z'], 100),
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_data_only_numeric(self):
        """Create sample data with only numeric columns."""
        np.random.seed(42)
        data = {
            'numeric_col1': np.random.normal(100, 15, 100),
            'numeric_col2': np.random.exponential(2, 100),
            'numeric_col3': np.random.uniform(0, 100, 100),
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def empty_data(self):
        """Create empty DataFrame."""
        return pd.DataFrame()
    
    def test_eda_initialization(self, sample_data):
        """Test EDA class initialization."""
        eda = EDA(sample_data)
        assert eda.data is not None
        assert len(eda.data) == len(sample_data)
        assert eda.target_column is None
        assert eda.plots == []
        assert eda.summaries == {}
    
    def test_eda_initialization_with_target(self, sample_data):
        """Test EDA class initialization with target column."""
        eda = EDA(sample_data, target_column='target_col')
        assert eda.target_column == 'target_col'
    
    def test_eda_data_independence(self, sample_data):
        """Test that EDA doesn't modify original data."""
        original_data = sample_data.copy()
        eda = EDA(sample_data)
        
        # Modify EDA data
        eda.data['new_col'] = 1
        
        # Check original data is unchanged
        assert 'new_col' not in original_data.columns
        assert len(original_data.columns) == len(sample_data.columns)
    
    def test_basic_info(self, sample_data):
        """Test basic_info method."""
        eda = EDA(sample_data)
        info = eda.basic_info()
        
        # Check returned structure
        assert isinstance(info, dict)
        expected_keys = ['shape', 'columns', 'dtypes', 'memory_usage', 'missing_values', 'duplicate_rows']
        for key in expected_keys:
            assert key in info
        
        # Check values
        assert info['shape'] == sample_data.shape
        assert info['columns'] == sample_data.columns.tolist()
        assert isinstance(info['dtypes'], dict)
        assert isinstance(info['memory_usage'], (int, np.integer))
        assert isinstance(info['missing_values'], dict)
        assert isinstance(info['duplicate_rows'], (int, np.integer))
        
        # Check summaries stored
        assert 'basic_info' in eda.summaries
    
    @patch('builtins.print')
    def test_basic_info_pretty_print(self, mock_print, sample_data):
        """Test basic_info method with pretty_print=True."""
        eda = EDA(sample_data)
        info = eda.basic_info(pretty_print=True)
        
        # Check that print was called (pretty printing occurred)
        mock_print.assert_called()
        
        # Check info still returned correctly
        assert isinstance(info, dict)
        assert 'shape' in info
    
    def test_basic_info_empty_data(self, empty_data):
        """Test basic_info method with empty DataFrame."""
        eda = EDA(empty_data)
        info = eda.basic_info()
        
        assert info['shape'] == (0, 0)
        assert info['columns'] == []
        assert info['duplicate_rows'] == 0
    
    def test_statistical_summary(self, sample_data):
        """Test statistical_summary method."""
        eda = EDA(sample_data)
        summary = eda.statistical_summary()
        
        # Check structure
        assert isinstance(summary, dict)
        assert 'numerical_summary' in summary
        assert 'correlation_matrix' in summary
        
        # Check numerical summary
        numerical_cols = sample_data.select_dtypes(include=[np.number]).columns
        assert len(summary['numerical_summary']) == len(numerical_cols)
        
        # Check correlation matrix (should exist if more than 1 numerical column)
        if len(numerical_cols) > 1:
            assert len(summary['correlation_matrix']) > 0
        else:
            assert len(summary['correlation_matrix']) == 0
        
        # Check summaries stored
        assert 'statistical_summary' in eda.summaries
    
    def test_statistical_summary_no_numeric(self, sample_data_only_categorical):
        """Test statistical_summary with no numeric columns."""
        eda = EDA(sample_data_only_categorical)
        summary = eda.statistical_summary()
        
        assert summary['numerical_summary'] == {}
        assert summary['correlation_matrix'] == {}
        
        # Check summaries stored
        assert 'statistical_summary' in eda.summaries
    
    def test_missing_values_analysis_with_missing(self, sample_data):
        """Test missing_values_analysis method with missing values."""
        eda = EDA(sample_data)
        fig = eda.missing_values_analysis()
        
        # Should return a figure since there are missing values
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check plots stored
        assert len(eda.plots) == 1
        assert eda.plots[0][0] == 'missing_values'
        
        plt.close(fig)  # Clean up
    
    @patch('builtins.print')
    def test_missing_values_analysis_no_missing(self, mock_print, sample_data_no_missing):
        """Test missing_values_analysis method with no missing values."""
        eda = EDA(sample_data_no_missing)
        fig = eda.missing_values_analysis()
        
        # Should return None since no missing values
        assert fig is None
        
        # Check appropriate message was printed
        mock_print.assert_called_with("✅ No missing values found in the dataset!")
        
        # Check no plots stored
        assert len(eda.plots) == 0
    
    def test_distribution_analysis(self, sample_data_only_numeric):
        """Test distribution_analysis method."""
        eda = EDA(sample_data_only_numeric)
        fig = eda.distribution_analysis()
        
        # Should return a figure
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check plots stored
        assert len(eda.plots) == 1
        assert eda.plots[0][0] == 'distribution_analysis'
        
        plt.close(fig)  # Clean up
    
    @patch('builtins.print')
    def test_distribution_analysis_no_numeric(self, mock_print, sample_data_only_categorical):
        """Test distribution_analysis with no numeric columns."""
        eda = EDA(sample_data_only_categorical)
        fig = eda.distribution_analysis()
        
        # Should return None
        assert fig is None
        
        # Check appropriate message was printed
        mock_print.assert_called_with("📊 No numerical columns found for distribution analysis")
    
    def test_categorical_analysis(self, sample_data_only_categorical):
        """Test categorical_analysis method."""
        eda = EDA(sample_data_only_categorical)
        fig = eda.categorical_analysis()
        
        # Should return a figure
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check plots stored
        assert len(eda.plots) == 1
        assert eda.plots[0][0] == 'categorical_analysis'
        
        plt.close(fig)  # Clean up
    
    @patch('builtins.print')
    def test_categorical_analysis_no_categorical(self, mock_print, sample_data_only_numeric):
        """Test categorical_analysis with no categorical columns."""
        eda = EDA(sample_data_only_numeric)
        fig = eda.categorical_analysis()
        
        # Should return None
        assert fig is None
        
        # Check appropriate message was printed
        mock_print.assert_called_with("📊 No categorical columns found for analysis")
    
    def test_correlation_heatmap(self, sample_data_only_numeric):
        """Test correlation_heatmap method."""
        eda = EDA(sample_data_only_numeric)
        fig = eda.correlation_heatmap()
        
        # Should return a figure (has 3 numeric columns)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check plots stored
        assert len(eda.plots) == 1
        assert eda.plots[0][0] == 'correlation_heatmap'
        
        plt.close(fig)  # Clean up
    
    @patch('builtins.print')
    def test_correlation_heatmap_insufficient_columns(self, mock_print):
        """Test correlation_heatmap with insufficient columns."""
        data = pd.DataFrame({'col1': [1, 2, 3]})
        eda = EDA(data)
        fig = eda.correlation_heatmap()
        
        # Should return None
        assert fig is None
        
        # Check appropriate message was printed
        mock_print.assert_called_with("📊 Need at least 2 numerical columns for correlation analysis")
    
    def test_outlier_analysis(self, sample_data_only_numeric):
        """Test outlier_analysis method."""
        eda = EDA(sample_data_only_numeric)
        fig = eda.outlier_analysis()
        
        # Should return a figure
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check plots stored
        assert len(eda.plots) == 1
        assert eda.plots[0][0] == 'outlier_analysis'
        
        plt.close(fig)  # Clean up
    
    @patch('builtins.print')
    def test_outlier_analysis_no_numeric(self, mock_print, sample_data_only_categorical):
        """Test outlier_analysis with no numeric columns."""
        eda = EDA(sample_data_only_categorical)
        fig = eda.outlier_analysis()
        
        # Should return None
        assert fig is None
        
        # Check appropriate message was printed
        mock_print.assert_called_with("📊 No numerical columns found for outlier analysis")
    
    def test_bivariate_analysis(self, sample_data_only_numeric):
        """Test bivariate_analysis method."""
        eda = EDA(sample_data_only_numeric)
        figures = eda.bivariate_analysis()
        
        # Should return list of figures (3 columns = 3 pairs)
        assert isinstance(figures, list)
        assert len(figures) == 3  # C(3,2) = 3 pairs
        
        for fig in figures:
            assert isinstance(fig, plt.Figure)
            plt.close(fig)  # Clean up
    
    @patch('builtins.print')
    def test_bivariate_analysis_insufficient_columns(self, mock_print):
        """Test bivariate_analysis with insufficient columns."""
        data = pd.DataFrame({'col1': [1, 2, 3]})
        eda = EDA(data)
        figures = eda.bivariate_analysis()
        
        # Should return empty list
        assert figures == []
        
        # Check appropriate message was printed
        mock_print.assert_called_with("📊 Need at least 2 numerical columns for bivariate analysis")
    
    def test_target_analysis_with_target(self, sample_data):
        """Test target_analysis method with valid target."""
        eda = EDA(sample_data, target_column='target_col')
        figures = eda.target_analysis()
        
        # Should return list of figures
        assert isinstance(figures, list)
        assert len(figures) > 0
        
        for fig in figures:
            assert isinstance(fig, plt.Figure)
            plt.close(fig)  # Clean up
    
    def test_target_analysis_no_target(self, sample_data):
        """Test target_analysis method without target column."""
        eda = EDA(sample_data)  # No target column specified
        figures = eda.target_analysis()
        
        # Should return empty list
        assert figures == []
    
    def test_target_analysis_invalid_target(self, sample_data):
        """Test target_analysis method with invalid target column."""
        eda = EDA(sample_data, target_column='nonexistent_col')
        figures = eda.target_analysis()
        
        # Should return empty list
        assert figures == []
    
    @patch('builtins.print')
    def test_run_full_analysis(self, mock_print, sample_data):
        """Test run_full_analysis method."""
        eda = EDA(sample_data, target_column='target_col')
        result = eda.run_full_analysis()
        
        # Check return structure
        assert isinstance(result, dict)
        expected_keys = ['basic_info', 'statistical_summary', 'total_plots']
        for key in expected_keys:
            assert key in result
        
        # Check that analysis methods were called
        assert 'basic_info' in eda.summaries
        assert 'statistical_summary' in eda.summaries
        assert len(eda.plots) > 0
        assert result['total_plots'] == len(eda.plots)
        
        # Check print statements were made
        mock_print.assert_called()
        
        # Clean up plots
        for _, fig in eda.plots:
            if fig is not None:
                plt.close(fig)
    
    def test_plot_storage(self, sample_data_only_numeric):
        """Test that plots are correctly stored."""
        eda = EDA(sample_data_only_numeric)
        
        # Generate multiple plots
        eda.distribution_analysis()
        eda.correlation_heatmap()
        eda.outlier_analysis()
        
        # Check plots are stored with correct naming
        assert len(eda.plots) == 3
        plot_names = [plot[0] for plot in eda.plots]
        assert 'distribution_analysis' in plot_names
        assert 'correlation_heatmap' in plot_names
        assert 'outlier_analysis' in plot_names
        
        # Clean up
        for _, fig in eda.plots:
            plt.close(fig)
    
    def test_edge_case_single_column(self):
        """Test EDA with single column DataFrame."""
        data = pd.DataFrame({'single_col': [1, 2, 3, 4, 5]})
        eda = EDA(data)
        
        # Should handle single column gracefully
        basic_info = eda.basic_info()
        assert basic_info['shape'] == (5, 1)
        
        # Statistical summary should work
        summary = eda.statistical_summary()
        assert len(summary['numerical_summary']) == 1
        assert summary['correlation_matrix'] == {}  # No correlation with single column
        
        # Distribution analysis should work
        fig = eda.distribution_analysis()
        assert fig is not None
        plt.close(fig)
    
    def test_edge_case_single_row(self, sample_data):
        """Test EDA with single row DataFrame."""
        single_row_data = sample_data.iloc[[0]]
        eda = EDA(single_row_data)
        
        # Should handle single row gracefully
        basic_info = eda.basic_info()
        assert basic_info['shape'][0] == 1
        
        # Methods should not crash
        eda.statistical_summary()
        eda.missing_values_analysis()
        
        # Visual methods might return figures or None
        dist_fig = eda.distribution_analysis()
        if dist_fig:
            plt.close(dist_fig)
    
    def test_data_types_handling(self):
        """Test EDA with various data types."""
        data = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
            'bool_col': [True, False, True, False, True],
            'datetime_col': pd.date_range('2023-01-01', periods=5)
        })
        
        eda = EDA(data)
        basic_info = eda.basic_info()
        
        # Check that all data types are captured
        assert len(basic_info['dtypes']) == 5
        assert 'int_col' in basic_info['dtypes']
        assert 'float_col' in basic_info['dtypes']
        assert 'str_col' in basic_info['dtypes']
        assert 'bool_col' in basic_info['dtypes']
        assert 'datetime_col' in basic_info['dtypes']
    
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.subplots')
    def test_matplotlib_integration(self, mock_subplots, mock_tight_layout, sample_data_only_numeric):
        """Test matplotlib integration and plot creation."""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        eda = EDA(sample_data_only_numeric)
        
        # This should call matplotlib functions
        result = eda.distribution_analysis()
        
        # Verify matplotlib functions were called
        mock_subplots.assert_called()
        mock_tight_layout.assert_called()
    
    def teardown_method(self, method):
        """Clean up after each test method."""
        # Close any open matplotlib figures to prevent memory leaks
        plt.close('all')


class TestEDAIntegration:
    """Integration tests for EDA class."""
    
    @pytest.fixture
    def realistic_data(self):
        """Create more realistic sample data for integration testing."""
        np.random.seed(42)
        n_samples = 500
        
        data = {
            # Music-related numeric features (similar to Spotify data)
            'danceability': np.random.beta(2, 3, n_samples),
            'energy': np.random.beta(2, 2, n_samples),
            'loudness': np.random.normal(-8, 4, n_samples),
            'tempo': np.random.normal(120, 30, n_samples),
            'valence': np.random.beta(2, 2, n_samples),
            
            # Categorical features
            'genre': np.random.choice(['rock', 'pop', 'jazz', 'electronic', 'classical'], n_samples),
            'artist_popularity': np.random.choice(['low', 'medium', 'high'], n_samples),
            
            # Target variable
            'user_rating': np.random.normal(3.5, 1.2, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Add some realistic missing values
        missing_idx = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
        df.loc[missing_idx, 'artist_popularity'] = np.nan
        
        return df
    
    def test_complete_eda_workflow(self, realistic_data):
        """Test complete EDA workflow with realistic data."""
        eda = EDA(realistic_data, target_column='user_rating')
        
        # Run complete analysis
        results = eda.run_full_analysis()
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'basic_info' in results
        assert 'statistical_summary' in results
        assert 'total_plots' in results
        
        # Verify basic info
        basic_info = results['basic_info']
        assert basic_info['shape'] == realistic_data.shape
        assert len(basic_info['missing_values']) == len(realistic_data.columns)
        
        # Verify statistical summary
        stats = results['statistical_summary']
        assert 'numerical_summary' in stats
        assert 'correlation_matrix' in stats
        
        # Verify plots were generated
        assert results['total_plots'] > 0
        assert len(eda.plots) == results['total_plots']
        
        # Clean up
        for _, fig in eda.plots:
            if fig is not None:
                plt.close(fig)
    
    def test_performance_with_large_dataset(self):
        """Test EDA performance with larger dataset."""
        np.random.seed(42)
        large_data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 10000),
            'col2': np.random.exponential(1, 10000),
            'col3': np.random.choice(['A', 'B', 'C', 'D', 'E'], 10000),
        })
        
        eda = EDA(large_data)
        
        # Should complete without errors or excessive time
        import time
        start_time = time.time()
        
        basic_info = eda.basic_info()
        eda.statistical_summary()
        eda.distribution_analysis()
        
        end_time = time.time()
        
        # Should complete in reasonable time (less than 10 seconds)
        assert (end_time - start_time) < 10
        
        # Should produce correct results
        assert basic_info['shape'] == (10000, 3)
        
        # Clean up
        for _, fig in eda.plots:
            if fig is not None:
                plt.close(fig)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])