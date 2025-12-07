import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")


class EDA:
    """
    Exploratory Data Analysis class for comprehensive data exploration.
    Generates interactive visualizations and statistical summaries.
    """
    
    def __init__(self, data: pd.DataFrame, target_column: Optional[str] = None):
        """
        Initialize EDA class with data.
        
        Args:
            data: pandas DataFrame to analyze
            target_column: name of target column if doing supervised analysis
        """
        self.data = data.copy()
        self.target_column = target_column
        self.plots = []
        self.summaries = {}
        
    def basic_info(self, pretty_print: bool = False) -> Dict:
        """Get basic information about the dataset."""
        info = {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicate_rows': self.data.duplicated().sum()
        }
        self.summaries['basic_info'] = info
        
        if pretty_print:
            self._pretty_print_basic_info(info)
        
        return info
    
    def _pretty_print_basic_info(self, info: Dict) -> None:
        """Pretty print basic dataset information."""
        print("="*60)
        print("📊 DATASET OVERVIEW")
        print("="*60)
        
        # Dataset basics
        print(f"📐 Shape: {info['shape'][0]:,} rows × {info['shape'][1]:,} columns")
        print(f"💾 Memory Usage: {info['memory_usage']:,} bytes ({info['memory_usage']/1024/1024:.2f} MB)")
        print(f"🔍 Duplicate Rows: {info['duplicate_rows']:,}")
        
        # Missing values summary
        total_missing = sum(info['missing_values'].values())
        total_cells = info['shape'][0] * info['shape'][1]
        missing_percent = (total_missing / total_cells * 100) if total_cells > 0 else 0
        print(f"❓ Total Missing Values: {total_missing:,} ({missing_percent:.2f}%)")
        
        print("\n" + "="*60)
        print("📋 COLUMN DETAILS")
        print("="*60)
        
        # Column information
        print(f"{'Column Name':<25} {'Data Type':<15} {'Missing':<8} {'Missing %':<10}")
        print("-" * 60)
        
        for col in info['columns']:
            dtype = str(info['dtypes'][col])
            missing = info['missing_values'][col]
            missing_pct = (missing / info['shape'][0] * 100) if info['shape'][0] > 0 else 0
            
            # Truncate long column names
            col_display = col[:22] + "..." if len(col) > 25 else col
            dtype_display = dtype[:12] + "..." if len(dtype) > 15 else dtype
            
            print(f"{col_display:<25} {dtype_display:<15} {missing:<8} {missing_pct:<10.1f}")
        
        # Data type summary
        dtype_counts = {}
        for dtype in info['dtypes'].values():
            dtype_str = str(dtype)
            dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
        
        print("\n" + "="*60)
        print("🏷️  DATA TYPE DISTRIBUTION")
        print("="*60)
        for dtype, count in sorted(dtype_counts.items()):
            print(f"{dtype:<20}: {count:>3} columns")
        
        print("="*60)
    
    def statistical_summary(self) -> Dict:
        """Generate statistical summary for numerical columns."""
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        summary = {
            'numerical_summary': self.data[numerical_cols].describe().to_dict() if len(numerical_cols) > 0 else {},
            'correlation_matrix': self.data[numerical_cols].corr().to_dict() if len(numerical_cols) > 1 else {}
        }
        self.summaries['statistical_summary'] = summary
        return summary
    
    def missing_values_analysis(self) -> Optional[plt.Figure]:
        """Analyze and visualize missing values."""
        missing_data = self.data.isnull().sum()
        
        # Filter to only columns with missing values
        missing_data_filtered = missing_data[missing_data > 0]
        
        # If no missing values, return None and don't create plot
        if missing_data_filtered.empty:
            print("✅ No missing values found in the dataset!")
            return None
        
        missing_percent = (missing_data_filtered / len(self.data)) * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(missing_data_filtered)), missing_percent.values, color='lightcoral', alpha=0.7)
        
        # Add percentage labels on bars
        for i, (bar, pct) in enumerate(zip(bars, missing_percent.values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{pct:.1f}%', ha='center', va='bottom')
        
        ax.set_title(f'Missing Values Analysis - {len(missing_data_filtered)} columns with missing values', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Columns', fontsize=12)
        ax.set_ylabel('Missing Percentage', fontsize=12)
        ax.set_xticks(range(len(missing_data_filtered)))
        ax.set_xticklabels(missing_data_filtered.index, rotation=45, ha='right')
        
        plt.tight_layout()
        self.plots.append(('missing_values', fig))
        print(f"📊 Found missing values in {len(missing_data_filtered)} columns")
        return fig
    
    def distribution_analysis(self) -> Optional[plt.Figure]:
        """Analyze distributions of numerical columns in a single subplot figure."""
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            print("📊 No numerical columns found for distribution analysis")
            return None
        
        # Calculate grid dimensions
        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Handle single subplot case
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            ax = axes[i]
            ax.hist(self.data[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'Distribution of {col}', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(f'Distribution Analysis - {len(numerical_cols)} Numerical Columns', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        self.plots.append(('distribution_analysis', fig))
        print(f"📊 Created distribution plots for {len(numerical_cols)} columns")
        return fig
    
    def categorical_analysis(self) -> Optional[plt.Figure]:
        """Analyze categorical columns in a single subplot figure."""
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            print("📊 No categorical columns found for analysis")
            return None
        
        # Calculate grid dimensions
        n_cols = min(2, len(categorical_cols))  # 2 columns for better readability
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows))
        
        # Handle single subplot case
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(categorical_cols):
            ax = axes[i]
            value_counts = self.data[col].value_counts().head(15)  # Top 15 for better visibility
            
            bars = ax.bar(range(len(value_counts)), value_counts.values, 
                         color='lightgreen', alpha=0.7, edgecolor='black')
            
            ax.set_title(f'Top Categories in {col}', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, value_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01, 
                       str(val), ha='center', va='bottom', fontsize=8)
        
        # Hide extra subplots
        for i in range(len(categorical_cols), len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(f'Categorical Analysis - {len(categorical_cols)} Columns', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        self.plots.append(('categorical_analysis', fig))
        print(f"📊 Created categorical plots for {len(categorical_cols)} columns")
        return fig
    
    def correlation_heatmap(self) -> Optional[plt.Figure]:
        """Create correlation heatmap for numerical variables."""
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            print("📊 Need at least 2 numerical columns for correlation analysis")
            return None
        
        corr_matrix = self.data[numerical_cols].corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'shrink': 0.8},
                   ax=ax)
        
        ax.set_title('Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        self.plots.append(('correlation_heatmap', fig))
        print(f"📊 Created correlation heatmap for {len(numerical_cols)} columns")
        return fig
    
    def outlier_analysis(self) -> Optional[plt.Figure]:
        """Detect and visualize outliers using box plots in a single subplot figure."""
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            print("📊 No numerical columns found for outlier analysis")
            return None
        
        # Calculate grid dimensions
        n_cols = min(4, len(numerical_cols))  # 4 columns for box plots
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        
        # Handle single subplot case
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            ax = axes[i]
            
            # Create box plot
            bp = ax.boxplot(self.data[col].dropna(), patch_artist=True, 
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2),
                          flierprops=dict(marker='o', markerfacecolor='red', 
                                        markersize=5, alpha=0.5))
            
            ax.set_title(f'Outlier Analysis: {col}', fontweight='bold')
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)
            
            # Add some statistics as text
            q1 = self.data[col].quantile(0.25)
            q3 = self.data[col].quantile(0.75)
            iqr = q3 - q1
            outliers = self.data[(self.data[col] < q1 - 1.5*iqr) | 
                               (self.data[col] > q3 + 1.5*iqr)][col]
            
            ax.text(0.02, 0.98, f'Outliers: {len(outliers)}', 
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide extra subplots
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(f'Outlier Analysis - {len(numerical_cols)} Numerical Columns', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        self.plots.append(('outlier_analysis', fig))
        print(f"📊 Created outlier analysis for {len(numerical_cols)} columns")
        return fig
    
    def bivariate_analysis(self) -> List[plt.Figure]:
        """Perform bivariate analysis between numerical variables."""
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        figures = []
        
        if len(numerical_cols) < 2:
            print("📊 Need at least 2 numerical columns for bivariate analysis")
            return figures
        
        # Limit to avoid too many plots - take most important pairs
        max_pairs = 15  # Limit number of scatter plots
        pair_count = 0
        
        for i in range(len(numerical_cols)):
            for j in range(i+1, len(numerical_cols)):
                if pair_count >= max_pairs:
                    break
                    
                col1, col2 = numerical_cols[i], numerical_cols[j]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Create scatter plot with some transparency
                scatter = ax.scatter(self.data[col1], self.data[col2], 
                                   alpha=0.6, s=20, color='steelblue', edgecolors='none')
                
                ax.set_title(f'{col1} vs {col2}', fontsize=14, fontweight='bold')
                ax.set_xlabel(col1, fontsize=12)
                ax.set_ylabel(col2, fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Add correlation coefficient
                corr_coef = self.data[[col1, col2]].corr().iloc[0, 1]
                ax.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                       transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                figures.append(fig)
                self.plots.append((f'bivariate_{col1}_{col2}', fig))
                pair_count += 1
            
            if pair_count >= max_pairs:
                break
        
        print(f"📊 Created {len(figures)} bivariate scatter plots")
        return figures
    
    def target_analysis(self) -> List[plt.Figure]:
        """Analyze relationship between features and target variable."""
        if not self.target_column or self.target_column not in self.data.columns:
            return []
        
        figures = []
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        
        # Numerical vs target
        for col in numerical_cols:
            if col != self.target_column:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                ax.scatter(self.data[col], self.data[self.target_column], 
                          alpha=0.6, s=20, color='darkgreen', edgecolors='none')
                
                ax.set_title(f'{col} vs {self.target_column}', fontsize=14, fontweight='bold')
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel(self.target_column, fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Add correlation coefficient
                corr_coef = self.data[[col, self.target_column]].corr().iloc[0, 1]
                ax.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                       transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                figures.append(fig)
                self.plots.append((f'target_vs_{col}', fig))
        
        # Categorical vs target
        for col in categorical_cols:
            if col != self.target_column:
                grouped = self.data.groupby(col)[self.target_column].mean().head(20)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                bars = ax.bar(range(len(grouped)), grouped.values, 
                            color='orange', alpha=0.7, edgecolor='black')
                
                ax.set_title(f'Average {self.target_column} by {col}', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel(f'Average {self.target_column}', fontsize=12)
                ax.set_xticks(range(len(grouped)))
                ax.set_xticklabels(grouped.index, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, val in zip(bars, grouped.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01, 
                           f'{val:.2f}', ha='center', va='bottom', fontsize=8)
                
                plt.tight_layout()
                figures.append(fig)
                self.plots.append((f'target_by_{col}', fig))
        
        return figures
    
    def run_full_analysis(self) -> Dict:
        """Run complete EDA analysis."""
        print("Running comprehensive EDA analysis...")
        
        # Basic information
        basic_info = self.basic_info()
        print(f"Dataset shape: {basic_info['shape']}")
        
        # Statistical summary
        stats = self.statistical_summary()
        
        # Generate all visualizations
        print("Generating visualizations...")
        missing_fig = self.missing_values_analysis()
        dist_fig = self.distribution_analysis()
        cat_fig = self.categorical_analysis()
        corr_fig = self.correlation_heatmap()
        outlier_fig = self.outlier_analysis()
        bivariate_figs = self.bivariate_analysis()
        
        if self.target_column:
            self.target_analysis()
        
        print(f"Generated {len(self.plots)} visualizations")
        
        return {
            'basic_info': basic_info,
            'statistical_summary': stats,
            'total_plots': len(self.plots)
        }