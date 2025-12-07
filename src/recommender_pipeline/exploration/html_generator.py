"""
HTML Report Generator for EDA analysis.
"""

import matplotlib.pyplot as plt
import base64
import io
import os
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .eda import EDA


class HTMLGenerator:
    """
    HTML report generator for EDA analysis results.
    Generates comprehensive interactive HTML reports from EDA class instances.
    """
    
    def __init__(self):
        """Initialize HTML generator."""
        self.default_styles = """
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #2c3e50; text-align: center; }
            h2 { color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
            .summary { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
            .plot-container { margin-bottom: 30px; }
            .info-table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
            .info-table th, .info-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            .info-table th { background-color: #f2f2f2; }
            .toc { background-color: #f1f2f6; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
            .toc ul { list-style-type: none; padding-left: 20px; }
            .toc a { text-decoration: none; color: #2c3e50; }
            .toc a:hover { color: #3498db; }
        """
    
    def _generate_table_of_contents(self, eda_instance: 'EDA') -> str:
        """Generate table of contents for the report."""
        toc_html = """
        <div class="toc">
            <h2>Table of Contents</h2>
            <ul>
                <li><a href="#overview">Dataset Overview</a></li>
        """
        
        # Add sections based on available plots
        plot_sections = set()
        for plot_name, _ in eda_instance.plots:
            section = plot_name.split('_')[0]
            plot_sections.add(section)
        
        section_names = {
            'missing': 'Missing Values Analysis',
            'distribution': 'Distribution Analysis',
            'categorical': 'Categorical Analysis', 
            'correlation': 'Correlation Analysis',
            'outliers': 'Outlier Analysis',
            'bivariate': 'Bivariate Analysis',
            'target': 'Target Variable Analysis'
        }
        
        for section in sorted(plot_sections):
            if section in section_names:
                toc_html += f'<li><a href="#{section}">{section_names[section]}</a></li>'
        
        toc_html += """
            </ul>
        </div>
        """
        return toc_html
    
    def _generate_overview_section(self, eda_instance: 'EDA') -> str:
        """Generate dataset overview section."""
        if 'basic_info' not in eda_instance.summaries:
            return ""
        
        info = eda_instance.summaries['basic_info']
        overview_html = f"""
        <div id="overview" class="summary">
            <h2>Dataset Overview</h2>
            <table class="info-table">
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Shape</td><td>{info['shape']}</td></tr>
                <tr><td>Number of Columns</td><td>{len(info['columns'])}</td></tr>
                <tr><td>Total Missing Values</td><td>{sum(info['missing_values'].values())}</td></tr>
                <tr><td>Duplicate Rows</td><td>{info['duplicate_rows']}</td></tr>
                <tr><td>Memory Usage (bytes)</td><td>{info['memory_usage']:,}</td></tr>
            </table>
            
            <h3>Column Information</h3>
            <table class="info-table">
                <tr><th>Column</th><th>Data Type</th><th>Missing Values</th><th>Missing %</th></tr>
        """
        
        total_rows = info['shape'][0]
        for col in info['columns']:
            dtype = str(info['dtypes'][col])
            missing = info['missing_values'][col]
            missing_pct = (missing / total_rows * 100) if total_rows > 0 else 0
            
            overview_html += f"""
                <tr>
                    <td>{col}</td>
                    <td>{dtype}</td>
                    <td>{missing}</td>
                    <td>{missing_pct:.1f}%</td>
                </tr>
            """
        
        overview_html += """
            </table>
        </div>
        """
        return overview_html
    
    def _generate_statistical_summary(self, eda_instance: 'EDA') -> str:
        """Generate statistical summary section."""
        if 'statistical_summary' not in eda_instance.summaries:
            return ""
        
        stats = eda_instance.summaries['statistical_summary']
        if not stats['numerical_summary']:
            return ""
        
        summary_html = """
        <div class="summary">
            <h2>Statistical Summary</h2>
            <table class="info-table">
                <tr><th>Statistic</th>
        """
        
        # Add column headers
        columns = list(stats['numerical_summary'].keys())
        for col in columns:
            summary_html += f"<th>{col}</th>"
        summary_html += "</tr>"
        
        # Add statistical rows
        stat_names = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        for stat in stat_names:
            summary_html += f"<tr><td><strong>{stat}</strong></td>"
            for col in columns:
                value = stats['numerical_summary'][col].get(stat, 'N/A')
                if isinstance(value, (int, float)):
                    summary_html += f"<td>{value:.2f}</td>"
                else:
                    summary_html += f"<td>{value}</td>"
            summary_html += "</tr>"
        
        summary_html += """
            </table>
        </div>
        """
        return summary_html
    
    def _group_plots_by_section(self, eda_instance: 'EDA') -> dict:
        """Group plots by their section type."""
        sections = {}
        for plot_name, fig in eda_instance.plots:
            section = plot_name.split('_')[0]
            if section not in sections:
                sections[section] = []
            sections[section].append((plot_name, fig))
        return sections
    
    def _matplotlib_to_base64(self, fig):
        """Convert matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close(fig)  # Close figure to free memory
        return img_base64
    
    def _generate_plot_sections(self, eda_instance: 'EDA') -> str:
        """Generate all plot sections."""
        sections = self._group_plots_by_section(eda_instance)
        
        section_titles = {
            'missing': 'Missing Values Analysis',
            'distribution': 'Distribution Analysis',
            'categorical': 'Categorical Analysis',
            'correlation': 'Correlation Analysis',
            'outliers': 'Outlier Analysis',
            'bivariate': 'Bivariate Analysis',
            'target': 'Target Variable Analysis'
        }
        
        plots_html = ""
        for section, plots in sections.items():
            section_title = section_titles.get(section, section.replace('_', ' ').title())
            plots_html += f"""
            <div id="{section}">
                <h2>{section_title}</h2>
            """
            
            for plot_name, fig in plots:
                plot_title = plot_name.replace('_', ' ').title()
                
                # Convert matplotlib figure to base64 image
                try:
                    img_base64 = self._matplotlib_to_base64(fig)
                    plots_html += f"""
                    <div class="plot-container">
                        <h3>{plot_title}</h3>
                        <img src="data:image/png;base64,{img_base64}" alt="{plot_title}" style="max-width: 100%; height: auto;">
                    </div>
                    """
                except Exception as e:
                    print(f"Warning: Could not convert plot {plot_name} to image: {e}")
                    plots_html += f"""
                    <div class="plot-container">
                        <h3>{plot_title}</h3>
                        <p style="color: red;">Error generating plot: {plot_name}</p>
                    </div>
                    """
            
            plots_html += "</div>"
        
        return plots_html
    
    def generate_report(self, eda_instance: 'EDA', filename: str = 'eda_report.html', 
                       title: str = 'Exploratory Data Analysis Report',
                       custom_styles: str = None) -> str:
        """
        Generate comprehensive HTML report from EDA instance.
        
        Args:
            eda_instance: EDA class instance with analysis results
            filename: name of the output HTML file
            title: title for the report
            custom_styles: optional custom CSS styles
            
        Returns:
            path to generated HTML file
        """
        print(f"Generating HTML report: {filename}")
        
        if not eda_instance.plots:
            print("Warning: No plots found in EDA instance. Run analysis first.")
            return ""
        
        # Use custom styles or default
        styles = custom_styles if custom_styles else self.default_styles
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                {styles}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p><strong>Generated on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Dataset shape:</strong> {eda_instance.data.shape}</p>
        """
        
        # Add table of contents
        html_content += self._generate_table_of_contents(eda_instance)
        
        # Add overview section
        html_content += self._generate_overview_section(eda_instance)
        
        # Add statistical summary
        html_content += self._generate_statistical_summary(eda_instance)
        
        # Add all plot sections
        html_content += self._generate_plot_sections(eda_instance)
        
        html_content += """
        </body>
        </html>
        """
        
        # Write to file
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created output directory: {output_dir}")
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"HTML report generated successfully: {filename}")
            print(f"Report contains {len(eda_instance.plots)} static visualizations")
            return filename
            
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            return ""