# Standard library imports
import os
import sqlite3
import threading
import time
import signal
import sys
from functools import wraps
import re
from datetime import datetime, timedelta

# Third-party imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import t
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from fuzzywuzzy import fuzz
import itertools
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Local imports
from src.api_model import EragAPI
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight
from src.print_pdf import PDFReportGenerator
from src.helper_da import get_technique_info

class TimeoutException(Exception):
    pass

class AdvancedExploratoryDataAnalysisB6:
    def __init__(self, worker_erag_api, supervisor_erag_api, db_path):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.db_path = db_path
        self.technique_counter = 0
        self.total_techniques = 15
        self.table_name = None
        self.output_folder = None
        self.text_output = ""
        self.pdf_content = []
        self.findings = []
        self.llm_name = f"Worker: {self.worker_erag_api.model}, Supervisor: {self.supervisor_erag_api.model}"
        self.toc_entries = []
        self.image_paths = []
        self.max_pixels = 400000
        self.timeout_seconds = 10
        self.image_data = []
        self.pdf_generator = None
        self.settings = settings
        self.database_description = ""
        self.paused = False
        self.entity_names_mapping = {}
        self.setup_signal_handler()

    def setup_signal_handler(self):
        """Set up signal handler for Ctrl+C"""
        signal.signal(signal.SIGINT, self.handle_signal)

    def handle_signal(self, sig, frame):
        """Handle Ctrl+C by pausing execution"""
        if not self.paused:
            self.paused = True
            print(warning("\nScript paused. Press Enter to continue or Ctrl+C again to exit..."))
            try:
                user_input = input()
                self.paused = False
                print(info("Resuming execution..."))
            except KeyboardInterrupt:
                print(error("\nExiting script..."))
                sys.exit(0)
        else:
            print(error("\nExiting script..."))
            sys.exit(0)

    def check_if_paused(self):
        """Check if execution is paused and wait for Enter if needed"""
        while self.paused:
            time.sleep(0.1)  # Small sleep to prevent CPU hogging

    def calculate_figure_size(self, aspect_ratio=16/9):
        max_width = int(np.sqrt(self.max_pixels * aspect_ratio))
        max_height = int(max_width / aspect_ratio)
        return (max_width / 100, max_height / 100)

    def timeout(timeout_duration):
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                result = [TimeoutException("Function call timed out")]

                def target():
                    try:
                        result[0] = func(self, *args, **kwargs)
                    except Exception as e:
                        result[0] = e

                thread = threading.Thread(target=target)
                thread.start()
                thread.join(timeout_duration)

                if thread.is_alive():
                    print(f"Warning: {func.__name__} timed out after {timeout_duration} seconds. Skipping this graphic.")
                    return None
                else:
                    if isinstance(result[0], Exception):
                        raise result[0]
                    return result[0]
            return wrapper
        return decorator

    def prompt_for_database_description(self):
        """Ask the user for a description of the database"""
        print(info("Please provide a description of the database. This will help the AI models provide better insights."))
        print(info("Describe the purpose, main tables, key data points, and any other relevant information:"))
        self.database_description = input("> ")
        print(success(f"Database description received: {self.database_description}"))

    def identify_entity_names(self, df):
        """
        Identify actual names of entities in the dataset instead of using generic names
        """
        entity_names = {}
        
        # Try to identify teams, customers, products, clusters, etc.
        for col in df.columns:
            if col.lower() in ['team', 'team_name', 'group', 'department']:
                entity_names['teams'] = df[col].unique().tolist()
            elif col.lower() in ['customer', 'client', 'customer_name', 'client_name']:
                entity_names['customers'] = df[col].unique().tolist()
            elif col.lower() in ['product', 'item', 'product_name', 'service']:
                entity_names['products'] = df[col].unique().tolist()
            elif col.lower() in ['cluster', 'segment', 'category']:
                entity_names['clusters'] = df[col].unique().tolist()
            elif col.lower() in ['region', 'location', 'area', 'country', 'city']:
                entity_names['regions'] = df[col].unique().tolist()
            elif col.lower() in ['case_id', 'process', 'activity']:
                entity_names['processes'] = df[col].unique().tolist()
        
        # For categorical columns with few unique values, they might be important entities
        for col in df.select_dtypes(include=['object', 'category']).columns:
            unique_values = df[col].unique()
            if 2 <= len(unique_values) <= 20 and col not in entity_names:
                entity_names[col.lower()] = unique_values.tolist()
        
        return entity_names

    def format_entity_description(self):
        """Format the entity names for inclusion in prompts"""
        if not self.entity_names_mapping:
            return "No specific entities identified."
        
        description = []
        for entity_type, names in self.entity_names_mapping.items():
            if names:
                description.append(f"{entity_type.capitalize()}: {', '.join(str(name) for name in names)}")
        
        return "\n".join(description)

    @timeout(10)
    def generate_plot(self, plot_function, *args, **kwargs):
        return plot_function(*args, **kwargs)

    def get_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]

    def run(self):
        print(info(f"Starting Advanced Exploratory Data Analysis (Batch 6) on {self.db_path}"))
        
        # Ask for database description before starting analysis
        self.prompt_for_database_description()
        
        tables = self.get_tables()
        for table in tables:
            self.analyze_table(table)
        
        self.save_text_output()
        self.generate_pdf_report()
        print(success(f"Advanced Exploratory Data Analysis (Batch 6) completed. Results saved in {self.output_folder}"))

    def analyze_table(self, table_name):
        self.table_name = table_name
        self.output_folder = os.path.join(settings.output_folder, f"axda_b6_{self.table_name}")
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, self.table_name)
        
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        
        # Load the entire dataset
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            print(info(f"Loaded full dataset with {len(df)} rows and {len(df.columns)} columns"))
        
        # Identify entity names in the dataset
        self.entity_names_mapping = self.identify_entity_names(df)
        entity_description = self.format_entity_description()
        
        print(info(f"Identified entities in the data: {entity_description}"))

        analysis_methods = [
            self.trend_analysis,
            self.variance_analysis,
            self.regression_analysis,
            self.stratification_analysis,
            self.gap_analysis,
            self.duplicate_detection,
            self.process_mining,
            self.data_validation_techniques,
            self.risk_scoring_models,
            self.fuzzy_matching,
            self.continuous_auditing_techniques,
            self.sensitivity_analysis,
            self.scenario_analysis,
            self.monte_carlo_simulation,
            self.kpi_analysis
        ]

        for method in analysis_methods:
            # Check if script is paused
            self.check_if_paused()
                
            try:
                method(df, table_name)
            except Exception as e:
                error_message = f"An error occurred during {method.__name__}: {str(e)}"
                print(error(error_message))
                self.text_output += f"\n{error_message}\n"
                self.pdf_content.append((method.__name__, [], error_message))
                
                # Write error to method-specific output file
                method_name = method.__name__
                with open(os.path.join(self.output_folder, f"{method_name}_results.txt"), "w", encoding='utf-8') as f:
                    f.write(error_message)
            finally:
                self.technique_counter += 1

    def trend_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Trend Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        # Try to convert string columns to datetime if no datetime columns are found
        if len(date_cols) == 0:
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols = [col]
                    print(info(f"Converted column {col} to datetime for trend analysis"))
                    break
                except:
                    continue
        
        if len(date_cols) == 0 or len(numeric_cols) == 0:
            print(warning("No suitable columns for trend analysis."))
            return
        
        date_col = date_cols[0]
        df = df.sort_values(by=date_col)
        
        def plot_trend():
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                ax.plot(df[date_col], df[col], label=col)
            ax.set_title('Trend Analysis')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            return fig, ax

        result = self.generate_plot(plot_trend)
        if result is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_trend_analysis.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(img_path)
            
            # Calculate trend statistics
            trend_stats = {}
            for col in numeric_cols:
                if df[col].iloc[0] != 0:  # Avoid division by zero
                    percent_change = ((df[col].iloc[-1] - df[col].iloc[0]) / df[col].iloc[0]) * 100
                else:
                    percent_change = np.nan
                    
                trend_stats[col] = {
                    'start': float(df[col].iloc[0]),
                    'end': float(df[col].iloc[-1]),
                    'change': float(df[col].iloc[-1] - df[col].iloc[0]),
                    'percent_change': float(percent_change) if not np.isnan(percent_change) else "N/A"
                }
            
            results = {
                'image_paths': image_paths,
                'trend_stats': trend_stats,
                'time_period': {
                    'start': df[date_col].min().strftime('%Y-%m-%d') if hasattr(df[date_col].min(), 'strftime') else str(df[date_col].min()),
                    'end': df[date_col].max().strftime('%Y-%m-%d') if hasattr(df[date_col].max(), 'strftime') else str(df[date_col].max()),
                    'total_periods': len(df)
                }
            }
            
            self.interpret_results("Trend Analysis", results, table_name)
        else:
            print("Skipping Trend Analysis plot due to timeout.")

    def variance_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Variance Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print(warning("No numeric columns for variance analysis."))
            return
        
        variance_stats = df[numeric_cols].var()
        
        # Calculate additional statistics for context
        std_stats = df[numeric_cols].std()
        mean_stats = df[numeric_cols].mean()
        cv_stats = std_stats / mean_stats  # Coefficient of variation
        
        def plot_variance():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
            
            # Variance bar chart
            variance_stats.plot(kind='bar', ax=ax1)
            ax1.set_title('Variance by Column')
            ax1.set_xlabel('Columns')
            ax1.set_ylabel('Variance')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Coefficient of variation bar chart
            cv_stats.plot(kind='bar', ax=ax2)
            ax2.set_title('Coefficient of Variation by Column')
            ax2.set_xlabel('Columns')
            ax2.set_ylabel('CV (std/mean)')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            return fig, (ax1, ax2)

        result = self.generate_plot(plot_variance)
        if result is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_variance_analysis.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(img_path)
            
            results = {
                'image_paths': image_paths,
                'variance_stats': variance_stats.to_dict(),
                'std_stats': std_stats.to_dict(),
                'mean_stats': mean_stats.to_dict(),
                'cv_stats': cv_stats.to_dict()
            }
            
            self.interpret_results("Variance Analysis", results, table_name)
        else:
            print("Skipping Variance Analysis plot due to timeout.")

    def regression_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Regression Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print(warning("Not enough numeric columns for regression analysis."))
            return
        
        # Perform simple linear regression for each pair of numeric columns
        regression_results = {}
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                X = df[col1].values.reshape(-1, 1)
                y = df[col2].values
                
                # Skip if either column has all same values (causing division by zero)
                if np.std(X) == 0 or np.std(y) == 0:
                    continue
                    
                model = LinearRegression()
                model.fit(X, y)
                r_squared = model.score(X, y)
                
                regression_results[f"{col1} vs {col2}"] = {
                    'r_squared': float(r_squared),
                    'coefficient': float(model.coef_[0]),
                    'intercept': float(model.intercept_),
                    'equation': f"{col2} = {model.coef_[0]:.4f} * {col1} + {model.intercept_:.4f}",
                    'correlation': float(np.corrcoef(df[col1], df[col2])[0, 1]),
                    'p_value': float(stats.pearsonr(df[col1].fillna(0), df[col2].fillna(0))[1])
                }
                
                def plot_regression():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.scatter(X, y, alpha=0.5)
                    ax.plot(X, model.predict(X), color='red', linewidth=2)
                    ax.set_title(f'Regression: {col1} vs {col2} (R² = {r_squared:.3f})')
                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2)
                    # Add the regression equation to the plot
                    equation_text = f"y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}"
                    ax.annotate(equation_text, xy=(0.05, 0.95), xycoords='axes fraction',
                                ha='left', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_regression)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_regression_{col1}_{col2}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)
                else:
                    print(f"Skipping Regression plot for {col1} vs {col2} due to timeout.")
        
        # Get top correlations
        correlations = []
        for pair, results in regression_results.items():
            correlations.append((pair, abs(results['correlation'])))
        
        top_correlations = sorted(correlations, key=lambda x: x[1], reverse=True)[:5]
        
        results = {
            'image_paths': image_paths,
            'regression_results': regression_results,
            'top_correlations': {pair: abs_corr for pair, abs_corr in top_correlations}
        }
        
        self.interpret_results("Regression Analysis", results, table_name)

    def stratification_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Stratification Analysis"))
        image_paths = []
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            print(warning("No suitable columns for stratification analysis."))
            return
        
        # Get actual entity names for categories
        entity_categories = {}
        for cat_col in categorical_cols:
            if cat_col in self.entity_names_mapping:
                entity_categories[cat_col] = self.entity_names_mapping[cat_col]
            else:
                unique_values = df[cat_col].unique()
                if len(unique_values) <= 20:  # Limit to categories with reasonable number of values
                    entity_categories[cat_col] = unique_values.tolist()
        
        stratification_results = {}
        for cat_col in categorical_cols[:2]:  # Limit to first 2 categorical columns
            for num_col in numeric_cols[:2]:  # Limit to first 2 numeric columns
                grouped = df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std', 'count'])
                
                # Calculate additional statistics
                total_mean = df[num_col].mean()
                grouped['variance_from_mean'] = ((grouped['mean'] - total_mean) / total_mean) * 100
                
                stratification_results[f"{cat_col} - {num_col}"] = {
                    'stats': grouped.to_dict(),
                    'total_mean': float(total_mean),
                    'total_std': float(df[num_col].std()),
                    'category_count': len(grouped),
                    'max_difference': float(grouped['mean'].max() - grouped['mean'].min()),
                    'max_variance_pct': float(grouped['variance_from_mean'].abs().max())
                }
                
                def plot_stratification():
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                    
                    # Boxplot
                    df.boxplot(column=num_col, by=cat_col, ax=ax1)
                    ax1.set_title(f'Distribution of {num_col} by {cat_col}')
                    ax1.set_xlabel(cat_col)
                    ax1.set_ylabel(num_col)
                    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
                    
                    # Bar chart of means
                    means = grouped['mean'].sort_values(ascending=False)
                    means.plot(kind='bar', ax=ax2)
                    ax2.set_title(f'Mean {num_col} by {cat_col}')
                    ax2.set_xlabel(cat_col)
                    ax2.set_ylabel(f'Mean {num_col}')
                    ax2.axhline(y=total_mean, color='r', linestyle='--', label=f'Overall Mean ({total_mean:.2f})')
                    ax2.legend()
                    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
                    
                    plt.tight_layout()
                    return fig, (ax1, ax2)

                result = self.generate_plot(plot_stratification)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_stratification_{cat_col}_{num_col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)
                else:
                    print(f"Skipping Stratification plot for {cat_col} - {num_col} due to timeout.")
        
        results = {
            'image_paths': image_paths,
            'stratification_results': stratification_results,
            'entity_categories': entity_categories
        }
        
        self.interpret_results("Stratification Analysis", results, table_name)

    def gap_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Gap Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print(warning("No numeric columns for gap analysis."))
            return
        
        gap_results = {}
        for col in numeric_cols:
            # Skip if column has all same values
            if df[col].std() == 0:
                continue
                
            current_value = df[col].mean()
            target_value = df[col].quantile(0.9)  # Using 90th percentile as target
            gap = target_value - current_value
            
            if target_value != 0:  # Avoid division by zero
                gap_percentage = (gap / target_value) * 100
            else:
                gap_percentage = np.nan
            
            gap_results[col] = {
                'current_value': float(current_value),
                'target_value': float(target_value),
                'gap': float(gap),
                'gap_percentage': float(gap_percentage) if not np.isnan(gap_percentage) else "N/A",
                'min_value': float(df[col].min()),
                'max_value': float(df[col].max()),
                'median_value': float(df[col].median())
            }
            
            def plot_gap():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                
                # Bar chart comparing current vs target
                ax1.bar(['Current', 'Target'], [current_value, target_value])
                ax1.set_title(f'Gap Analysis: {col}')
                ax1.set_ylabel('Value')
                for i, v in enumerate([current_value, target_value]):
                    ax1.text(i, v, f"{v:.2f}", ha='center', va='bottom')
                
                # Histogram showing distribution with current and target markers
                ax2.hist(df[col], bins=20, alpha=0.7)
                ax2.axvline(x=current_value, color='r', linestyle='--', label=f'Current: {current_value:.2f}')
                ax2.axvline(x=target_value, color='g', linestyle='--', label=f'Target: {target_value:.2f}')
                ax2.set_title(f'Distribution of {col}')
                ax2.set_xlabel(col)
                ax2.set_ylabel('Frequency')
                ax2.legend()
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_gap)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_gap_analysis_{col}.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Gap Analysis plot for {col} due to timeout.")
        
        # Find largest gaps (both absolute and percentage)
        if gap_results:
            largest_absolute_gap = max(gap_results.items(), key=lambda x: abs(x[1]['gap']))
            largest_pct_gaps = [(col, info) for col, info in gap_results.items() 
                              if isinstance(info['gap_percentage'], (int, float))]
            
            if largest_pct_gaps:
                largest_percentage_gap = max(largest_pct_gaps, key=lambda x: abs(x[1]['gap_percentage']))
            else:
                largest_percentage_gap = (None, None)
            
            summary = {
                'total_metrics_analyzed': len(gap_results),
                'largest_absolute_gap': {
                    'metric': largest_absolute_gap[0],
                    'gap': largest_absolute_gap[1]['gap']
                },
                'largest_percentage_gap': {
                    'metric': largest_percentage_gap[0] if largest_percentage_gap[0] else "N/A",
                    'gap_percentage': largest_percentage_gap[1]['gap_percentage'] if largest_percentage_gap[1] else "N/A"
                }
            }
        else:
            summary = {
                'total_metrics_analyzed': 0,
                'largest_absolute_gap': {'metric': "N/A", 'gap': "N/A"},
                'largest_percentage_gap': {'metric': "N/A", 'gap_percentage': "N/A"}
            }
        
        results = {
            'image_paths': image_paths,
            'gap_results': gap_results,
            'summary': summary
        }
        
        self.interpret_results("Gap Analysis", results, table_name)

    def duplicate_detection(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Duplicate Detection"))
        image_paths = []
        
        # Find exact duplicates
        exact_duplicates = df.duplicated()
        exact_duplicate_count = exact_duplicates.sum()
        exact_duplicate_percentage = (exact_duplicate_count / len(df)) * 100
        
        # Find potential near-duplicates (only check a subset of rows for efficiency)
        near_duplicates = 0
        near_duplicate_samples = []
        
        if len(df) > 0:
            # Check a subset of columns for near-duplicates
            subset_cols = df.select_dtypes(include=['object']).columns.tolist()[:3]
            if subset_cols:
                # Get counts of value combinations
                value_counts = df[subset_cols].value_counts()
                # Find combinations that appear more than once but aren't exact duplicates
                potential_near_dups = value_counts[value_counts > 1].index.tolist()
                near_duplicates = len(potential_near_dups)
                
                # Get sample near-duplicate rows
                if near_duplicates > 0 and len(potential_near_dups) > 0:
                    for vals in potential_near_dups[:3]:  # Limit to first 3 sets of values
                        matches = df
                        for i, col in enumerate(subset_cols):
                            matches = matches[matches[col] == vals[i]]
                        if len(matches) > 1:
                            near_duplicate_samples.append(matches.iloc[:2].to_dict('records'))
        
        def plot_duplicate_distribution():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
            
            # Bar plot
            ax1.bar(['Unique', 'Exact Duplicate', 'Near Duplicate'], 
                   [len(df) - exact_duplicate_count - near_duplicates, exact_duplicate_count, near_duplicates])
            ax1.set_title('Distribution of Unique vs Duplicate Rows')
            ax1.set_ylabel('Count')
            for i, v in enumerate([len(df) - exact_duplicate_count - near_duplicates, 
                                  exact_duplicate_count, near_duplicates]):
                ax1.text(i, v, str(v), ha='center', va='bottom')
            
            # Pie chart
            ax2.pie([len(df) - exact_duplicate_count - near_duplicates, exact_duplicate_count, near_duplicates], 
                    labels=['Unique', 'Exact Duplicate', 'Near Duplicate'], 
                    autopct='%1.1f%%', 
                    startangle=90)
            ax2.set_title('Proportion of Unique vs Duplicate Rows')
            
            plt.tight_layout()
            return fig, (ax1, ax2)

        result = self.generate_plot(plot_duplicate_distribution)
        if result is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_duplicate_distribution.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(img_path)
        
        duplicate_results = {
            'exact_duplicate_count': int(exact_duplicate_count),
            'exact_duplicate_percentage': float(exact_duplicate_percentage),
            'near_duplicate_count': near_duplicates,
            'near_duplicate_percentage': (near_duplicates / len(df)) * 100 if len(df) > 0 else 0,
            'total_rows': len(df),
            'unique_rows': len(df) - exact_duplicate_count - near_duplicates,
            'duplicate_samples': df[exact_duplicates].head(3).to_dict('records') if exact_duplicate_count > 0 else [],
            'near_duplicate_samples': near_duplicate_samples
        }
        
        results = {
            'image_paths': image_paths,
            'duplicate_results': duplicate_results
        }
        
        self.interpret_results("Duplicate Detection", results, table_name)

    def process_mining(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Process Mining"))
        image_paths = []
        
        # Check if required columns exist or find suitable alternatives
        case_id_col = None
        activity_col = None
        timestamp_col = None
        
        # Look for case_id column
        for col in df.columns:
            if 'case' in col.lower() or 'id' in col.lower():
                case_id_col = col
                break
        
        # Look for activity column
        for col in df.columns:
            if 'activ' in col.lower() or 'task' in col.lower() or 'step' in col.lower():
                activity_col = col
                break
                
        # Look for timestamp column
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            timestamp_col = date_cols[0]
        else:
            # Try to convert string columns to datetime
            for col in df.select_dtypes(include=['object']).columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                        timestamp_col = col
                        break
                    except:
                        continue
        
        if not all([case_id_col, activity_col, timestamp_col]):
            print(warning("Required columns for process mining not found."))
            return
        
        try:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                
            # Sort by case_id and timestamp
            df = df.sort_values([case_id_col, timestamp_col])
            
            # Get process sequences
            process_sequences = df.groupby(case_id_col)[activity_col].agg(list)
            
            # Convert lists to tuples for counting
            sequence_tuples = process_sequences.apply(tuple)
            unique_sequences = sequence_tuples.value_counts()
            
            # Calculate transition frequencies
            transitions = {}
            for seq in process_sequences:
                for i in range(len(seq) - 1):
                    transition = (seq[i], seq[i+1])
                    transitions[transition] = transitions.get(transition, 0) + 1
            
            # Sort transitions by frequency
            sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
            
            # Get actual activity names
            activity_names = df[activity_col].unique().tolist()
            
            def plot_process_flow():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                
                # Top process sequences
                top_sequences = unique_sequences.head(10)
                top_sequences.plot(kind='bar', ax=ax1)
                ax1.set_title('Top 10 Process Sequences')
                ax1.set_xlabel('Process Sequence')
                ax1.set_ylabel('Frequency')
                ax1.set_xticklabels([])  # Hide sequence labels as they can be too long
                
                # Top transitions
                top_transitions = sorted_transitions[:10]
                transition_labels = [f"{t[0][0]} → {t[0][1]}" for t in top_transitions]
                transition_values = [t[1] for t in top_transitions]
                ax2.barh(transition_labels, transition_values)
                ax2.set_title('Top 10 Activity Transitions')
                ax2.set_xlabel('Frequency')
                ax2.set_ylabel('Transition')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_process_flow)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_process_mining.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            
            # Calculate process statistics
            process_mining_results = {
                'total_cases': len(process_sequences),
                'unique_sequence_count': len(unique_sequences),
                'average_activities_per_case': process_sequences.apply(len).mean(),
                'most_common_sequence': str(unique_sequences.index[0]) if len(unique_sequences) > 0 else "N/A",
                'most_common_sequence_count': int(unique_sequences.iloc[0]) if len(unique_sequences) > 0 else 0,
                'top_transitions': [
                    {
                        'from': str(t[0][0]),
                        'to': str(t[0][1]),
                        'frequency': int(t[1])
                    } for t in sorted_transitions[:10]
                ],
                'activity_names': activity_names
            }
            
            results = {
                'image_paths': image_paths,
                'process_mining_results': process_mining_results
            }
            
            self.interpret_results("Process Mining", results, table_name)
        except Exception as e:
            print(warning(f"Error during process mining: {str(e)}"))
            self.interpret_results("Process Mining", {'error': str(e)}, table_name)

    def data_validation_techniques(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Data Validation Techniques"))
        image_paths = []
        
        # Calculate missing values
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        
        # Identify negative values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        negative_values = {col: (df[col] < 0).sum() for col in numeric_cols}
        
        # Check for out of range values (assuming some reasonable ranges)
        out_of_range_values = {}
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            out_of_range = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            out_of_range_values[col] = out_of_range
        
        def plot_data_validation():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
            
            # Missing values bar chart
            missing_pct = missing_percentage.sort_values(ascending=False)
            missing_pct = missing_pct[missing_pct > 0]  # Only show columns with missing values
            
            if len(missing_pct) > 0:
                missing_pct.plot(kind='bar', ax=ax1)
                ax1.set_title('Missing Values by Column')
                ax1.set_xlabel('Column')
                ax1.set_ylabel('Missing Percentage')
                ax1.set_ylim(0, 100)
                plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            else:
                ax1.text(0.5, 0.5, 'No Missing Values', 
                        ha='center', va='center', 
                        fontsize=12, transform=ax1.transAxes)
                ax1.set_title('Missing Values')
            
            # Out of range values bar chart
            out_of_range_series = pd.Series(out_of_range_values)
            out_of_range_series = out_of_range_series.sort_values(ascending=False)
            out_of_range_series = out_of_range_series[out_of_range_series > 0]  # Only show columns with out of range values
            
            if len(out_of_range_series) > 0:
                out_of_range_series.plot(kind='bar', ax=ax2)
                ax2.set_title('Out of Range Values by Column')
                ax2.set_xlabel('Column')
                ax2.set_ylabel('Count')
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            else:
                ax2.text(0.5, 0.5, 'No Out of Range Values', 
                        ha='center', va='center', 
                        fontsize=12, transform=ax2.transAxes)
                ax2.set_title('Out of Range Values')
            
            plt.tight_layout()
            return fig, (ax1, ax2)

        result = self.generate_plot(plot_data_validation)
        if result is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_data_validation.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(img_path)
        
        # Check for potentially invalid date values
        date_validity = {}
        for col in df.select_dtypes(include=['datetime64']).columns:
            future_dates = (df[col] > datetime.now()).sum()
            very_old_dates = (df[col] < datetime(1900, 1, 1)).sum()
            date_validity[col] = {
                'future_dates': int(future_dates),
                'very_old_dates': int(very_old_dates)
            }
        
        # Format results
        validation_results = {
            'missing_values': missing_values.to_dict(),
            'missing_percentage': missing_percentage.to_dict(),
            'negative_values': negative_values,
            'out_of_range_values': out_of_range_values,
            'date_validity': date_validity,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'summary': {
                'columns_with_missing_data': (missing_values > 0).sum(),
                'columns_with_out_of_range_values': sum(1 for v in out_of_range_values.values() if v > 0),
                'columns_with_negative_values': sum(1 for v in negative_values.values() if v > 0)
            }
        }
        
        results = {
            'image_paths': image_paths,
            'validation_results': validation_results
        }
        
        self.interpret_results("Data Validation Techniques", results, table_name)

    def risk_scoring_models(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Risk Scoring Models"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print(warning("Not enough numeric columns for risk scoring model."))
            return
        
        try:
            # Simple risk scoring model: weighted sum of normalized values
            # First, handle NaN values
            numeric_df = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            # Create a scaler
            scaler = MinMaxScaler()
            normalized_df = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_cols)
            
            # Assign weights (using simple approach where all features have equal weight)
            weights = pd.Series(1, index=numeric_cols)
            
            # Calculate risk scores
            risk_scores = (normalized_df * weights).sum(axis=1)
            
            # Define risk categories
            low_threshold = risk_scores.quantile(0.5)
            high_threshold = risk_scores.quantile(0.9)
            
            risk_categories = pd.cut(
                risk_scores, 
                bins=[float('-inf'), low_threshold, high_threshold, float('inf')],
                labels=['Low', 'Medium', 'High']
            )
            
            # Count risk categories
            risk_counts = risk_categories.value_counts()
            low_risk = risk_counts.get('Low', 0)
            medium_risk = risk_counts.get('Medium', 0)
            high_risk = risk_counts.get('High', 0)
            
            # Get high risk examples
            high_risk_examples = []
            if 'High' in risk_categories.values:
                high_risk_indices = risk_categories[risk_categories == 'High'].index[:3]  # Get top 3
                high_risk_examples = df.iloc[high_risk_indices].to_dict('records')
            
            def plot_risk_distribution():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                
                # Histogram
                ax1.hist(risk_scores, bins=20)
                ax1.axvline(x=low_threshold, color='y', linestyle='--', label=f'Low Threshold: {low_threshold:.2f}')
                ax1.axvline(x=high_threshold, color='r', linestyle='--', label=f'High Threshold: {high_threshold:.2f}')
                ax1.set_title('Distribution of Risk Scores')
                ax1.set_xlabel('Risk Score')
                ax1.set_ylabel('Frequency')
                ax1.legend()
                
                # Pie chart
                ax2.pie([low_risk, medium_risk, high_risk], 
                        labels=['Low Risk', 'Medium Risk', 'High Risk'], 
                        autopct='%1.1f%%', 
                        startangle=90)
                ax2.set_title('Risk Categories Distribution')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_risk_distribution)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_risk_distribution.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            
            # Calculate contribution of each feature to risk score
            feature_contributions = {}
            for col in numeric_cols:
                contrib = (normalized_df[col] * weights[col]).mean()
                feature_contributions[col] = float(contrib)
            
            risk_scoring_results = {
                'average_risk_score': float(risk_scores.mean()),
                'median_risk_score': float(risk_scores.median()),
                'risk_score_std_dev': float(risk_scores.std()),
                'high_risk_threshold': float(high_threshold),
                'low_risk_threshold': float(low_threshold),
                'high_risk_count': int(high_risk),
                'medium_risk_count': int(medium_risk),
                'low_risk_count': int(low_risk),
                'high_risk_percentage': float((high_risk / len(df)) * 100),
                'feature_contributions': feature_contributions,
                'high_risk_examples': high_risk_examples
            }
            
            results = {
                'image_paths': image_paths,
                'risk_scoring_results': risk_scoring_results
            }
            
            self.interpret_results("Risk Scoring Models", results, table_name)
        except Exception as e:
            print(warning(f"Error during risk scoring: {str(e)}"))
            self.interpret_results("Risk Scoring Models", {'error': str(e)}, table_name)

    def fuzzy_matching(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Fuzzy Matching"))
        image_paths = []
        
        text_cols = df.select_dtypes(include=['object']).columns
        
        if len(text_cols) == 0:
            print(warning("No text columns for fuzzy matching."))
            return
        
        fuzzy_results = {}
        for col in text_cols:
            # Skip columns with large number of unique values or long text fields
            unique_values = df[col].dropna().unique()
            if len(unique_values) > 100 or any(len(str(val)) > 100 for val in unique_values):
                continue
            
            # Sort unique values by frequency to prioritize common values
            value_counts = df[col].value_counts()
            # Keep only top 100 values for efficiency
            common_values = value_counts.head(100).index.tolist()
            
            matches = []
            for i, val1 in enumerate(common_values):
                for val2 in common_values[i+1:]:
                    # Skip if either value is NaN
                    if pd.isna(val1) or pd.isna(val2):
                        continue
                        
                    # Skip if values are identical or empty
                    if str(val1) == str(val2) or not str(val1).strip() or not str(val2).strip():
                        continue
                        
                    ratio = fuzz.ratio(str(val1), str(val2))
                    if ratio > 80:  # Consider as a match if similarity > 80%
                        matches.append({
                            'value1': str(val1),
                            'value2': str(val2),
                            'similarity': ratio,
                            'frequency1': int(value_counts.get(val1, 0)),
                            'frequency2': int(value_counts.get(val2, 0))
                        })
            
            fuzzy_results[col] = matches
        
        def plot_fuzzy_matches():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
            
            # Bar chart of match counts
            match_counts = {col: len(matches) for col, matches in fuzzy_results.items()}
            if match_counts:
                cols = list(match_counts.keys())
                counts = list(match_counts.values())
                ax1.bar(cols, counts)
                ax1.set_title('Fuzzy Matches per Column')
                ax1.set_xlabel('Columns')
                ax1.set_ylabel('Number of Fuzzy Matches')
                plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            else:
                ax1.text(0.5, 0.5, 'No Fuzzy Matches Found', 
                        ha='center', va='center', 
                        fontsize=12, transform=ax1.transAxes)
                ax1.set_title('Fuzzy Matches')
            
            # Bar chart of potential duplicates saved
            duplicate_savings = {}
            for col, matches in fuzzy_results.items():
                savings = sum(match['frequency2'] for match in matches)
                if savings > 0:
                    duplicate_savings[col] = savings
            
            if duplicate_savings:
                cols = list(duplicate_savings.keys())
                savings = list(duplicate_savings.values())
                ax2.bar(cols, savings)
                ax2.set_title('Potential Records Affected')
                ax2.set_xlabel('Columns')
                ax2.set_ylabel('Number of Records')
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            else:
                ax2.text(0.5, 0.5, 'No Duplicate Savings', 
                        ha='center', va='center', 
                        fontsize=12, transform=ax2.transAxes)
                ax2.set_title('Potential Records Affected')
            
            plt.tight_layout()
            return fig, (ax1, ax2)

        result = self.generate_plot(plot_fuzzy_matches)
        if result is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_fuzzy_matches.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(img_path)
        
        # Prepare summary of results
        total_matches = sum(len(matches) for matches in fuzzy_results.values())
        
        # Get top matches with highest similarity
        all_matches = []
        for col, matches in fuzzy_results.items():
            for match in matches:
                match['column'] = col
                all_matches.append(match)
        
        top_matches = sorted(all_matches, key=lambda x: x['similarity'], reverse=True)[:10]
        
        results = {
            'image_paths': image_paths,
            'fuzzy_results': fuzzy_results,
            'summary': {
                'total_columns_analyzed': len(text_cols),
                'columns_with_matches': len([col for col, matches in fuzzy_results.items() if matches]),
                'total_fuzzy_matches': total_matches,
                'top_matches': top_matches
            }
        }
        
        self.interpret_results("Fuzzy Matching", results, table_name)

    def continuous_auditing_techniques(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Continuous Auditing Techniques"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print(warning("No numeric columns for continuous auditing."))
            return
        
        audit_results = {}
        for col in numeric_cols:
            # Skip columns with all NaN values
            if df[col].isna().all():
                continue
                
            # Fill NaN values for calculations
            series = df[col].fillna(df[col].mean())
            
            # Calculate z-scores
            z_scores = np.abs(stats.zscore(series))
            outliers = np.where(z_scores > 3)[0]
            
            # Benford's Law analysis for positive integers
            benford_compliance = None
            if all(series > 0) and all(series.astype(int) == series):
                # Get first digits
                first_digits = series.astype(str).str[0].astype(int)
                digit_counts = first_digits.value_counts().sort_index()
                
                # Expected Benford distribution
                benford_expected = pd.Series(
                    [np.log10(1 + 1/d) * 100 for d in range(1, 10)],
                    index=range(1, 10)
                )
                
                # Calculate chi-square statistic
                observed = digit_counts.reindex(range(1, 10), fill_value=0)
                expected = benford_expected * len(series) / 100
                chi2_stat = sum((observed - expected)**2 / expected)
                chi2_pvalue = stats.chi2.sf(chi2_stat, 8)  # 8 degrees of freedom (9 digits - 1)
                
                benford_compliance = {
                    'chi2_stat': float(chi2_stat),
                    'p_value': float(chi2_pvalue),
                    'complies_with_benford': bool(chi2_pvalue > 0.05),
                    'first_digit_frequencies': {
                        str(digit): {
                            'observed': float(observed.get(digit, 0)),
                            'expected': float(expected.get(digit, 0)),
                            'observed_pct': float(observed.get(digit, 0) / len(series) * 100) if len(series) > 0 else 0,
                            'expected_pct': float(benford_expected.get(digit, 0))
                        } for digit in range(1, 10)
                    }
                }
            
            audit_results[col] = {
                'mean': float(series.mean()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'outliers_count': int(len(outliers)),
                'outliers_percentage': float(len(outliers) / len(series) * 100),
                'benford_analysis': benford_compliance
            }
        
        def plot_outliers_and_benford():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
            
            # Outliers bar chart
            outlier_counts = {col: results['outliers_count'] for col, results in audit_results.items()}
            if outlier_counts:
                cols = list(outlier_counts.keys())
                counts = list(outlier_counts.values())
                ax1.bar(cols, counts)
                ax1.set_title('Outliers per Column')
                ax1.set_xlabel('Columns')
                ax1.set_ylabel('Number of Outliers')
                plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            else:
                ax1.text(0.5, 0.5, 'No Outliers Found', 
                        ha='center', va='center', 
                        fontsize=12, transform=ax1.transAxes)
                ax1.set_title('Outliers per Column')
            
            # Benford's Law comparison for a column
            benford_cols = [col for col, results in audit_results.items() 
                           if results.get('benford_analysis') is not None]
            
            if benford_cols:
                sample_col = benford_cols[0]
                benford_data = audit_results[sample_col]['benford_analysis']
                
                observed = [benford_data['first_digit_frequencies'][str(d)]['observed_pct'] 
                           for d in range(1, 10)]
                expected = [benford_data['first_digit_frequencies'][str(d)]['expected_pct'] 
                           for d in range(1, 10)]
                
                x = np.arange(1, 10)
                width = 0.35
                
                ax2.bar(x - width/2, observed, width, label='Observed')
                ax2.bar(x + width/2, expected, width, label='Expected (Benford)')
                ax2.set_title(f"Benford's Law Analysis\n{sample_col}")
                ax2.set_xlabel('First Digit')
                ax2.set_ylabel('Percentage')
                ax2.set_xticks(x)
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, "No Benford's Law Analysis", 
                        ha='center', va='center', 
                        fontsize=12, transform=ax2.transAxes)
                ax2.set_title("Benford's Law Analysis")
            
            plt.tight_layout()
            return fig, (ax1, ax2)

        result = self.generate_plot(plot_outliers_and_benford)
        if result is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_continuous_auditing.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(img_path)
        
        # Summary stats
        total_outliers = sum(result['outliers_count'] for result in audit_results.values())
        benford_compliant = sum(1 for result in audit_results.values() 
                              if result.get('benford_analysis') is not None and 
                              result['benford_analysis']['complies_with_benford'])
        benford_analyzed = sum(1 for result in audit_results.values() 
                             if result.get('benford_analysis') is not None)
        
        results = {
            'image_paths': image_paths,
            'audit_results': audit_results,
            'summary': {
                'total_columns_analyzed': len(numeric_cols),
                'total_outliers': total_outliers,
                'columns_with_outliers': sum(1 for result in audit_results.values() if result['outliers_count'] > 0),
                'columns_benford_compliant': benford_compliant,
                'columns_benford_analyzed': benford_analyzed
            }
        }
        
        self.interpret_results("Continuous Auditing Techniques", results, table_name)

    def sensitivity_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Sensitivity Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print(warning("Not enough numeric columns for sensitivity analysis."))
            return
        
        try:
            # Perform simple sensitivity analysis: impact of 10% change in each variable
            baseline = df[numeric_cols].mean()
            sensitivity_results = {}
            
            for col in numeric_cols:
                changed = baseline.copy()
                # Avoid division by zero
                if baseline[col] != 0:
                    changed[col] *= 1.1  # 10% increase
                    impact = (changed - baseline) / baseline * 100
                    # Replace inf and -inf with large values
                    impact = impact.replace([np.inf, -np.inf], np.nan).fillna(0)
                    sensitivity_results[col] = impact.to_dict()
            
            # Create a sensitivity matrix
            sensitivity_matrix = pd.DataFrame(sensitivity_results)
            
            def plot_sensitivity():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                
                # Heatmap of sensitivity matrix
                sns.heatmap(sensitivity_matrix, annot=True, cmap='coolwarm', ax=ax1)
                ax1.set_title('Sensitivity Matrix\nImpact of 10% Increase in Row on Column (%)')
                
                # Bar chart of most sensitive variables
                # Calculate total impact of each variable
                total_impact = sensitivity_matrix.abs().sum()
                total_impact = total_impact.sort_values(ascending=False)
                
                total_impact.plot(kind='bar', ax=ax2)
                ax2.set_title('Total Sensitivity Impact by Variable')
                ax2.set_xlabel('Variables')
                ax2.set_ylabel('Total Absolute Impact (%)')
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_sensitivity)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_sensitivity_analysis.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            
            # Calculate most sensitive variables
            var_influence = {}
            for col in numeric_cols:
                # Calculate how much this variable affects others
                outgoing_impact = sum(abs(val) for val in sensitivity_matrix[col])
                # Calculate how much this variable is affected by others
                incoming_impact = sum(abs(sensitivity_matrix.loc[col, other_col]) for other_col in numeric_cols if other_col != col)
                
                var_influence[col] = {
                    'outgoing_impact': float(outgoing_impact),
                    'incoming_impact': float(incoming_impact),
                    'net_impact': float(outgoing_impact - incoming_impact)
                }
            
            # Find top influencers and influencees
            sorted_by_outgoing = sorted(var_influence.items(), key=lambda x: x[1]['outgoing_impact'], reverse=True)
            sorted_by_incoming = sorted(var_influence.items(), key=lambda x: x[1]['incoming_impact'], reverse=True)
            
            top_influencers = [{'variable': var, 'impact': data['outgoing_impact']} 
                              for var, data in sorted_by_outgoing[:3]]
            top_influencees = [{'variable': var, 'impact': data['incoming_impact']} 
                              for var, data in sorted_by_incoming[:3]]
            
            results = {
                'image_paths': image_paths,
                'sensitivity_results': sensitivity_results,
                'variables_influence': var_influence,
                'top_influencers': top_influencers,
                'top_influencees': top_influencees
            }
            
            self.interpret_results("Sensitivity Analysis", results, table_name)
        except Exception as e:
            print(warning(f"Error during sensitivity analysis: {str(e)}"))
            self.interpret_results("Sensitivity Analysis", {'error': str(e)}, table_name)

    def scenario_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Scenario Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print(warning("No numeric columns for scenario analysis."))
            return
        
        # Define scenarios
        scenarios = {
            'Base Case': {'factor': 1.0, 'description': 'Current values without changes'},
            'Optimistic': {'factor': 1.2, 'description': '20% increase in all metrics'},
            'Pessimistic': {'factor': 0.8, 'description': '20% decrease in all metrics'},
            'Mixed': {'factor': None, 'description': 'Random variation in different metrics'}
        }
        
        # For mixed scenario, assign random factors between 0.7 and 1.3
        np.random.seed(42)  # For reproducibility
        mixed_factors = np.random.uniform(0.7, 1.3, size=len(numeric_cols))
        mixed_factors_dict = dict(zip(numeric_cols, mixed_factors))
        
        # Calculate scenario values
        scenario_results = {}
        for scenario_name, scenario_info in scenarios.items():
            if scenario_name == 'Mixed':
                scenario_values = {col: df[col].mean() * mixed_factors_dict[col] for col in numeric_cols}
            else:
                scenario_values = {col: df[col].mean() * scenario_info['factor'] for col in numeric_cols}
            
            scenario_results[scenario_name] = {
                'values': scenario_values,
                'description': scenario_info['description'],
                'factor': scenario_info['factor'] if scenario_name != 'Mixed' else mixed_factors_dict
            }
        
        def plot_scenario_comparison():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
            
            # Bar chart comparing scenarios
            scenario_sums = {scenario: sum(values['values'].values()) 
                            for scenario, values in scenario_results.items()}
            
            scenarios = list(scenario_sums.keys())
            sums = list(scenario_sums.values())
            ax1.bar(scenarios, sums)
            ax1.set_title('Total Sum of Metrics by Scenario')
            ax1.set_xlabel('Scenario')
            ax1.set_ylabel('Sum of Metrics')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Spider plot for comparing metrics across scenarios
            # Select a subset of metrics for readability
            selected_metrics = list(numeric_cols)[:5]  # First 5 metrics
            
            # Number of variables
            N = len(selected_metrics)
            
            # Calculate angle for each axis
            angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
            angles += angles[:1]  # Close the polygon
            
            # Set up the axes
            ax2.set_theta_offset(np.pi / 2)
            ax2.set_theta_direction(-1)
            ax2.set_rlabel_position(0)
            
            # Plot each scenario
            for scenario in scenarios:
                values = [scenario_results[scenario]['values'][metric] for metric in selected_metrics]
                # Close the polygon
                values += values[:1]
                
                # Plot values
                ax2.plot(angles, values, linewidth=1, label=scenario)
                ax2.fill(angles, values, alpha=0.1)
            
            # Set the labels
            plt.xticks(angles[:-1], selected_metrics)
            ax2.set_title('Scenario Comparison by Metric')
            ax2.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plt.tight_layout()
            return fig, (ax1, ax2)

        result = self.generate_plot(plot_scenario_comparison)
        if result is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_scenario_comparison.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(img_path)
        
        # Calculate scenario comparisons
        comparisons = {}
        for scenario in scenarios:
            if scenario == 'Base Case':
                continue
                
            comparison = {}
            for col in numeric_cols:
                base_value = scenario_results['Base Case']['values'][col]
                scenario_value = scenario_results[scenario]['values'][col]
                
                if base_value != 0:
                    percentage_change = ((scenario_value - base_value) / base_value) * 100
                else:
                    percentage_change = np.nan
                
                comparison[col] = {
                    'base_value': float(base_value),
                    'scenario_value': float(scenario_value),
                    'absolute_change': float(scenario_value - base_value),
                    'percentage_change': float(percentage_change) if not np.isnan(percentage_change) else "N/A"
                }
            
            comparisons[scenario] = comparison
        
        results = {
            'image_paths': image_paths,
            'scenario_results': scenario_results,
            'scenario_comparisons': comparisons
        }
        
        self.interpret_results("Scenario Analysis", results, table_name)

    def monte_carlo_simulation(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Monte Carlo Simulation"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print(warning("No numeric columns for Monte Carlo simulation."))
            return
        
        try:
            # Perform simple Monte Carlo simulation
            n_simulations = 1000
            simulation_results = {}
            
            for col in numeric_cols:
                # Skip if standard deviation is zero (constant values)
                if df[col].std() == 0:
                    continue
                    
                mean = df[col].mean()
                std = df[col].std()
                
                # Generate simulations
                simulations = np.random.normal(mean, std, n_simulations)
                
                # Calculate statistics
                simulation_results[col] = {
                    'mean': float(np.mean(simulations)),
                    'median': float(np.median(simulations)),
                    '5th_percentile': float(np.percentile(simulations, 5)),
                    '95th_percentile': float(np.percentile(simulations, 95)),
                    'std_dev': float(np.std(simulations)),
                    'min': float(np.min(simulations)),
                    'max': float(np.max(simulations))
                }
                
                def plot_monte_carlo():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    
                    # Histogram with KDE
                    sns.histplot(simulations, kde=True, ax=ax)
                    
                    # Add vertical lines for key statistics
                    ax.axvline(x=simulation_results[col]['mean'], color='r', linestyle='--', 
                              label=f"Mean: {simulation_results[col]['mean']:.2f}")
                    ax.axvline(x=simulation_results[col]['5th_percentile'], color='g', linestyle='--', 
                              label=f"5th %ile: {simulation_results[col]['5th_percentile']:.2f}")
                    ax.axvline(x=simulation_results[col]['95th_percentile'], color='y', linestyle='--', 
                              label=f"95th %ile: {simulation_results[col]['95th_percentile']:.2f}")
                    
                    ax.set_title(f'Monte Carlo Simulation: {col} ({n_simulations:,} runs)')
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
                    ax.legend()
                    
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_monte_carlo)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_monte_carlo_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)
                else:
                    print(f"Skipping Monte Carlo plot for {col} due to timeout.")
            
            # Calculate probability of extreme values
            probability_analysis = {}
            for col in numeric_cols:
                if col not in simulation_results:
                    continue
                    
                # Calculate probability of value being more than 50% above or below mean
                mean = df[col].mean()
                
                if mean != 0:
                    high_threshold = mean * 1.5
                    low_threshold = mean * 0.5
                    
                    # Generate simulations
                    simulations = np.random.normal(mean, df[col].std(), n_simulations)
                    
                    prob_high = np.mean(simulations > high_threshold) * 100
                    prob_low = np.mean(simulations < low_threshold) * 100
                    
                    probability_analysis[col] = {
                        'mean': float(mean),
                        'high_threshold': float(high_threshold),
                        'low_threshold': float(low_threshold),
                        'prob_above_high': float(prob_high),
                        'prob_below_low': float(prob_low)
                    }
            
            results = {
                'image_paths': image_paths,
                'simulation_results': simulation_results,
                'probability_analysis': probability_analysis,
                'simulation_parameters': {
                    'n_simulations': n_simulations,
                    'distribution': 'Normal',
                    'parameters': {
                        col: {
                            'mean': float(df[col].mean()),
                            'std_dev': float(df[col].std())
                        } for col in numeric_cols if col in simulation_results
                    }
                }
            }
            
            self.interpret_results("Monte Carlo Simulation", results, table_name)
        except Exception as e:
            print(warning(f"Error during Monte Carlo simulation: {str(e)}"))
            self.interpret_results("Monte Carlo Simulation", {'error': str(e)}, table_name)

    def kpi_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - KPI Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print(warning("No numeric columns for KPI analysis."))
            return
        
        # Define some example KPIs
        kpis = {
            'Average': np.mean,
            'Median': np.median,
            'Standard Deviation': np.std,
            'Coefficient of Variation': lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else np.nan,
            'Range': lambda x: np.max(x) - np.min(x),
            'Interquartile Range': lambda x: np.percentile(x, 75) - np.percentile(x, 25)
        }
        
        kpi_results = {}
        for col in numeric_cols:
            # Skip columns with all NaN values
            if df[col].isna().all():
                continue
                
            # Calculate KPIs
            kpi_values = {}
            for kpi_name, kpi_func in kpis.items():
                try:
                    value = kpi_func(df[col].dropna())
                    kpi_values[kpi_name] = float(value) if not np.isnan(value) else "N/A"
                except Exception as e:
                    kpi_values[kpi_name] = f"Error: {str(e)}"
            
            kpi_results[col] = kpi_values
        
        def plot_kpis():
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            
            # Create a heatmap of KPI values
            kpi_df = pd.DataFrame(kpi_results).T
            
            # Replace non-numeric values with NaN for plotting
            for col in kpi_df.columns:
                kpi_df[col] = pd.to_numeric(kpi_df[col], errors='coerce')
            
            # Create heatmap with annotations
            sns.heatmap(kpi_df, annot=True, cmap='YlGnBu', ax=ax, fmt='.2f')
            ax.set_title('KPI Analysis by Column')
            ax.set_ylabel('Columns')
            ax.set_xlabel('KPIs')
            
            plt.tight_layout()
            return fig, ax

        result = self.generate_plot(plot_kpis)
        if result is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_kpi_analysis.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(img_path)
        
        # Determine target KPI values based on business rules
        # For demonstration, we'll use 90th percentile as a target
        target_values = {}
        for col in numeric_cols:
            if col not in kpi_results:
                continue
                
            target_values[col] = {
                'target_value': float(df[col].quantile(0.9)),
                'current_value': float(df[col].mean()),
                'gap': float(df[col].quantile(0.9) - df[col].mean()),
                'achievement_percentage': float(df[col].mean() / df[col].quantile(0.9) * 100) if df[col].quantile(0.9) != 0 else "N/A"
            }
        
        results = {
            'image_paths': image_paths,
            'kpi_results': kpi_results,
            'target_values': target_values
        }
        
        self.interpret_results("KPI Analysis", results, table_name)

    def interpret_results(self, analysis_type, results, table_name):
        technique_info = get_technique_info(analysis_type)

        if isinstance(results, dict) and "Numeric Statistics" in results:
            numeric_stats = results["Numeric Statistics"]
            categorical_stats = results["Categorical Statistics"]
            
            numeric_table = "| Statistic | " + " | ".join(numeric_stats.keys()) + " |\n"
            numeric_table += "| --- | " + " | ".join(["---" for _ in numeric_stats.keys()]) + " |\n"
            for stat in numeric_stats[list(numeric_stats.keys())[0]].keys():
                numeric_table += f"| {stat} | " + " | ".join([f"{numeric_stats[col][stat]:.2f}" for col in numeric_stats.keys()]) + " |\n"
            
            categorical_summary = "\n".join([f"{col}:\n" + "\n".join([f"  - {value}: {count}" for value, count in stats.items()]) for col, stats in categorical_stats.items()])
            
            results_str = f"Numeric Statistics:\n{numeric_table}\n\nCategorical Statistics:\n{categorical_summary}"
        elif isinstance(results, pd.DataFrame):
            results_str = f"DataFrame with shape {results.shape}:\n{results.to_string()}"
        elif isinstance(results, dict):
            results_str = "\n".join([f"{k}: {v}" for k, v in results.items() if k != 'image_paths'])
        else:
            results_str = str(results)

        # Add information about number of visualizations
        num_visualizations = len(results.get('image_paths', []))
        results_str += f"\n\nNumber of visualizations created: {num_visualizations}"

        # Add entity information
        entity_description = self.format_entity_description()

        common_prompt = f"""
        Analysis type: {analysis_type}
        Table name: {table_name}
        
        Database Description:
        {self.database_description}
        
        Entities in the data:
        {entity_description}

        Technique Context:
        {technique_info['context']}

        Results:
        {results_str}

        Interpretation Guidelines:
        {technique_info['guidelines']}
        """

        worker_prompt = f"""
        You are an expert data analyst providing insights on exploratory data analysis results. Your task is to interpret the following analysis results and provide a detailed, data-driven interpretation, focusing on discovering patterns and hidden insights. Avoid jargon.

        {common_prompt}

        Please provide a thorough interpretation of these results, highlighting noteworthy patterns, anomalies, or insights. Focus on aspects that would be valuable for business decisions and operational improvements. Always provide specific numbers and percentages.
        
        When referring to entities in the data (like teams, products, regions, etc.), use their actual names from the data rather than generic terms like "Team A" or "Cluster 1".

        Structure your response in the following format:

        1. Analysis performed and Key Insights:
        [Briefly describe the analysis performed. List at least 2-3 important insights discovered, with relevant numbers and percentages. Provide detailed explanations for each insight.]

        2. Patterns and Trends:
        [Describe at least 2-3 significant patterns or trends observed in the data. Explain their potential significance.]

        3. Potential Issues:
        [Highlight any anomalies, unusual trends, or areas of concern. Mention at least 2-3 potential problems, red flags, audit findings, fraud cases always including relevant numbers and percentages.]

        Ensure your interpretation is comprehensive and focused on actionable insights. While you can be detailed, strive for clarity in your explanations. Use technical terms when necessary, but provide brief explanations for complex concepts.

        Interpretation:
        """

        worker_interpretation = self.worker_erag_api.chat([{"role": "system", "content": "You are an expert data analyst providing insights for business leaders and analysts. Respond in the requested format."}, 
                                                    {"role": "user", "content": worker_prompt}])

        supervisor_prompt = f"""
        You are an expert data analyst providing insights on exploratory data analysis results. Your task is to interpret the following analysis results and provide a detailed, data-driven interpretation.

        {common_prompt}

        Please provide a thorough interpretation of these results, highlighting noteworthy patterns, anomalies, or insights. Focus on the most important aspects that would be valuable for business operations and decision-making. Always provide specific numbers and percentages when discussing findings.
        
        When referring to entities in the data (like teams, products, regions, etc.), use their actual names from the data rather than generic terms like "Team A" or "Cluster 1".
        
        If some data appears to be missing or incomplete, work with the available information without mentioning the limitations. Your goal is to extract as much insight as possible from the given data.
        
        Structure your response in the following format:
        1. Analysis:
        [Provide a detailed description of the analysis performed, including specific metrics and their values]
        2. Key Findings:
        [List the most important discoveries, always including relevant numbers and percentages]
        3. Implications:
        [Discuss the potential impact of these findings on business operations and decision-making]
        4. Operational Recommendations:
        [Suggest concrete operational steps or changes based on these results. Focus on actionable recommendations that can improve business processes, efficiency, or outcomes. Avoid recommending further data analysis.]
        
        Ensure your interpretation is concise yet comprehensive, focusing on actionable insights derived from the data that can be directly applied to business operations.

        Business Analysis:
        """

        supervisor_analysis = self.supervisor_erag_api.chat([
            {"role": "system", "content": "You are a senior business analyst providing insights based on data analysis results. Provide a concise yet comprehensive business analysis."},
            {"role": "user", "content": supervisor_prompt}
        ])

        combined_interpretation = f"""
        Data Analysis:
        {worker_interpretation.strip()}

        Business Analysis:
        {supervisor_analysis.strip()}
        """

        print(success(f"Combined Interpretation for {analysis_type}:"))
        print(combined_interpretation.strip())

        self.text_output += f"\n{combined_interpretation.strip()}\n\n"
        
        # Save individual interpretation to file
        interpretation_file = os.path.join(self.output_folder, f"{analysis_type.lower().replace(' ', '_')}_interpretation.txt")
        with open(interpretation_file, "w", encoding='utf-8') as f:
            f.write(combined_interpretation.strip())
        print(success(f"Interpretation saved to file: {interpretation_file}"))
        
        # Save raw results to file
        results_file = os.path.join(self.output_folder, f"{analysis_type.lower().replace(' ', '_')}_results.txt")
        with open(results_file, "w", encoding='utf-8') as f:
            f.write(f"Results for {analysis_type}:\n\n")
            f.write(results_str)
        print(success(f"Results saved to file: {results_file}"))

        # Handle images for the PDF report
        image_data = []
        if isinstance(results, dict) and 'image_paths' in results:
            for img in results['image_paths']:
                if isinstance(img, tuple) and len(img) == 2:
                    image_data.append(img)
                elif isinstance(img, str):
                    image_data.append((analysis_type, img))

        # Prepare content for PDF report
        pdf_content = f"""
        # {analysis_type}

        ## Data Analysis
        {worker_interpretation.strip()}

        
        ## Business Analysis
        {supervisor_analysis.strip()}
        """

        self.pdf_content.append((analysis_type, image_data, pdf_content))

        # Extract important findings
        self.findings.append(f"{analysis_type}:")
        lines = combined_interpretation.strip().split('\n')
        for i, line in enumerate(lines):
            if line.startswith("1. Analysis performed and Key Insights:") or line.startswith("2. Key Findings:"):
                for finding in lines[i+1:]:
                    if finding.strip() and not finding.startswith(("2.", "3.", "4.")):
                        self.findings.append(finding.strip())
                    elif finding.startswith(("2.", "3.", "4.")):
                        break

        # Update self.image_data for the PDF report
        self.image_data.extend(image_data)

    def save_text_output(self):
        output_file = os.path.join(self.output_folder, "axda_b6_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        report_title = f"Advanced Exploratory Data Analysis (Batch 6) Report for {self.table_name}"
        
        # Add database description to the report
        if self.database_description:
            report_title += f"\nDatabase Description: {self.database_description}"
        
        # Ensure all image data is in the correct format
        formatted_image_data = []
        for item in self.pdf_content:
            analysis_type, images, interpretation = item
            if isinstance(images, list):
                for image in images:
                    if isinstance(image, tuple) and len(image) == 2:
                        formatted_image_data.append(image)
                    elif isinstance(image, str):
                        # If it's just a string (path), use the analysis type as the title
                        formatted_image_data.append((analysis_type, image))
            elif isinstance(images, str):
                # If it's just a string (path), use the analysis type as the title
                formatted_image_data.append((analysis_type, images))
        
        pdf_file = self.pdf_generator.create_enhanced_pdf_report(
            self.findings,
            self.pdf_content,
            formatted_image_data,  # Use the formatted image data
            filename=f"axda_b6_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
            return pdf_file
        else:
            print(error("Failed to generate PDF report"))
            return None