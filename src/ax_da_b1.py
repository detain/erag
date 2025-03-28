import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, anderson, pearsonr, probplot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import networkx as nx
import os
import signal
import sys
import threading
import time
from functools import wraps

from src.api_model import EragAPI
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight
from src.print_pdf import PDFReportGenerator
from src.helper_da import get_technique_info
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

class TimeoutException(Exception):
    pass

class AdvancedExploratoryDataAnalysisB1:
    def __init__(self, worker_erag_api, supervisor_erag_api, db_path):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.db_path = db_path
        self.technique_counter = 1
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

    @timeout(10)
    def generate_plot(self, plot_function, *args, **kwargs):
        return plot_function(*args, **kwargs)

    def prompt_for_database_description(self):
        """Ask the user for a description of the database"""
        print(info("Please provide a description of the database for advanced analysis. This will help the AI models provide better insights."))
        print(info("Describe the purpose, main tables, key data points, and any other relevant information:"))
        self.database_description = input("> ")
        print(success(f"Database description received: {self.database_description}"))

    def run(self):
        self.prompt_for_database_description()
        print(info(f"Starting Advanced Exploratory Data Analysis on {self.db_path}"))
        
        tables = self.get_tables()
        for table in tables:
            self.analyze_table(table)
        
        self.save_text_output()
        self.generate_pdf_report()
        print(success(f"Advanced Exploratory Data Analysis completed. Results saved in {self.output_folder}"))

    def get_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]

    def analyze_table(self, table_name):
        self.table_name = table_name
        self.output_folder = os.path.join(settings.output_folder, f"axda_b1_{self.table_name}")
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, self.table_name)
        
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            print(info(f"Loaded full dataset with {len(df)} rows and {len(df.columns)} columns"))

        analysis_methods = [
            self.value_counts_analysis,
            self.grouped_summary_statistics,
            self.frequency_distribution_analysis,
            self.kde_plot_analysis,
            self.violin_plot_analysis,
            self.pair_plot_analysis,
            self.box_plot_analysis,
            self.scatter_plot_analysis,
            self.time_series_analysis,
            self.outlier_detection,
            self.feature_importance_analysis,
            self.pca_analysis,
            self.cluster_analysis,
            self.correlation_network_analysis,
            self.qq_plot_analysis
        ]

        for method in analysis_methods:
            try:
                # Check if execution is paused
                self.check_if_paused()
                method(df, table_name)
            except Exception as e:
                error_message = f"An error occurred during {method.__name__}: {str(e)}"
                print(error(error_message))
                self.text_output += f"\n{error_message}\n"
                
                # Write error to method-specific output file
                method_name = method.__name__
                with open(os.path.join(self.output_folder, f"{method_name}_results.txt"), "w", encoding='utf-8') as f:
                    f.write(error_message)
                
                # Add error to the PDF report
                self.pdf_content.append((method.__name__, [], error_message))
            finally:
                # Ensure we always increment the technique counter, even if the method fails
                self.technique_counter += 1

    def value_counts_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Value Counts Analysis"))
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        results = {}
        image_paths = []
        
        if len(categorical_columns) > 0:
            # Include actual category values for context
            for col in categorical_columns:
                value_counts = df[col].value_counts()
                
                # Get actual category names and values
                category_data = [
                    {"category": str(cat), "count": int(count), "percentage": float((count/len(df))*100)} 
                    for cat, count in value_counts.items()
                ]
                
                results[col] = {
                    "total_categories": len(value_counts),
                    "unique_values": df[col].nunique(),
                    "null_count": int(df[col].isna().sum()),
                    "category_data": category_data
                }
                
                # Create pie chart
                def plot_pie_chart():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    
                    # Limit to top 10 categories for readability
                    if len(value_counts) > 10:
                        top_n = value_counts.nlargest(9)
                        others = pd.Series({'Others': value_counts.iloc[9:].sum()})
                        plot_data = pd.concat([top_n, others])
                    else:
                        plot_data = value_counts
                    
                    plot_data.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                    ax.set_title(f'Distribution of {col}')
                    ax.set_ylabel('')  # Remove y-label
                    return fig, ax

                result = self.generate_plot(plot_pie_chart)
                if result is not None:
                    fig, ax = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_{col}_pie_chart.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append((f"{col} Distribution", img_path))
        else:
            results["note"] = "No categorical columns found for value counts analysis"
        
        results['image_paths'] = image_paths
        self.interpret_results("Value Counts Analysis", results, table_name)

    def grouped_summary_statistics(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Grouped Summary Statistics"))
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        results = {}
        
        if len(categorical_columns) > 0 and len(numerical_columns) > 0:
            # Include actual group names and values for context
            for cat_col in categorical_columns:
                for num_col in numerical_columns:
                    # Get actual group statistics with real category names
                    grouped_stats = df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std', 'min', 'max', 'count'])
                    
                    # Convert to a more informative format with actual group names
                    group_stats = []
                    for group_name, stats in grouped_stats.iterrows():
                        group_stats.append({
                            "group_name": str(group_name),
                            "count": int(stats['count']),
                            "mean": float(stats['mean']),
                            "median": float(stats['median']),
                            "std": float(stats['std']) if not np.isnan(stats['std']) else None,
                            "min": float(stats['min']),
                            "max": float(stats['max']),
                            "range": float(stats['max'] - stats['min'])
                        })
                    
                    results[f"{cat_col} - {num_col}"] = {
                        "groups": group_stats,
                        "total_groups": len(grouped_stats),
                        "overall_stats": {
                            "mean": float(df[num_col].mean()),
                            "median": float(df[num_col].median()),
                            "std": float(df[num_col].std()),
                            "min": float(df[num_col].min()),
                            "max": float(df[num_col].max())
                        }
                    }
                    
                    # Create bar chart of means by group
                    def plot_group_means():
                        fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                        grouped_stats['mean'].sort_values(ascending=False).plot(kind='bar', ax=ax)
                        ax.set_title(f'Mean {num_col} by {cat_col}')
                        ax.set_xlabel(cat_col)
                        ax.set_ylabel(f'Mean {num_col}')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        return fig, ax
                    
                    result = self.generate_plot(plot_group_means)
                    if result is not None:
                        fig, ax = result
                        img_path = os.path.join(self.output_folder, f"{table_name}_{cat_col}_{num_col}_group_means.png")
                        plt.savefig(img_path, dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        results.setdefault('image_paths', []).append((f"Mean {num_col} by {cat_col}", img_path))
        else:
            if len(categorical_columns) == 0:
                results["note"] = "No categorical columns found for grouping"
            else:
                results["note"] = "No numerical columns found for analysis"
        
        self.interpret_results("Grouped Summary Statistics", results, table_name)

    def frequency_distribution_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Frequency Distribution Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        results = {}
        
        if len(numerical_columns) > 0:
            # Include actual distribution statistics for each column
            for col in numerical_columns:
                # Calculate distribution statistics
                col_stats = {
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "mode": float(df[col].mode().iloc[0]) if not df[col].mode().empty else None,
                    "std": float(df[col].std()),
                    "skewness": float(df[col].skew()),
                    "kurtosis": float(df[col].kurtosis()),
                    "range": float(df[col].max() - df[col].min()),
                    "q1": float(df[col].quantile(0.25)),
                    "q3": float(df[col].quantile(0.75)),
                    "iqr": float(df[col].quantile(0.75) - df[col].quantile(0.25)),
                }
                
                # Calculate bin information for more detailed stats
                hist, bin_edges = np.histogram(df[col].dropna(), bins='auto')
                bin_info = [
                    {"bin_start": float(bin_edges[i]), 
                     "bin_end": float(bin_edges[i+1]), 
                     "count": int(hist[i]),
                     "percentage": float(hist[i]/len(df)*100)} 
                    for i in range(len(hist))
                ]
                
                results[col] = {
                    "statistics": col_stats,
                    "bin_information": bin_info
                }
                
                def plot_frequency_distribution():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    sns.histplot(df[col], kde=True, ax=ax)
                    ax.set_title(f'Frequency Distribution of {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
                    return fig, ax

                result = self.generate_plot(plot_frequency_distribution)
                if result is not None:
                    fig, ax = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_{col}_frequency_distribution.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append((f"Frequency Distribution of {col}", img_path))
                else:
                    print(f"Skipping frequency distribution plot for {col} due to timeout.")
        else:
            results["note"] = "No numerical columns found for frequency distribution analysis"
        
        results['image_paths'] = image_paths
        self.interpret_results("Frequency Distribution Analysis", results, table_name)

    def kde_plot_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - KDE Plot Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        results = {}
        
        if len(numerical_columns) > 0:
            # Include density estimation stats for each column
            for col in numerical_columns:
                # Calculate KDE statistics (we'll use basic stats since actual KDE values are complex)
                kde_stats = {
                    "peak_around": float(df[col].median()),  # Approximation of peak
                    "distribution_width": float(df[col].std() * 2),  # Approximation of distribution width
                    "significant_range": [
                        float(df[col].mean() - 2*df[col].std()),
                        float(df[col].mean() + 2*df[col].std())
                    ]
                }
                
                results[col] = kde_stats
                
                def plot_kde():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    sns.kdeplot(df[col], shade=True, ax=ax)
                    ax.set_title(f'KDE Plot for {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Density')
                    return fig, ax

                result = self.generate_plot(plot_kde)
                if result is not None:
                    fig, ax = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_{col}_kde_plot.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append((f"KDE Plot for {col}", img_path))
                else:
                    print(f"Skipping KDE plot for {col} due to timeout.")
        else:
            results["note"] = "No numerical columns found for KDE plot analysis"
        
        results['image_paths'] = image_paths
        self.interpret_results("KDE Plot Analysis", results, table_name)

    def violin_plot_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Violin Plot Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        image_paths = []
        results = {}
        
        if len(categorical_columns) > 0 and len(numerical_columns) > 0:
            # Use the first categorical column for grouping
            categorical_col = categorical_columns[0]
            
            # Get actual distribution statistics per group
            for num_col in numerical_columns:
                group_stats = []
                for group_name, group_data in df.groupby(categorical_col):
                    if len(group_data) > 0:
                        group_stats.append({
                            "group_name": str(group_name),
                            "count": int(len(group_data)),
                            "mean": float(group_data[num_col].mean()),
                            "median": float(group_data[num_col].median()),
                            "q1": float(group_data[num_col].quantile(0.25)),
                            "q3": float(group_data[num_col].quantile(0.75)),
                            "min": float(group_data[num_col].min()),
                            "max": float(group_data[num_col].max()),
                            "skewness": float(group_data[num_col].skew())
                        })
                
                results[f"{categorical_col} - {num_col}"] = {
                    "group_statistics": group_stats,
                    "total_groups": len(group_stats)
                }
                
                def plot_violin():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    sns.violinplot(x=categorical_col, y=num_col, data=df, ax=ax)
                    ax.set_title(f'Violin Plot of {num_col} grouped by {categorical_col}')
                    ax.set_xlabel(categorical_col)
                    ax.set_ylabel(num_col)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_violin)
                if result is not None:
                    fig, ax = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_{num_col}_violin_plot.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append((f"Violin Plot: {num_col} by {categorical_col}", img_path))
                else:
                    print(f"Skipping violin plot for {num_col} due to timeout.")
        else:
            if len(categorical_columns) == 0:
                results["note"] = "No categorical columns found for violin plot analysis"
            else:
                results["note"] = "No numerical columns found for violin plot analysis"
        
        results['image_paths'] = image_paths
        self.interpret_results("Violin Plot Analysis", results, table_name)

    def pair_plot_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Pair Plot Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        results = {}
        
        if len(numerical_columns) > 1:
            # Get pairwise correlation statistics for context
            correlation_matrix = df[numerical_columns].corr()
            
            # Extract meaningful pairwise relationships
            strong_pairs = []
            for i in range(len(numerical_columns)):
                for j in range(i+1, len(numerical_columns)):
                    col1 = numerical_columns[i]
                    col2 = numerical_columns[j]
                    corr = correlation_matrix.loc[col1, col2]
                    
                    if abs(corr) > 0.5:  # Only include significant correlations
                        strong_pairs.append({
                            "variable_1": str(col1),
                            "variable_2": str(col2),
                            "correlation": float(corr),
                            "relationship": "positive" if corr > 0 else "negative",
                            "strength": "strong" if abs(corr) > 0.7 else "moderate"
                        })
            
            results = {
                "pairwise_relationships": strong_pairs,
                "num_strong_relationships": len(strong_pairs),
                "variables_analyzed": list(numerical_columns)
            }
            
            # Limit to first 8 numerical columns to avoid timeout
            subset_cols = numerical_columns[:min(8, len(numerical_columns))]
            
            def plot_pair():
                fig = sns.pairplot(df[subset_cols], height=3, aspect=1.2)
                fig.fig.suptitle("Pair Plot of Numerical Variables", y=1.02)
                return fig

            result = self.generate_plot(plot_pair)
            if result is not None:
                img_path = os.path.join(self.output_folder, f"{table_name}_pair_plot.png")
                result.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(result.fig)
                image_paths.append(("Pair Plot Analysis", img_path))
            else:
                print("Skipping pair plot due to timeout.")
        else:
            results["note"] = "Not enough numerical columns for pair plot analysis"
        
        results['image_paths'] = image_paths
        self.interpret_results("Pair Plot Analysis", results, table_name)

    def box_plot_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Box Plot Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        results = {}
        
        if len(numerical_columns) > 0:
            # Get distribution statistics for context
            for col in numerical_columns:
                # Calculate box plot statistics
                q1 = df[col].quantile(0.25)
                median = df[col].median()
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Get list of outliers
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                
                results[col] = {
                    "quartiles": {
                        "q1": float(q1),
                        "median": float(median),
                        "q3": float(q3)
                    },
                    "iqr": float(iqr),
                    "whiskers": {
                        "lower": float(max(lower_bound, df[col].min())),
                        "upper": float(min(upper_bound, df[col].max()))
                    },
                    "outliers": {
                        "count": int(len(outliers)),
                        "percentage": float(len(outliers) / len(df) * 100),
                        "min": float(outliers.min()) if len(outliers) > 0 else None,
                        "max": float(outliers.max()) if len(outliers) > 0 else None
                    }
                }
                
                def plot_box():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    sns.boxplot(y=df[col], ax=ax)
                    ax.set_title(f'Box Plot of {col}')
                    ax.set_ylabel(col)
                    return fig, ax

                result = self.generate_plot(plot_box)
                if result is not None:
                    fig, ax = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_{col}_box_plot.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append((f"Box Plot of {col}", img_path))
                else:
                    print(f"Skipping box plot for {col} due to timeout.")
        else:
            results["note"] = "No numerical columns found for box plot analysis"
        
        results['image_paths'] = image_paths
        self.interpret_results("Box Plot Analysis", results, table_name)

    def scatter_plot_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Scatter Plot Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        results = {}
        
        if len(numerical_columns) > 1:
            # Get pairwise correlation statistics for context
            significant_relationships = []
            
            for i in range(len(numerical_columns)):
                for j in range(i+1, len(numerical_columns)):
                    col1, col2 = numerical_columns[i], numerical_columns[j]
                    
                    # Calculate correlation
                    corr, p_value = pearsonr(df[col1].fillna(df[col1].mean()), df[col2].fillna(df[col2].mean()))
                    
                    relationship = {
                        "variable_1": str(col1),
                        "variable_2": str(col2),
                        "correlation": float(corr),
                        "p_value": float(p_value),
                        "significant": bool(p_value < 0.05),
                        "relationship_type": "positive" if corr > 0 else "negative",
                        "strength": "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.5 else "weak"
                    }
                    
                    significant_relationships.append(relationship)
                    
                    # Create scatter plot (only for strong/moderate relationships to save time)
                    if abs(corr) > 0.5:
                        def plot_scatter():
                            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                            sns.scatterplot(x=df[col1], y=df[col2], ax=ax)
                            
                            # Add regression line
                            sns.regplot(x=df[col1], y=df[col2], scatter=False, ax=ax, line_kws={"color": "red"})
                            
                            ax.set_title(f'Scatter Plot of {col1} vs {col2} (r={corr:.2f})')
                            ax.set_xlabel(col1)
                            ax.set_ylabel(col2)
                            return fig, ax

                        result = self.generate_plot(plot_scatter)
                        if result is not None:
                            fig, ax = result
                            img_path = os.path.join(self.output_folder, f"{table_name}_{col1}_vs_{col2}_scatter_plot.png")
                            plt.savefig(img_path, dpi=100, bbox_inches='tight')
                            plt.close(fig)
                            image_paths.append((f"Scatter: {col1} vs {col2}", img_path))
            
            # Sort by correlation strength
            significant_relationships.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            results = {
                "relationships": significant_relationships,
                "strongest_positive": significant_relationships[0] if significant_relationships and significant_relationships[0]["correlation"] > 0 else None,
                "strongest_negative": next((r for r in significant_relationships if r["correlation"] < 0), None)
            }
        else:
            results["note"] = "Not enough numerical columns for scatter plot analysis"
        
        results['image_paths'] = image_paths
        self.interpret_results("Scatter Plot Analysis", results, table_name)

    def time_series_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Time Series Analysis"))
        
        # Try to identify date columns
        date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # If no datetime columns found, try to convert string columns to datetime
        if not date_columns:
            for col in df.select_dtypes(include=['object']):
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_columns.append(col)
                except ValueError:
                    continue
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        results = {}
        
        if date_columns and len(numerical_columns) > 0:
            date_col = date_columns[0]  # Use the first identified date column
            df_sorted = df.sort_values(date_col)  # Sort by date
            
            # Extract time series insights for context
            time_series_data = {
                "date_column": str(date_col),
                "time_range": {
                    "start": str(df_sorted[date_col].min()),
                    "end": str(df_sorted[date_col].max()),
                },
                "total_periods": len(df_sorted),
                "series_analyzed": []
            }
            
            for num_col in numerical_columns:
                # Calculate time series metrics
                series_data = df_sorted[num_col]
                
                # Calculate basic trend (simple approximation)
                first_value = series_data.iloc[0]
                last_value = series_data.iloc[-1]
                
                series_analysis = {
                    "variable": str(num_col),
                    "trend_direction": "increasing" if last_value > first_value else "decreasing" if last_value < first_value else "stable",
                    "change_percentage": float((last_value - first_value) / first_value * 100) if first_value != 0 else None,
                    "min": {
                        "value": float(series_data.min()),
                        "date": str(df_sorted.loc[series_data.idxmin(), date_col])
                    },
                    "max": {
                        "value": float(series_data.max()),
                        "date": str(df_sorted.loc[series_data.idxmax(), date_col])
                    },
                    "volatility": float(series_data.std() / series_data.mean() * 100) if series_data.mean() != 0 else None
                }
                
                time_series_data["series_analyzed"].append(series_analysis)
                
                def plot_time_series():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    sns.lineplot(x=df_sorted[date_col], y=df_sorted[num_col], ax=ax)
                    ax.set_title(f'Time Series Plot of {num_col}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel(num_col)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_time_series)
                if result is not None:
                    fig, ax = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_{num_col}_time_series_plot.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append((f"Time Series: {num_col}", img_path))
                else:
                    print(f"Skipping time series plot for {num_col} due to timeout.")
                    
            results = time_series_data
        else:
            if not date_columns:
                results["note"] = "No suitable date columns found for time series analysis"
            else:
                results["note"] = "No numerical columns found for time series analysis"
        
        results['image_paths'] = image_paths
        self.interpret_results("Time Series Analysis", results, table_name)

    def outlier_detection(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Outlier Detection"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        results = {}
        image_paths = []
        
        if len(numerical_columns) > 0:
            outlier_summary = {
                "variables_analyzed": numerical_columns.tolist(),
                "method": "IQR (1.5 rule)",
                "variable_results": {}
            }
            
            for col in numerical_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                # Get actual outlier values for context
                high_outliers = df[df[col] > upper_bound][col].nlargest(5).tolist() if len(df[df[col] > upper_bound]) > 0 else []
                low_outliers = df[df[col] < lower_bound][col].nsmallest(5).tolist() if len(df[df[col] < lower_bound]) > 0 else []
                
                outlier_summary["variable_results"][col] = {
                    "total_outliers": len(outliers),
                    "outlier_percentage": float(len(outliers) / len(df) * 100),
                    "bounds": {
                        "lower": float(lower_bound),
                        "upper": float(upper_bound)
                    },
                    "examples": {
                        "high_outliers": [float(val) for val in high_outliers[:5]],
                        "low_outliers": [float(val) for val in low_outliers[:5]]
                    }
                }
                
                # Create combined box/scatter plot to highlight outliers
                def plot_outliers():
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.calculate_figure_size()[0]*2, self.calculate_figure_size()[1]))
                    
                    # Box plot
                    sns.boxplot(y=df[col], ax=ax1)
                    ax1.set_title(f'Box Plot with Outliers: {col}')
                    ax1.set_ylabel(col)
                    
                    # Scatter plot with outliers highlighted
                    indices = range(len(df))
                    is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
                    
                    # Plot non-outliers
                    ax2.scatter(indices, df[col], c='blue', alpha=0.5, s=10, label='Normal')
                    
                    # Plot outliers in red
                    if is_outlier.any():
                        outlier_indices = [i for i, o in enumerate(is_outlier) if o]
                        ax2.scatter(outlier_indices, df.iloc[outlier_indices][col], c='red', s=20, label='Outlier')
                    
                    ax2.set_title(f'Outliers in {col}')
                    ax2.set_xlabel('Index')
                    ax2.set_ylabel(col)
                    ax2.legend()
                    
                    plt.tight_layout()
                    return fig, (ax1, ax2)

                result = self.generate_plot(plot_outliers)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_{col}_outliers.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append((f"Outliers: {col}", img_path))
            
            results = outlier_summary
        else:
            results["note"] = "No numerical columns found for outlier detection"
        
        results['image_paths'] = image_paths
        self.interpret_results("Outlier Detection", results, table_name)

    def feature_importance_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Feature Importance Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        results = {}
        image_paths = []
        
        if len(numerical_columns) > 1:
            # Choose the last column as the target variable
            target_variable = numerical_columns[-1]
            feature_variables = numerical_columns[:-1]
            
            X = df[feature_variables].fillna(df[feature_variables].mean())  # Handle NaN values
            y = df[target_variable].fillna(df[target_variable].mean())      # Handle NaN values
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Get feature importance with actual feature names
            feature_importance = []
            for i, feature in enumerate(feature_variables):
                feature_importance.append({
                    "feature": str(feature),
                    "importance_score": float(model.feature_importances_[i]),
                    "importance_percentage": float(model.feature_importances_[i] * 100)
                })
            
            # Sort by importance score
            feature_importance.sort(key=lambda x: x["importance_score"], reverse=True)
            
            results = {
                "target_variable": str(target_variable),
                "feature_count": len(feature_variables),
                "feature_importance": feature_importance,
                "top_features": [item["feature"] for item in feature_importance[:3]],
                "model_type": "Random Forest Regressor"
            }
            
            def plot_feature_importance():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.calculate_figure_size()[0]*2, self.calculate_figure_size()[1]))
                
                # Create DataFrame for plotting
                plot_df = pd.DataFrame({
                    'feature': feature_variables,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Bar plot
                sns.barplot(x='importance', y='feature', data=plot_df, ax=ax1)
                ax1.set_title(f'Feature Importance for {target_variable}')
                ax1.set_xlabel('Importance')
                ax1.set_ylabel('Feature')
                
                # Pie chart
                ax2.pie(plot_df['importance'], labels=plot_df['feature'], autopct='%1.1f%%')
                ax2.set_title('Feature Importance (Pie Chart)')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_feature_importance)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_feature_importance.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Feature Importance", img_path))
        else:
            results["note"] = "Not enough numerical columns for feature importance analysis"
        
        results['image_paths'] = image_paths
        self.interpret_results("Feature Importance Analysis", results, table_name)

    def pca_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - PCA Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        results = {}
        image_paths = []
        
        if len(numerical_columns) > 2:
            # Prepare data
            X = df[numerical_columns].fillna(df[numerical_columns].mean())  # Handle NaN values
            X_scaled = StandardScaler().fit_transform(X)
            
            # Perform PCA
            pca = PCA()
            pca_result = pca.fit_transform(X_scaled)
            
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            
            # Calculate number of components for different variance thresholds
            components_for_70pct = next((i+1 for i, c in enumerate(cumulative_variance_ratio) if c >= 0.7), len(cumulative_variance_ratio))
            components_for_80pct = next((i+1 for i, c in enumerate(cumulative_variance_ratio) if c >= 0.8), len(cumulative_variance_ratio))
            components_for_90pct = next((i+1 for i, c in enumerate(cumulative_variance_ratio) if c >= 0.9), len(cumulative_variance_ratio))
            
            # Get component loadings with actual feature names
            loadings = []
            for i, component in enumerate(pca.components_[:3]):  # Get first 3 components
                component_loading = {}
                for j, feature in enumerate(numerical_columns):
                    component_loading[str(feature)] = float(component[j])
                loadings.append(component_loading)
            
            results = {
                "variables_analyzed": numerical_columns.tolist(),
                "total_components": len(explained_variance_ratio),
                "explained_variance": [float(v) for v in explained_variance_ratio],
                "cumulative_variance": [float(v) for v in cumulative_variance_ratio],
                "dimensionality_reduction": {
                    "original_dimensions": len(numerical_columns),
                    "components_for_70pct": components_for_70pct,
                    "components_for_80pct": components_for_80pct,
                    "components_for_90pct": components_for_90pct
                },
                "principal_components": {
                    "PC1_loadings": loadings[0] if len(loadings) > 0 else None,
                    "PC2_loadings": loadings[1] if len(loadings) > 1 else None,
                    "PC3_loadings": loadings[2] if len(loadings) > 2 else None
                }
            }
            
            def plot_pca():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.calculate_figure_size()[0]*2, self.calculate_figure_size()[1]))
                
                # Scree plot
                ax1.plot(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, 'bo-')
                ax1.set_xlabel('Principal Component')
                ax1.set_ylabel('Explained Variance Ratio')
                ax1.set_title('Scree Plot')
                
                # Cumulative explained variance plot
                ax2.plot(range(1, len(cumulative_variance_ratio)+1), cumulative_variance_ratio, 'ro-')
                ax2.set_xlabel('Number of Components')
                ax2.set_ylabel('Cumulative Explained Variance Ratio')
                ax2.set_title('Cumulative Explained Variance')
                
                # Add reference lines at 70%, 80%, and 90%
                for threshold in [0.7, 0.8, 0.9]:
                    ax2.axhline(y=threshold, color='gray', linestyle='--')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_pca)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_pca_analysis.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("PCA Analysis", img_path))
                
                # If we have at least 2 components, also create the 2D scatter plot
                if pca_result.shape[1] >= 2:
                    def plot_pca_scatter():
                        fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                        ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
                        ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.2%} variance)')
                        ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.2%} variance)')
                        ax.set_title('2D PCA Projection')
                        return fig, ax
                        
                    result = self.generate_plot(plot_pca_scatter)
                    if result is not None:
                        fig, ax = result
                        img_path = os.path.join(self.output_folder, f"{table_name}_pca_scatter.png")
                        plt.savefig(img_path, dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        image_paths.append(("PCA Scatter Plot", img_path))
        else:
            results["note"] = "Not enough numerical columns for PCA analysis (need at least 3)"
        
        results['image_paths'] = image_paths
        self.interpret_results("PCA Analysis", results, table_name)

    def cluster_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Cluster Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        results = {}
        image_paths = []
        
        if len(numerical_columns) > 1:
            # Prepare data
            X = df[numerical_columns].fillna(df[numerical_columns].mean())  # Handle NaN values
            X_scaled = StandardScaler().fit_transform(X)
            
            # Determine optimal number of clusters using elbow method
            inertias = []
            max_clusters = min(10, X_scaled.shape[0] - 1)  # Limit to 10 clusters or one less than number of samples
            for k in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
            
            # Find elbow point
            elbow = next(i for i in range(1, len(inertias)) if inertias[i-1] - inertias[i] < (inertias[0] - inertias[-1]) / 10)
            
            # Perform K-means clustering with optimal number of clusters
            kmeans = KMeans(n_clusters=elbow, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to the data
            df_with_clusters = df.copy()
            df_with_clusters['cluster'] = cluster_labels
            
            # Generate cluster profiles with actual values
            cluster_profiles = []
            for cluster_id in range(elbow):
                cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
                
                profile = {
                    "cluster_id": cluster_id,
                    "size": len(cluster_data),
                    "percentage": float(len(cluster_data) / len(df) * 100),
                    "characteristics": {}
                }
                
                # Calculate characteristics for each feature
                for col in numerical_columns:
                    cluster_mean = cluster_data[col].mean()
                    overall_mean = df[col].mean()
                    
                    # Calculate how this cluster differs from the overall average
                    difference = cluster_mean - overall_mean
                    difference_pct = (difference / overall_mean * 100) if overall_mean != 0 else float('inf')
                    
                    profile["characteristics"][str(col)] = {
                        "cluster_mean": float(cluster_mean),
                        "overall_mean": float(overall_mean),
                        "difference": float(difference),
                        "difference_percentage": float(difference_pct),
                        "direction": "above average" if difference > 0 else "below average" if difference < 0 else "average"
                    }
                
                cluster_profiles.append(profile)
            
            results = {
                "optimal_clusters": elbow,
                "elbow_method_inertias": [float(i) for i in inertias],
                "cluster_profiles": cluster_profiles,
                "variables_used": numerical_columns.tolist()
            }
            
            def plot_clusters():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.calculate_figure_size()[0]*2, self.calculate_figure_size()[1]))
                
                # Elbow plot
                ax1.plot(range(1, max_clusters + 1), inertias, 'bo-')
                ax1.set_xlabel('Number of Clusters (k)')
                ax1.set_ylabel('Inertia')
                ax1.set_title('Elbow Method for Optimal k')
                ax1.axvline(x=elbow, color='r', linestyle='--')
                
                # 2D projection of clusters
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
                ax2.set_xlabel('First Principal Component')
                ax2.set_ylabel('Second Principal Component')
                ax2.set_title(f'2D PCA Projection of {elbow} Clusters')
                
                # Add legend
                legend1 = ax2.legend(*scatter.legend_elements(), title="Clusters")
                ax2.add_artist(legend1)
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_clusters)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_cluster_analysis.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Cluster Analysis", img_path))
        else:
            results["note"] = "Not enough numerical columns for cluster analysis"
        
        results['image_paths'] = image_paths
        self.interpret_results("Cluster Analysis", results, table_name)

    def correlation_network_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Correlation Network Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        results = {}
        image_paths = []
        
        if len(numerical_columns) > 1:
            # Calculate correlation matrix
            X = df[numerical_columns].fillna(df[numerical_columns].mean())  # Handle NaN values
            corr_matrix = X.corr()
            
            # Extract significant correlations for context
            significant_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    
                    if abs(corr_value) > 0.5:  # Only include correlations > 0.5
                        significant_correlations.append({
                            "variable_1": str(col1),
                            "variable_2": str(col2),
                            "correlation": float(corr_value),
                            "strength": "strong" if abs(corr_value) > 0.7 else "moderate",
                            "type": "positive" if corr_value > 0 else "negative"
                        })
            
            # Sort by absolute correlation strength
            significant_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            # Create a graph from the correlation matrix
            G = nx.Graph()
            for corr in significant_correlations:
                G.add_edge(
                    corr["variable_1"], 
                    corr["variable_2"], 
                    weight=abs(corr["correlation"]),
                    color="blue" if corr["correlation"] > 0 else "red"
                )
            
            # Calculate network metrics
            if len(G.nodes) > 0:
                betweenness = nx.betweenness_centrality(G)
                degree = dict(G.degree())
                
                # Find central variables
                central_variables = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
                hub_variables = sorted(degree.items(), key=lambda x: x[1], reverse=True)
                
                results = {
                    "variables_analyzed": numerical_columns.tolist(),
                    "significant_correlations": significant_correlations,
                    "network_stats": {
                        "total_nodes": len(G.nodes),
                        "total_edges": len(G.edges),
                        "network_density": nx.density(G),
                        "average_degree": sum(dict(G.degree()).values()) / len(G.nodes) if len(G.nodes) > 0 else 0
                    },
                    "central_variables": [{"variable": var, "centrality": float(cent)} for var, cent in central_variables[:3]],
                    "hub_variables": [{"variable": var, "connections": deg} for var, deg in hub_variables[:3]]
                }
                
                def plot_correlation_network():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    
                    # Get position and colors for nodes and edges
                    pos = nx.spring_layout(G, seed=42)
                    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
                    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]  # Scale up for visibility
                    
                    # Draw nodes with size based on betweenness centrality
                    node_sizes = [betweenness[node] * 5000 + 500 for node in G.nodes()]
                    
                    # Draw the network
                    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_sizes, ax=ax)
                    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_weights, alpha=0.7, ax=ax)
                    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
                    
                    # Add a title
                    ax.set_title('Correlation Network (|r| > 0.5)')
                    ax.axis('off')  # Turn off the axis
                    
                    # Add a legend
                    blue_line = plt.Line2D([0], [0], color='blue', lw=2)
                    red_line = plt.Line2D([0], [0], color='red', lw=2)
                    ax.legend([blue_line, red_line], ['Positive Correlation', 'Negative Correlation'], loc='lower right')
                    
                    return fig, ax

                result = self.generate_plot(plot_correlation_network)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_correlation_network.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(("Correlation Network", img_path))
            else:
                results["note"] = "No significant correlations found to create a network"
        else:
            results["note"] = "Not enough numerical columns for correlation network analysis"
        
        results['image_paths'] = image_paths
        self.interpret_results("Correlation Network Analysis", results, table_name)

    def qq_plot_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Q-Q Plot Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        results = {}
        image_paths = []
        
        if len(numerical_columns) > 0:
            normality_results = {}
            
            for col in numerical_columns:
                # Get data
                col_data = df[col].dropna()
                
                # Run Anderson-Darling test for normality
                try:
                    anderson_result = anderson(col_data)
                    
                    # Determine if normally distributed (using 5% significance level)
                    is_normal = anderson_result.statistic < anderson_result.critical_values[2]
                    
                    normality_results[str(col)] = {
                        "is_normal": bool(is_normal),
                        "anderson_statistic": float(anderson_result.statistic),
                        "critical_value_5pct": float(anderson_result.critical_values[2]),
                        "skewness": float(col_data.skew()),
                        "kurtosis": float(col_data.kurtosis()),
                        "normality_assessment": "normal" if is_normal else "non-normal",
                        "skew_type": "right-skewed" if col_data.skew() > 0.5 else "left-skewed" if col_data.skew() < -0.5 else "approximately symmetric"
                    }
                except:
                    normality_results[str(col)] = {
                        "is_normal": None,
                        "normality_assessment": "could not be determined",
                        "error": "Anderson-Darling test failed, possibly due to insufficient data or constant values"
                    }
                
                def plot_qq():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    
                    # Create Q-Q plot
                    res = probplot(col_data, dist="norm", plot=ax)
                    
                    # Add more context to the plot
                    ax.set_title(f'Q-Q Plot of {col} ({"Normal" if normality_results[str(col)].get("is_normal", False) else "Non-normal"} Distribution)')
                    ax.set_xlabel('Theoretical Quantiles')
                    ax.set_ylabel('Sample Quantiles')
                    
                    # Add skewness and kurtosis information
                    skew = col_data.skew()
                    kurt = col_data.kurtosis()
                    plt.figtext(0.15, 0.15, f'Skewness: {skew:.3f}\nKurtosis: {kurt:.3f}', 
                                bbox=dict(facecolor='white', alpha=0.8))
                    
                    return fig, ax

                result = self.generate_plot(plot_qq)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_{col}_qq_plot.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append((f"Q-Q Plot: {col}", img_path))
                else:
                    print(f"Skipping Q-Q plot for {col} due to timeout.")
            
            results = {
                "variables_analyzed": numerical_columns.tolist(),
                "normality_results": normality_results,
                "normal_variables": [col for col, res in normality_results.items() if res.get("is_normal", False)],
                "non_normal_variables": [col for col, res in normality_results.items() if res.get("is_normal", False) is False]
            }
        else:
            results["note"] = "No numerical columns found for Q-Q plot analysis"
        
        results['image_paths'] = image_paths
        self.interpret_results("Q-Q Plot Analysis", results, table_name)

    def save_results(self, analysis_type, results):
        if not self.settings.save_results_to_txt:
            return  # Skip saving if the option is disabled

        results_file = os.path.join(self.output_folder, f"{analysis_type.lower().replace(' ', '_')}_results.txt")
        with open(results_file, "w", encoding='utf-8') as f:
            f.write(f"Results for {analysis_type}:\n")
            if isinstance(results, dict):
                for key, value in results.items():
                    if key != 'image_paths':
                        f.write(f"{key}: {value}\n")
            else:
                f.write(str(results))
        print(success(f"Results saved as txt file: {results_file}"))

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

        # Save the results
        self.save_results(analysis_type, results)

        common_prompt = f"""
        Analysis type: {analysis_type}
        Table name: {table_name}
        Database description: {self.database_description}

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
        
        Use actual names and values from the data instead of generic references. For example, say "Team George" instead of "Team Alpha", "Product XYZ" instead of "Product Category 1", etc.

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
        
        Use actual names and values from the data instead of generic references. For example, use "Team George" instead of "Team Alpha", "Product XYZ" instead of "Product Category 1", etc.
        
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
        output_file = os.path.join(self.output_folder, "axda_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        report_title = f"Advanced Exploratory Data Analysis (Batch 1) Report for {self.table_name}"
        
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
            filename=f"axda_b1_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
            return pdf_file
        else:
            print(error("Failed to generate PDF report"))
            return None