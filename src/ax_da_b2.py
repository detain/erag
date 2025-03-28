import os
import time
import sqlite3
import threading
import signal
import sys
from functools import wraps

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import folium
import joypy
import shap

from scipy import stats
from scipy.stats import norm, anderson, pearsonr, probplot
from scipy.cluster.hierarchy import dendrogram

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.impute import SimpleImputer

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.mosaicplot import mosaic

from wordcloud import WordCloud

from src.api_model import EragAPI
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight
from src.print_pdf import PDFReportGenerator
from src.helper_da import get_technique_info

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')

class TimeoutException(Exception):
    pass

class AdvancedExploratoryDataAnalysisB2:
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
        print(info("Please provide a description of the database for advanced analysis (Batch 2). This will help the AI models provide better insights."))
        print(info("Describe the purpose, main tables, key data points, and any other relevant information:"))
        self.database_description = input("> ")
        print(success(f"Database description received: {self.database_description}"))

    def run(self):
        self.prompt_for_database_description()
        print(info(f"Starting Advanced Exploratory Data Analysis (Batch2) on {self.db_path}"))
        
        tables = self.get_tables()
        for table in tables:
            self.analyze_table(table)
        
        self.save_text_output()
        self.generate_pdf_report()
        print(success(f"Advanced Exploratory Data Analysis (Batch2) completed. Results saved in {self.output_folder}"))

    def get_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]

    def analyze_table(self, table_name):
        self.table_name = table_name
        self.output_folder = os.path.join(settings.output_folder, f"axda_b2_{self.table_name}")
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, self.table_name)
        
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            print(info(f"Loaded full dataset with {len(df)} rows and {len(df.columns)} columns"))

        analysis_methods = [
            self.parallel_coordinates_plot,
            self.andrews_curves,
            self.radar_charts,
            self.sankey_diagrams,
            self.bubble_charts,
            self.geographical_plots,
            self.word_clouds,
            self.hierarchical_clustering_dendrogram,
            self.ecdf_plots,
            self.ridgeline_plots,
            self.hexbin_plots,
            self.mosaic_plots,
            self.lag_plots,
            self.shapley_value_analysis,
            self.partial_dependence_plots
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

    def parallel_coordinates_plot(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Parallel Coordinates Plot"))
        image_paths = []
        results = {}
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 1:
            # Extract actual column names and data info for context
            results["analysis_info"] = {
                "numerical_columns_used": numerical_columns.tolist(),
                "total_numerical_columns": len(numerical_columns),
                "sample_size": min(1000, len(df)) # We limit to 1000 rows for visualization
            }
            
            def plot_parallel_coordinates():
                try:
                    # Limit the number of columns and rows to prevent extremely large plots
                    columns_to_plot = numerical_columns[:10]  # Plot at most 10 columns
                    df_plot = df[columns_to_plot].head(1000)  # Limit to 1000 rows
                    
                    # If there's no 'target' column, use the first column as a proxy
                    target_column = 'target' if 'target' in df_plot.columns else df_plot.columns[0]
                    
                    # Calculate figure size based on the number of columns
                    width, height = self.calculate_figure_size()
                    fig, ax = plt.subplots(figsize=(width * len(columns_to_plot) / 10, height))
                    
                    pd.plotting.parallel_coordinates(df_plot, target_column, ax=ax)
                    ax.set_title('Parallel Coordinates Plot (Sample)')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    return fig, ax
                except Exception as e:
                    print(f"Error in creating parallel coordinates plot: {str(e)}")
                    return None

            result = self.generate_plot(plot_parallel_coordinates)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_parallel_coordinates.png")
                try:
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(("Parallel Coordinates Plot", img_path))
                except Exception as e:
                    print(f"Error saving parallel coordinates plot: {str(e)}")
                    print("Skipping parallel coordinates plot due to error.")
            else:
                print("Skipping parallel coordinates plot due to error in plot generation.")
                results["error"] = "Failed to generate parallel coordinates plot"
        else:
            results["error"] = "Not enough numerical columns for parallel coordinates plot"
            print("Not enough numerical columns for parallel coordinates plot.")

        results['image_paths'] = image_paths
        self.interpret_results("Parallel Coordinates Plot", results, table_name)

    def andrews_curves(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Andrews Curves"))
        image_paths = []
        results = {}
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 1:
            # Extract context information
            results["analysis_info"] = {
                "numerical_columns_used": numerical_columns.tolist(),
                "total_numerical_columns": len(numerical_columns),
                "sample_size": min(1000, len(df))
            }
            
            def plot_andrews_curves():
                try:
                    # Limit the number of columns and rows to prevent extremely large plots
                    columns_to_plot = numerical_columns[:10]  # Plot at most 10 columns
                    df_plot = df[columns_to_plot].head(1000)  # Limit to 1000 rows
                    
                    # If there's no 'target' column, use the first column as a proxy
                    target_column = 'target' if 'target' in df_plot.columns else df_plot.columns[0]
                    
                    # Calculate figure size
                    width, height = self.calculate_figure_size()
                    fig, ax = plt.subplots(figsize=(width, height))
                    
                    pd.plotting.andrews_curves(df_plot, target_column, ax=ax)
                    ax.set_title('Andrews Curves (Sample)')
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.tight_layout()
                    return fig, ax
                except Exception as e:
                    print(f"Error in creating Andrews curves plot: {str(e)}")
                    return None

            result = self.generate_plot(plot_andrews_curves)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_andrews_curves.png")
                try:
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(("Andrews Curves", img_path))
                except Exception as e:
                    print(f"Error saving Andrews curves plot: {str(e)}")
                    results["error"] = f"Failed to save Andrews curves plot: {str(e)}"
            else:
                print("Skipping Andrews curves plot due to error in plot generation.")
                results["error"] = "Failed to generate Andrews curves plot"
        else:
            results["error"] = "Not enough numerical columns for Andrews curves"
            print("Not enough numerical columns for Andrews curves.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Andrews Curves", results, table_name)

    def radar_charts(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Radar Charts"))
        image_paths = []
        results = {}
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 2:
            # Extract column means for context
            column_means = {}
            for col in numerical_columns:
                column_means[str(col)] = float(df[col].mean())
            
            results["analysis_info"] = {
                "numerical_columns_used": numerical_columns.tolist(),
                "total_numerical_columns": len(numerical_columns),
                "column_means": column_means,
                "data_shape": df.shape
            }
            
            def plot_radar_chart():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size(), subplot_kw=dict(projection='polar'))
                values = df[numerical_columns].mean().values
                angles = np.linspace(0, 2*np.pi, len(numerical_columns), endpoint=False)
                values = np.concatenate((values, [values[0]]))
                angles = np.concatenate((angles, [angles[0]]))
                ax.plot(angles, values)
                ax.fill(angles, values, alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(numerical_columns)
                ax.set_title('Radar Chart of Average Values')
                return fig, ax

            result = self.generate_plot(plot_radar_chart)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_radar_chart.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Radar Chart", img_path))
            else:
                print("Skipping radar chart plot due to timeout.")
                results["error"] = "Failed to generate radar chart due to timeout"
        else:
            results["error"] = "Not enough numerical columns for radar chart"
            print("Not enough numerical columns for radar chart.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Radar Charts", results, table_name)

    def sankey_diagrams(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Sankey Diagrams"))
        image_paths = []
        results = {}

        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) >= 2:
            # Extract actual category names for context
            source_column = categorical_columns[0]
            target_column = categorical_columns[1]
            
            # Get unique categories and their counts
            source_categories = df[source_column].value_counts().to_dict()
            target_categories = df[target_column].value_counts().to_dict()
            
            # Get flow information (how many items flow from each source to each target)
            flow_counts = df.groupby([source_column, target_column]).size()
            
            # Format the flow data for better context
            flows = []
            for (source, target), count in flow_counts.items():
                flows.append({
                    "source": str(source),
                    "target": str(target),
                    "count": int(count),
                    "percentage": float(count / len(df) * 100)
                })
            
            results["analysis_info"] = {
                "source_column": str(source_column),
                "target_column": str(target_column),
                "source_categories": {str(k): int(v) for k, v in source_categories.items()},
                "target_categories": {str(k): int(v) for k, v in target_categories.items()},
                "significant_flows": sorted(flows, key=lambda x: x["count"], reverse=True)[:10]  # Top 10 flows
            }
            
            def plot_sankey():
                source = df[categorical_columns[0]]
                target = df[categorical_columns[1]]
                value = df[df.columns[0]]  # Using the first column as value

                label = list(set(source) | set(target))
                color = plt.cm.Set3(np.linspace(0, 1, len(label)))

                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                
                sankey = pd.DataFrame({'source': source, 'target': target, 'value': value})
                G = nx.from_pandas_edgelist(sankey, 'source', 'target', 'value')
                pos = nx.spring_layout(G)
                
                nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=color)
                nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, alpha=0.5)
                nx.draw_networkx_labels(G, pos, font_size=10)
                
                ax.set_title(f'Flow from {source_column} to {target_column}')
                ax.axis('off')
                return fig, ax

            result = self.generate_plot(plot_sankey)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_sankey_diagram.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Sankey Diagram", img_path))
            else:
                print("Skipping Sankey diagram plot due to timeout.")
                results["error"] = "Failed to generate Sankey diagram due to timeout"
        else:
            results["error"] = "Not enough categorical columns for Sankey diagram"
            print("Not enough categorical columns for Sankey diagram.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Sankey Diagrams", results, table_name)

    def bubble_charts(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Bubble Charts"))
        image_paths = []
        results = {}

        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 3:
            # Extract actual column names and statistics for context
            x_column = numerical_columns[0]
            y_column = numerical_columns[1]
            size_column = numerical_columns[2]
            
            correlation = float(df[x_column].corr(df[y_column]))
            
            # Calculate statistics for each dimension
            x_stats = {
                "mean": float(df[x_column].mean()),
                "median": float(df[x_column].median()),
                "min": float(df[x_column].min()),
                "max": float(df[x_column].max())
            }
            
            y_stats = {
                "mean": float(df[y_column].mean()),
                "median": float(df[y_column].median()),
                "min": float(df[y_column].min()),
                "max": float(df[y_column].max())
            }
            
            size_stats = {
                "mean": float(df[size_column].mean()),
                "median": float(df[size_column].median()),
                "min": float(df[size_column].min()),
                "max": float(df[size_column].max())
            }
            
            results["analysis_info"] = {
                "x_column": str(x_column),
                "y_column": str(y_column),
                "size_column": str(size_column),
                "correlation_xy": correlation,
                "x_statistics": x_stats,
                "y_statistics": y_stats,
                "size_statistics": size_stats,
                "sample_size": len(df)
            }
            
            def plot_bubble_chart():
                x = df[numerical_columns[0]]
                y = df[numerical_columns[1]]
                size = df[numerical_columns[2]]
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(x, y, s=size, alpha=0.5)
                ax.set_xlabel(numerical_columns[0])
                ax.set_ylabel(numerical_columns[1])
                ax.set_title(f'Bubble Chart: {numerical_columns[0]} vs {numerical_columns[1]} (size: {numerical_columns[2]})')
                plt.colorbar(scatter)
                return fig, ax

            result = self.generate_plot(plot_bubble_chart)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_bubble_chart.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Bubble Chart", img_path))
            else:
                print("Skipping bubble chart plot due to timeout.")
                results["error"] = "Failed to generate bubble chart due to timeout"
        else:
            results["error"] = "Not enough numerical columns for bubble chart"
            print("Not enough numerical columns for bubble chart.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Bubble Charts", results, table_name)

    def geographical_plots(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Geographical Plots"))
        image_paths = []
        results = {}

        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Extract actual geographical data for context
            geo_stats = {
                "latitude": {
                    "min": float(df['latitude'].min()),
                    "max": float(df['latitude'].max()),
                    "center": float(df['latitude'].mean())
                },
                "longitude": {
                    "min": float(df['longitude'].min()),
                    "max": float(df['longitude'].max()),
                    "center": float(df['longitude'].mean())
                },
                "point_count": len(df),
                "region_span_km": {
                    "north_south": float(abs(df['latitude'].max() - df['latitude'].min()) * 111),  # Approx km per degree
                    "east_west": float(abs(df['longitude'].max() - df['longitude'].min()) * 111 * np.cos(np.radians(df['latitude'].mean())))
                }
            }
            
            # Identify point clusters if present
            if len(df) > 10:
                from sklearn.cluster import KMeans
                # Use just lat/long for clustering
                coords = df[['latitude', 'longitude']].dropna()
                if len(coords) > 5:  # Need at least a few points
                    kmeans = KMeans(n_clusters=min(5, len(coords)), random_state=42)
                    cluster_labels = kmeans.fit_predict(coords)
                    centers = kmeans.cluster_centers_
                    
                    clusters = []
                    for i, center in enumerate(centers):
                        point_count = sum(cluster_labels == i)
                        clusters.append({
                            "cluster_id": i,
                            "center": {"latitude": float(center[0]), "longitude": float(center[1])},
                            "point_count": int(point_count),
                            "percentage": float(point_count / len(coords) * 100)
                        })
                    
                    geo_stats["clusters"] = clusters
            
            results["analysis_info"] = geo_stats
            
            def plot_geographical():
                m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=6)
                for idx, row in df.iterrows():
                    folium.Marker([row['latitude'], row['longitude']]).add_to(m)
                img_path = os.path.join(self.output_folder, f"{table_name}_geographical_plot.html")
                m.save(img_path)
                
                # Also create a static image for the PDF
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(df['longitude'], df['latitude'], alpha=0.7)
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title('Geographic Distribution')
                
                static_img_path = os.path.join(self.output_folder, f"{table_name}_geographical_plot_static.png")
                plt.savefig(static_img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                return img_path, static_img_path

            try:
                html_path, static_path = self.generate_plot(plot_geographical)
                if html_path is not None:
                    image_paths.append(("Geographic Plot (Interactive)", html_path))
                    image_paths.append(("Geographic Plot (Static)", static_path))
                else:
                    print("Skipping geographical plot due to timeout.")
                    results["error"] = "Failed to generate geographical plot due to timeout"
            except Exception as e:
                print(f"Error creating geographical plot: {str(e)}")
                results["error"] = f"Failed to generate geographical plot: {str(e)}"
        else:
            results["error"] = "No latitude and longitude columns found for geographical plot"
            print("No latitude and longitude columns found for geographical plot.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Geographical Plots", results, table_name)

    def word_clouds(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Word Clouds"))
        image_paths = []
        results = {}

        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            # Extract actual text statistics for context
            text_column = text_columns[0]
            
            # Get word frequency counts (using WordCloud's process)
            text = " ".join(df[text_column].dropna().astype(str))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            word_freq = wordcloud.words_
            
            # Get top words for context
            top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20])
            
            results["analysis_info"] = {
                "text_column": str(text_column),
                "text_field_count": len(df[text_column].dropna()),
                "unique_words": len(word_freq),
                "top_words": {str(word): float(freq) for word, freq in top_words.items()},
                "average_word_count": len(text.split()) / max(1, len(df[text_column].dropna()))
            }
            
            def plot_word_cloud_and_pie():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.calculate_figure_size()[0]*2, self.calculate_figure_size()[1]))
                
                # Word Cloud
                ax1.imshow(wordcloud, interpolation='bilinear')
                ax1.axis('off')
                ax1.set_title(f'Word Cloud for {text_column}')
                
                # Pie Chart of top 10 words
                top_10_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10])
                ax2.pie(top_10_words.values(), labels=top_10_words.keys(), autopct='%1.1f%%')
                ax2.set_title('Top 10 Most Frequent Words')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_word_cloud_and_pie)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_word_cloud_and_pie.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Word Cloud and Frequency Pie Chart", img_path))
            else:
                print("Skipping word cloud and pie chart plot due to timeout.")
                results["error"] = "Failed to generate word cloud due to timeout"
        else:
            results["error"] = "No text columns found for word cloud"
            print("No text columns found for word cloud and pie chart.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Word Clouds and Frequency Pie Chart", results, table_name)

    def hierarchical_clustering_dendrogram(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Hierarchical Clustering Dendrogram"))
        image_paths = []
        results = {}

        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 1:
            # Extract context information about the actual clustering
            try:
                X = df[numerical_columns]
                imputer = SimpleImputer(strategy='mean')
                X_imputed = imputer.fit_transform(X)
                X_scaled = StandardScaler().fit_transform(X_imputed)
                
                n_clusters = min(5, len(df) - 1)  # Set a reasonable number of clusters
                model = AgglomerativeClustering(n_clusters=n_clusters)
                model = model.fit(X_scaled)
                
                # Calculate statistics for each cluster
                df_with_clusters = df.copy()
                df_with_clusters['cluster'] = model.labels_
                
                cluster_stats = []
                for i in range(n_clusters):
                    cluster_df = df_with_clusters[df_with_clusters['cluster'] == i]
                    cluster_stat = {
                        "cluster_id": i,
                        "size": len(cluster_df),
                        "percentage": float(len(cluster_df) / len(df) * 100),
                        "features": {}
                    }
                    
                    # Calculate statistics for key features
                    for col in numerical_columns:
                        if cluster_df[col].nunique() > 0:  # Only if we have data
                            cluster_stat["features"][str(col)] = {
                                "mean": float(cluster_df[col].mean()),
                                "median": float(cluster_df[col].median()),
                                "std": float(cluster_df[col].std()),
                                "min": float(cluster_df[col].min()),
                                "max": float(cluster_df[col].max())
                            }
                    
                    cluster_stats.append(cluster_stat)
                
                results["analysis_info"] = {
                    "numerical_columns_used": numerical_columns.tolist(),
                    "number_of_clusters": n_clusters,
                    "cluster_statistics": cluster_stats
                }
            except Exception as e:
                results["error"] = f"Error in cluster analysis: {str(e)}"
            
            def plot_dendrogram_and_pie():
                X = df[numerical_columns]
                imputer = SimpleImputer(strategy='mean')
                X_imputed = imputer.fit_transform(X)
                X_scaled = StandardScaler().fit_transform(X_imputed)
                
                model = AgglomerativeClustering(n_clusters=n_clusters)
                model = model.fit(X_scaled)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.calculate_figure_size()[0]*2, self.calculate_figure_size()[1]))
                
                # Dendrogram
                def plot_dendrogram_recursive(model, ax):
                    counts = np.zeros(model.children_.shape[0])
                    n_samples = len(model.labels_)
                    for i, merge in enumerate(model.children_):
                        current_count = 0
                        for child_idx in merge:
                            if child_idx < n_samples:
                                current_count += 1
                            else:
                                current_count += counts[child_idx - n_samples]
                        counts[i] = current_count

                    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
                    ax.set_title('Hierarchical Clustering Dendrogram')
                    ax.set_xlabel('Number of points in node (or index of point if no parenthesis)')
                    ax.set_ylabel('Distance')
                    dendrogram(linkage_matrix, ax=ax, truncate_mode='level', p=3)
                
                plot_dendrogram_recursive(model, ax1)
                ax1.set_title('Hierarchical Clustering Dendrogram')
                ax1.set_xlabel('Number of points in node')
                ax1.set_ylabel('Distance')
                
                # Pie chart of cluster distribution
                cluster_counts = pd.Series(model.labels_).value_counts()
                ax2.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%')
                ax2.set_title('Cluster Distribution')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_dendrogram_and_pie)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_dendrogram_and_pie.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Hierarchical Clustering Dendrogram", img_path))
            else:
                print("Skipping hierarchical clustering dendrogram and pie chart plot due to timeout.")
                results["error"] = "Failed to generate dendrogram due to timeout"
        else:
            results["error"] = "Not enough numerical columns for hierarchical clustering"
            print("Not enough numerical columns for hierarchical clustering dendrogram and pie chart.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Hierarchical Clustering Dendrogram and Cluster Distribution", results, table_name)

    def ecdf_plots(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - ECDF Plots"))
        image_paths = []
        results = {}

        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            # Extract distribution statistics for context
            distribution_stats = {}
            columns_to_plot = numerical_columns[:5]  # Limit to 5 columns for readability
            
            for col in columns_to_plot:
                col_data = df[col].dropna()
                
                # Get quantiles at key points
                quantiles = {
                    "q10": float(col_data.quantile(0.1)),
                    "q25": float(col_data.quantile(0.25)),
                    "median": float(col_data.quantile(0.5)),
                    "q75": float(col_data.quantile(0.75)),
                    "q90": float(col_data.quantile(0.9))
                }
                
                distribution_stats[str(col)] = {
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean()),
                    "quantiles": quantiles,
                    "data_points": len(col_data)
                }
            
            results["analysis_info"] = {
                "columns_analyzed": [str(col) for col in columns_to_plot],
                "distribution_statistics": distribution_stats
            }
            
            def plot_ecdf():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                for column in columns_to_plot:
                    data = df[column].dropna()
                    x = np.sort(data)
                    y = np.arange(1, len(data) + 1) / len(data)
                    ax.step(x, y, label=column)
                ax.set_xlabel('Value')
                ax.set_ylabel('ECDF')
                ax.set_title('Empirical Cumulative Distribution Function')
                ax.legend()
                return fig, ax

            result = self.generate_plot(plot_ecdf)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_ecdf_plot.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("ECDF Plot", img_path))
            else:
                print("Skipping ECDF plot due to timeout.")
                results["error"] = "Failed to generate ECDF plot due to timeout"
        else:
            results["error"] = "No numerical columns found for ECDF plot"
            print("No numerical columns found for ECDF plot.")
        
        results['image_paths'] = image_paths
        self.interpret_results("ECDF Plots", results, table_name)

    def ridgeline_plots(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Ridgeline Plots"))
        image_paths = []
        results = {}

        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numerical_columns) > 0 and len(categorical_columns) > 0:
            # Extract actual data distribution for context
            numerical_col = numerical_columns[0]
            categorical_col = categorical_columns[0]
            
            # Calculate statistics for each group
            group_stats = []
            for group_name, group_data in df.groupby(categorical_col):
                if len(group_data) > 0:
                    group_stats.append({
                        "group_name": str(group_name),
                        "count": int(len(group_data)),
                        "percentage": float(len(group_data) / len(df) * 100),
                        "mean": float(group_data[numerical_col].mean()),
                        "median": float(group_data[numerical_col].median()),
                        "std": float(group_data[numerical_col].std()),
                        "min": float(group_data[numerical_col].min()),
                        "max": float(group_data[numerical_col].max())
                    })
            
            results["analysis_info"] = {
                "numerical_column": str(numerical_col),
                "categorical_column": str(categorical_col),
                "total_groups": len(group_stats),
                "group_statistics": group_stats
            }
            
            def plot_ridgeline():
                # Ensure we have multiple categories
                if df[categorical_col].nunique() < 2:
                    print(f"Not enough categories in {categorical_col} for ridgeline plot.")
                    return None
                
                # Create the plot
                fig, axes = joypy.joyplot(
                    data=df,
                    by=categorical_col,
                    column=numerical_col,
                    colormap=plt.cm.viridis,
                    title=f"Ridgeline Plot of {numerical_col} by {categorical_col}",
                    labels=df[categorical_col].unique(),
                    figsize=self.calculate_figure_size()
                )
                
                # Adjust layout
                plt.tight_layout()
                return fig, axes

            result = self.generate_plot(plot_ridgeline)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_ridgeline_plot.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Ridgeline Plot", img_path))
            else:
                print("Skipping ridgeline plot due to timeout or insufficient data.")
                results["error"] = "Failed to generate ridgeline plot"
        else:
            if len(numerical_columns) == 0:
                results["error"] = "No numerical columns found for ridgeline plot"
            else:
                results["error"] = "No categorical columns found for ridgeline plot"
            print("Not enough numerical and categorical columns for ridgeline plot.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Ridgeline Plots", results, table_name)

    def hexbin_plots(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Hexbin Plots"))
        image_paths = []
        results = {}

        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            # Extract actual column data for context
            x_col = numerical_columns[0]
            y_col = numerical_columns[1]
            
            # Calculate statistics and correlation
            correlation = float(df[x_col].corr(df[y_col]))
            
            # Identify regions of high density
            x_data = df[x_col].dropna()
            y_data = df[y_col].dropna()
            
            # Create simplified density map (simulating hexbin)
            x_bins = np.linspace(x_data.min(), x_data.max(), 10)
            y_bins = np.linspace(y_data.min(), y_data.max(), 10)
            
            # Count data points in each bin
            bin_counts = np.zeros((len(x_bins)-1, len(y_bins)-1))
            for x, y in zip(x_data, y_data):
                x_idx = min(int((x - x_data.min()) / (x_data.max() - x_data.min()) * (len(x_bins)-1)), len(x_bins)-2)
                y_idx = min(int((y - y_data.min()) / (y_data.max() - y_data.min()) * (len(y_bins)-1)), len(y_bins)-2)
                if x_idx >= 0 and y_idx >= 0:
                    bin_counts[x_idx, y_idx] += 1
            
            # Find areas of high density
            high_density_regions = []
            for i in range(len(x_bins)-1):
                for j in range(len(y_bins)-1):
                    if bin_counts[i, j] > np.mean(bin_counts) + np.std(bin_counts):
                        high_density_regions.append({
                            "x_range": [float(x_bins[i]), float(x_bins[i+1])],
                            "y_range": [float(y_bins[j]), float(y_bins[j+1])],
                            "count": int(bin_counts[i, j]),
                            "percentage": float(bin_counts[i, j] / len(df) * 100)
                        })
            
            results["analysis_info"] = {
                "x_column": str(x_col),
                "y_column": str(y_col),
                "correlation": correlation,
                "data_points": len(df),
                "high_density_regions": high_density_regions[:5]  # Top 5 dense regions
            }
            
            def plot_hexbin():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                x = df[numerical_columns[0]]
                y = df[numerical_columns[1]]
                hb = ax.hexbin(x, y, gridsize=20, cmap='YlOrRd')
                ax.set_xlabel(numerical_columns[0])
                ax.set_ylabel(numerical_columns[1])
                ax.set_title(f'Hexbin Plot: {numerical_columns[0]} vs {numerical_columns[1]}')
                plt.colorbar(hb, label='Count')
                return fig, ax

            result = self.generate_plot(plot_hexbin)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_hexbin_plot.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Hexbin Plot", img_path))
            else:
                print("Skipping hexbin plot due to timeout.")
                results["error"] = "Failed to generate hexbin plot due to timeout"
        else:
            results["error"] = "Not enough numerical columns for hexbin plot"
            print("Not enough numerical columns for hexbin plot.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Hexbin Plots", results, table_name)

    def mosaic_plots(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Mosaic Plots"))
        image_paths = []
        results = {}

        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) >= 2:
            # Extract actual categorical relationships for context
            cat_col1 = categorical_columns[0]
            cat_col2 = categorical_columns[1]
            
            # Create contingency table for these categories
            contingency = pd.crosstab(df[cat_col1], df[cat_col2])
            
            # Convert to proportions for analysis
            prop_table = contingency.div(contingency.sum().sum())
            
            # Extract key relationships
            relationships = []
            for i, row_name in enumerate(contingency.index):
                for j, col_name in enumerate(contingency.columns):
                    count = int(contingency.iloc[i, j])
                    proportion = float(prop_table.iloc[i, j])
                    
                    relationships.append({
                        f"{cat_col1}": str(row_name),
                        f"{cat_col2}": str(col_name),
                        "count": count,
                        "proportion": proportion,
                        "percentage": proportion * 100
                    })
            
            # Sort by count
            relationships = sorted(relationships, key=lambda x: x["count"], reverse=True)
            
            results["analysis_info"] = {
                "categorical_column_1": str(cat_col1),
                "categorical_column_2": str(cat_col2),
                "top_relationships": relationships[:10],  # Top 10 relationships
                "total_combinations": len(relationships)
            }
            
            def plot_mosaic():
                try:
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    mosaic(df, [cat_col1, cat_col2], ax=ax, gap=0.05)
                    ax.set_title(f'Mosaic Plot of {cat_col1} vs {cat_col2}')
                    plt.tight_layout()
                    return fig, ax
                except Exception as e:
                    print(f"Error in creating mosaic plot: {str(e)}")
                    return None

            result = self.generate_plot(plot_mosaic)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_mosaic_plot.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Mosaic Plot", img_path))
            else:
                print("Skipping mosaic plot due to timeout or error.")
                results["error"] = "Failed to generate mosaic plot"
        else:
            results["error"] = "Not enough categorical columns for mosaic plot"
            print("Not enough categorical columns for mosaic plot.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Mosaic Plots", results, table_name)

    def lag_plots(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Lag Plots"))
        image_paths = []
        results = {}

        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            # Extract time series properties for context
            target_col = numerical_columns[0]
            
            # Calculate lag-1 autocorrelation
            series = df[target_col].dropna()
            if len(series) > 1:
                lag1_autocorr = float(series.autocorr(1))
                
                # Check for trend
                x = np.arange(len(series))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
                
                # Detect seasonality (simplified)
                has_seasonality = False
                seasonal_period = None
                
                for period in [2, 3, 4, 6, 12]:
                    if len(series) >= period * 2:
                        seasonal_autocorr = float(series.autocorr(period))
                        if abs(seasonal_autocorr) > 0.3:  # Arbitrary threshold
                            has_seasonality = True
                            seasonal_period = period
                            break
                
                results["analysis_info"] = {
                    "time_series_column": str(target_col),
                    "data_points": len(series),
                    "lag1_autocorrelation": lag1_autocorr,
                    "trend": {
                        "slope": float(slope),
                        "significant": bool(abs(r_value) > 0.5 and p_value < 0.05)
                    },
                    "seasonality": {
                        "detected": has_seasonality,
                        "period": seasonal_period
                    }
                }
            else:
                results["error"] = f"Not enough data points in {target_col} for lag analysis"
            
            def plot_lag():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                pd.plotting.lag_plot(df[numerical_columns[0]], lag=1, ax=ax)
                ax.set_title(f'Lag Plot for {numerical_columns[0]}')
                return fig, ax

            result = self.generate_plot(plot_lag)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_lag_plot.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Lag Plot", img_path))
            else:
                print("Skipping lag plot due to timeout.")
                results["error"] = "Failed to generate lag plot due to timeout"
        else:
            results["error"] = "No numerical columns found for lag plot"
            print("No numerical columns found for lag plot.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Lag Plots", results, table_name)

    def shapley_value_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Shapley Value Analysis"))
        image_paths = []
        results = {}

        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 1:
            try:
                # Extract actual feature importance for context
                X = df[numerical_columns]
                imputer = SimpleImputer(strategy='mean')
                X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                
                y = X_imputed.iloc[:, -1]  # Last column as target
                X = X_imputed.iloc[:, :-1]  # All other columns as features
                
                target_col = numerical_columns[-1]
                feature_cols = numerical_columns[:-1]
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # Extract feature importance from the model
                feature_importance = []
                for i, feature in enumerate(feature_cols):
                    feature_importance.append({
                        "feature": str(feature),
                        "importance": float(model.feature_importances_[i]),
                        "importance_percentage": float(model.feature_importances_[i] * 100 / sum(model.feature_importances_))
                    })
                
                # Sort by importance
                feature_importance = sorted(feature_importance, key=lambda x: x["importance"], reverse=True)
                
                # Generate SHAP values for more detailed analysis
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                # Calculate average absolute SHAP values for each feature
                mean_abs_shap = np.abs(shap_values).mean(0)
                
                # Extract SHAP-based importance
                shap_importance = []
                for i, feature in enumerate(feature_cols):
                    shap_importance.append({
                        "feature": str(feature),
                        "shap_value": float(mean_abs_shap[i]),
                        "importance_percentage": float(mean_abs_shap[i] * 100 / sum(mean_abs_shap))
                    })
                
                # Sort by SHAP value
                shap_importance = sorted(shap_importance, key=lambda x: x["shap_value"], reverse=True)
                
                results["analysis_info"] = {
                    "target_variable": str(target_col),
                    "feature_variables": [str(col) for col in feature_cols],
                    "random_forest_importance": feature_importance,
                    "shap_importance": shap_importance
                }
                
                def plot_shapley_and_pie():
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.calculate_figure_size()[0]*2, self.calculate_figure_size()[1]))
                    
                    # Shapley summary plot
                    shap.summary_plot(shap_values, X, plot_type="bar", show=False, ax=ax1)
                    ax1.set_title(f'Shapley Value Analysis for {target_col}')
                    
                    # Pie chart of feature importance
                    feature_importance = np.abs(shap_values).mean(0)
                    ax2.pie(feature_importance, labels=X.columns, autopct='%1.1f%%')
                    ax2.set_title('Feature Importance (Shapley Values)')
                    
                    plt.tight_layout()
                    return fig, (ax1, ax2)

                result = self.generate_plot(plot_shapley_and_pie)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_shapley_and_pie.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(("Shapley Value Analysis", img_path))
                else:
                    print("Skipping Shapley value analysis and pie chart plot due to timeout or error.")
                    results["error"] = "Failed to generate Shapley plot due to timeout"
            except Exception as e:
                results["error"] = f"Error in Shapley analysis: {str(e)}"
                print(f"Error in Shapley analysis: {str(e)}")
        else:
            results["error"] = "Not enough numerical columns for Shapley analysis"
            print("Not enough numerical columns for Shapley value analysis and pie chart.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Shapley Value Analysis and Feature Importance Pie Chart", results, table_name)

    def partial_dependence_plots(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Partial Dependence Plots"))
        image_paths = []
        results = {}
        
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_columns) > 1:
            try:
                # Extract actual feature relationships
                target = numeric_columns[-1]
                features = numeric_columns[:-1]
                
                X = df[features]
                y = df[target]
                
                # Handle missing values
                imputer_X = SimpleImputer(strategy='mean')
                imputer_y = SimpleImputer(strategy='mean')
                
                X_imputed = pd.DataFrame(imputer_X.fit_transform(X), columns=X.columns)
                y_imputed = pd.Series(imputer_y.fit_transform(y.values.reshape(-1, 1)).ravel(), name=y.name)
                
                # Remove any remaining NaN values
                mask = ~np.isnan(y_imputed)
                X_imputed = X_imputed[mask]
                y_imputed = y_imputed[mask]
                
                if len(y_imputed) == 0:
                    results["error"] = "No valid data remaining after handling NaN values"
                    print("No valid data remaining after handling NaN values. Skipping Partial Dependence Plots.")
                else:
                    # Build model
                    model = HistGradientBoostingRegressor(random_state=42)
                    model.fit(X_imputed, y_imputed)
                    
                    # Extract feature importance
                    feature_importances = model.feature_importances_
                    
                    # Calculate partial dependence for each feature
                    feature_dependencies = []
                    
                    for i, feature in enumerate(features):
                        # Calculate range of feature values for grid
                        feature_min = float(X_imputed[feature].min())
                        feature_max = float(X_imputed[feature].max())
                        feature_grid = np.linspace(feature_min, feature_max, 20)
                        
                        # Create grid for this feature
                        X_grid = np.tile(X_imputed.mean().values, (len(feature_grid), 1))
                        X_grid_df = pd.DataFrame(X_grid, columns=X_imputed.columns)
                        X_grid_df[feature] = feature_grid
                        
                        # Calculate predictions for grid
                        y_pred = model.predict(X_grid_df)
                        
                        # Sample key points from the dependence curve
                        sample_points = []
                        for j in range(0, len(feature_grid), 4):  # Sample every 4th point
                            sample_points.append({
                                "feature_value": float(feature_grid[j]),
                                "predicted_target": float(y_pred[j])
                            })
                        
                        feature_dependencies.append({
                            "feature": str(feature),
                            "importance": float(feature_importances[i]),
                            "importance_rank": int(np.argsort(-feature_importances)[i]) + 1,
                            "range": [feature_min, feature_max],
                            "dependence_curve_samples": sample_points
                        })
                    
                    results["analysis_info"] = {
                        "target_variable": str(target),
                        "feature_variables": [str(f) for f in features],
                        "feature_dependencies": feature_dependencies
                    }
                
                # Generate plots for each feature
                for feature in features:
                    def plot_pdp():
                        fig, ax = plt.subplots(figsize=(8, 6))
                        try:
                            PartialDependenceDisplay.from_estimator(model, X_imputed, [feature], ax=ax)
                            ax.set_title(f'Partial Dependence of {target} on {feature}')
                        except Exception as e:
                            print(f"Error plotting partial dependence for feature '{feature}': {str(e)}")
                            ax.text(0.5, 0.5, f"Error plotting {feature}", ha='center', va='center')
                        plt.tight_layout()
                        return fig, ax

                    result = self.generate_plot(plot_pdp)
                    if result is not None:
                        fig, _ = result
                        img_path = os.path.join(self.output_folder, f"{table_name}_partial_dependence_plot_{feature}.png")
                        plt.savefig(img_path, dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        image_paths.append((f"Partial Dependence: {target} on {feature}", img_path))
                    else:
                        print(f"Skipping Partial Dependence Plot for {feature} due to timeout.")
            except Exception as e:
                results["error"] = f"Error in partial dependence analysis: {str(e)}"
                print(f"Error in partial dependence analysis: {str(e)}")
        else:
            results["error"] = "Not enough numeric columns for Partial Dependence Plots"
            print("Not enough numeric columns for Partial Dependence Plots.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Partial Dependence Plots", results, table_name)

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

        worker_interpretation = self.worker_erag_api.chat([
            {"role": "system", "content": "You are an expert data analyst providing insights for business leaders and analysts. Respond in the requested format."}, 
            {"role": "user", "content": worker_prompt}
        ])

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
        output_file = os.path.join(self.output_folder, "axda_b2_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        report_title = f"Advanced Exploratory Data Analysis (Batch 2) Report for {self.table_name}"
        
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
            filename=f"axda_b2_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
            return pdf_file
        else:
            print(error("Failed to generate PDF report"))
            return None