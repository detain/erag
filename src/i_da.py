# Standard library imports
import os
import sqlite3
import threading
import time
import signal
import sys
from functools import wraps

# Third-party imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
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

class InnovativeDataAnalysis:
    def __init__(self, worker_erag_api, supervisor_erag_api, db_path):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.db_path = db_path
        self.technique_counter = 0
        self.total_techniques = 2  # AMPR and ETSF
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

    def calculate_figure_size(self, data=None, aspect_ratio=16/9):
        if data is not None:
            rows, cols = data.shape
            base_size = 8
            width = base_size * min(cols, 10) / 5
            height = base_size * min(rows, 20) / 10
        else:
            max_width = int(np.sqrt(self.max_pixels * aspect_ratio))
            max_height = int(max_width / aspect_ratio)
            width, height = max_width / 100, max_height / 100
        return (width, height)

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

    def run(self):
        print(info(f"Starting Innovative Data Analysis on {self.db_path}"))
        
        # Ask for database description before starting analysis
        self.prompt_for_database_description()
        
        all_tables = self.get_tables()
        user_tables = [table for table in all_tables if table.lower() not in ['information_schema', 'sqlite_master', 'sqlite_sequence', 'sqlite_stat1']]
        
        if not user_tables:
            print(error("No user tables found in the database. Exiting."))
            return
        
        selected_table = user_tables[0]  # Automatically select the first user table
        
        print(info(f"Analyzing table: '{selected_table}'"))
        
        self.analyze_table(selected_table)
        
        self.save_text_output()
        self.save_results_as_txt()
        self.generate_pdf_report()
        print(success(f"Innovative Data Analysis completed. Results saved in {self.output_folder}"))


    def get_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]

    def analyze_table(self, table_name):
        self.table_name = table_name
        self.output_folder = os.path.join(settings.output_folder, f"ida_{self.table_name}")
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, self.table_name)
        
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        
        # Load the full dataset
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            print(info(f"Loaded full dataset with {len(df)} rows and {len(df.columns)} columns"))
        
        # Identify entity names in the dataset
        self.entity_names_mapping = self.identify_entity_names(df)
        entity_description = self.format_entity_description()
        
        print(info(f"Identified entities in the data: {entity_description}"))

        analysis_methods = [
            self.ampr_analysis,
            self.etsf_analysis
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



    def ampr_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter+1}/{self.total_techniques} - Adaptive Multi-dimensional Pattern Recognition (AMPR)"))
        image_paths = []

        try:
            # Find numeric columns for analysis
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            X = df[numeric_columns].dropna()

            if X.empty:
                raise ValueError("No numeric data available for AMPR analysis.")

            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Apply PCA
            pca = PCA()
            pca.fit(X_scaled)
            cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
            X_pca = pca.transform(X_scaled)[:, :n_components]

            # Find optimal DBSCAN parameters
            best_silhouette = -1
            best_eps = 0.1
            for eps in np.arange(0.1, 1.1, 0.1):
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=5)
                    labels = dbscan.fit_predict(X_pca)
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(X_pca, labels)
                        if score > best_silhouette:
                            best_silhouette = score
                            best_eps = eps
                except Exception as e:
                    print(f"Error during clustering with eps={eps}: {str(e)}")

            # Apply clustering with best parameters
            if best_eps > 0:
                dbscan = DBSCAN(eps=best_eps, min_samples=5)
                cluster_labels = dbscan.fit_predict(X_pca)
            else:
                print("Could not find suitable clustering parameters. Using default KMeans clustering.")
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=3)
                cluster_labels = kmeans.fit_predict(X_pca)

            # Apply Isolation Forest for anomaly detection
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = isolation_forest.fit_predict(X_pca)

            # Get cluster sizes and properties
            clusters = np.unique(cluster_labels)
            cluster_sizes = {f"Cluster {i}": np.sum(cluster_labels == i) for i in clusters if i != -1}
            if -1 in clusters:
                cluster_sizes["Noise"] = np.sum(cluster_labels == -1)
            
            # Get anomaly count
            anomaly_count = np.sum(anomaly_labels == -1)
            
            # Identify outliers in original feature space
            outlier_indices = np.where(anomaly_labels == -1)[0]
            outlier_entities = {}
            
            # Try to connect outliers to entity names if available
            for entity_type, entity_values in self.entity_names_mapping.items():
                for col in df.columns:
                    if col.lower() == entity_type or any(term in col.lower() for term in [entity_type.lower(), entity_type.lower().rstrip('s')]):
                        outlier_values = df.iloc[outlier_indices][col].value_counts().to_dict()
                        if outlier_values:
                            outlier_entities[col] = outlier_values

            # Cluster Analysis Plot
            def plot_cluster_analysis():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
                ax.set_title('Cluster Analysis')
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                
                # Add legend
                legend_labels = [f'Cluster {i}' for i in np.unique(cluster_labels) if i != -1]
                if -1 in np.unique(cluster_labels):
                    legend_labels.append('Noise')
                
                # Create legend handles manually
                from matplotlib.lines import Line2D
                cmap = plt.cm.viridis
                unique_labels = np.unique(cluster_labels)
                colors = [cmap(i / max(1, len(unique_labels) - 1)) for i in range(len(unique_labels))]
                handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                                markersize=10, label=legend_labels[i]) 
                        for i in range(len(legend_labels))]
                
                ax.legend(handles=handles, loc='best')
                
                return fig, ax

            result = self.generate_plot(plot_cluster_analysis)
            if result is not None:
                fig, _ = result
                cluster_img_path = os.path.join(self.output_folder, f"{table_name}_ampr_cluster_analysis.png")
                plt.savefig(cluster_img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(cluster_img_path)

            # Feature Correlation Plot
            def plot_feature_correlation():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                correlation_matrix = pd.DataFrame(X_pca).corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
                ax.set_title('Feature Correlation of Principal Components')
                return fig, ax

            result = self.generate_plot(plot_feature_correlation)
            if result is not None:
                fig, _ = result
                correlation_img_path = os.path.join(self.output_folder, f"{table_name}_ampr_feature_correlation.png")
                plt.savefig(correlation_img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(correlation_img_path)

            # Anomaly Detection Plot
            def plot_anomaly_detection():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=anomaly_labels, cmap='RdYlGn')
                ax.set_title('Anomaly Detection')
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                
                # Add legend
                from matplotlib.lines import Line2D
                handles = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Normal'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Anomaly')
                ]
                ax.legend(handles=handles, loc='best')
                
                return fig, ax

            result = self.generate_plot(plot_anomaly_detection)
            if result is not None:
                fig, _ = result
                anomaly_img_path = os.path.join(self.output_folder, f"{table_name}_ampr_anomaly_detection.png")
                plt.savefig(anomaly_img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(anomaly_img_path)

            # PCA Explained Variance Plot
            def plot_pca_explained_variance():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.calculate_figure_size()[0]*2, self.calculate_figure_size()[1]))
                
                # Bar chart of individual variance
                explained_variance_ratio = pca.explained_variance_ratio_[:n_components]
                ax1.bar(range(1, n_components + 1), explained_variance_ratio)
                ax1.set_xlabel('Principal Component')
                ax1.set_ylabel('Explained Variance Ratio')
                ax1.set_title('Individual Explained Variance')
                
                # Line chart of cumulative variance
                ax2.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                        cumulative_variance_ratio, 
                        marker='o')
                ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
                ax2.axvline(x=n_components, color='g', linestyle='--', 
                           label=f'{n_components} Components Selected')
                ax2.set_xlabel('Number of Components')
                ax2.set_ylabel('Cumulative Explained Variance')
                ax2.set_title('Cumulative Explained Variance')
                ax2.legend()
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_pca_explained_variance)
            if result is not None:
                fig, _ = result
                pca_img_path = os.path.join(self.output_folder, f"{table_name}_ampr_pca_explained_variance.png")
                plt.savefig(pca_img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(pca_img_path)
            
            # Calculate feature importance from PCA loadings
            feature_importance = {}
            for i in range(min(3, n_components)):  # Top 3 components
                loadings = pca.components_[i]
                abs_loadings = np.abs(loadings)
                # Get top 5 features for this component
                top_indices = abs_loadings.argsort()[-5:][::-1]
                top_features = {numeric_columns[idx]: float(loadings[idx]) for idx in top_indices}
                feature_importance[f"PC{i+1}"] = top_features

            results = {
                "n_components": n_components,
                "best_eps": best_eps,
                "best_silhouette_score": best_silhouette,
                "n_clusters": len(np.unique(cluster_labels)),
                "cluster_sizes": cluster_sizes,
                "n_anomalies": int(anomaly_count),
                "anomaly_percentage": float((anomaly_count / len(X)) * 100),
                "feature_importance": feature_importance,
                "outlier_entities": outlier_entities,
                "image_paths": [
                    ("AMPR Analysis - Cluster Analysis", cluster_img_path),
                    ("AMPR Analysis - Feature Correlation", correlation_img_path),
                    ("AMPR Analysis - Anomaly Detection", anomaly_img_path),
                    ("AMPR Analysis - PCA Explained Variance", pca_img_path)
                ]
            }

            self.interpret_results("AMPR Analysis", results, table_name)

        except Exception as e:
            error_message = f"An error occurred during AMPR Analysis: {str(e)}"
            print(error(error_message))
            self.interpret_results("AMPR Analysis", {'error': error_message}, table_name)
        
        finally:
            self.technique_counter += 1

    def etsf_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter+1}/{self.total_techniques} - Enhanced Time Series Forecasting (ETSF)"))
        image_paths = []

        try:
            # Find date columns
            date_columns = df.select_dtypes(include=['datetime64']).columns
            
            # If no datetime columns, try to convert string columns to datetime
            if len(date_columns) == 0:
                for col in df.select_dtypes(include=['object']).columns:
                    try:
                        # Check if column name suggests it's a date
                        if any(date_term in col.lower() for date_term in ['date', 'time', 'day', 'year', 'month']):
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            if not df[col].isna().all():
                                date_columns = [col]
                                print(info(f"Converted {col} to datetime for time series analysis"))
                                break
                    except:
                        continue
            
            if len(date_columns) == 0:
                raise ValueError("No date column found for ETSF analysis.")

            date_column = date_columns[0]
            
            # Find numeric columns
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            
            if len(numeric_columns) == 0:
                raise ValueError("No numeric columns found for ETSF analysis.")

            # For ETSF, choose the first numeric column as target
            # In a real-world scenario, you might want to let the user choose
            value_column = numeric_columns[0]
            
            # Identify entities and their relationship to the time series
            entity_time_series = {}
            for entity_type, entity_values in self.entity_names_mapping.items():
                for col in df.columns:
                    if col.lower() == entity_type or any(term in col.lower() for term in [entity_type.lower(), entity_type.lower().rstrip('s')]):
                        # Check if we have enough entities for meaningful analysis (not too many)
                        if len(entity_values) <= 10:
                            # Create time series for each entity
                            for entity in entity_values:
                                entity_data = df[df[col] == entity]
                                if len(entity_data) >= 10:  # Need enough data points
                                    entity_time_series[f"{entity} ({col})"] = entity_data

            # Main time series analysis with the whole dataset
            df_ts = df[[date_column, value_column]].dropna().sort_values(date_column)
            df_ts = df_ts.set_index(date_column)

            if len(df_ts) < 30:
                raise ValueError("Not enough data for meaningful ETSF analysis (need at least 30 data points).")

            # Enhance the time series with additional features
            df_ts['lag_1'] = df_ts[value_column].shift(1)
            df_ts['lag_7'] = df_ts[value_column].shift(7)
            
            # Add Fourier terms for seasonality
            df_ts['fourier_sin'] = np.sin(2 * np.pi * df_ts.index.dayofyear / 365.25)
            df_ts['fourier_cos'] = np.cos(2 * np.pi * df_ts.index.dayofyear / 365.25)
            
            # Drop rows with NAs from the lag features
            df_ts = df_ts.dropna()

            # Run stationarity test
            result = adfuller(df_ts[value_column].dropna())
            adf_result = {
                'ADF Statistic': float(result[0]),
                'p-value': float(result[1]),
                'Critical Values': {f"{key}": float(val) for key, val in result[4].items()}
            }
            
            # Determine seasonality
            # Try to infer frequency from the data
            if df_ts.index.freq is None:
                inferred_freq = pd.infer_freq(df_ts.index)
                if inferred_freq is not None:
                    df_ts = df_ts.asfreq(inferred_freq)
            
            # Determine appropriate seasonal period based on data frequency
            if df_ts.index.freq == 'D' or str(df_ts.index.freq).startswith('D'):
                seasonal_period = 7  # Weekly seasonality for daily data
            elif df_ts.index.freq == 'M' or str(df_ts.index.freq).startswith('M'):
                seasonal_period = 12  # Yearly seasonality for monthly data
            elif df_ts.index.freq == 'Q' or str(df_ts.index.freq).startswith('Q'):
                seasonal_period = 4  # Yearly seasonality for quarterly data
            else:
                # Default value - using a reasonable period based on data length
                seasonal_period = min(len(df_ts) // 4, 12)
                
            print(info(f"Using seasonal period of {seasonal_period} for time series decomposition"))

            # Seasonal decomposition
            decomposition = seasonal_decompose(df_ts[value_column], model='additive', period=seasonal_period)
            
            # Observed Data Plot
            def plot_observed_data():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(decomposition.observed)
                ax.set_title(f'Observed Time Series - {value_column}')
                ax.set_xlabel('Date')
                ax.set_ylabel(value_column)
                plt.xticks(rotation=45)
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_observed_data)
            if result is not None:
                fig, _ = result
                observed_img_path = os.path.join(self.output_folder, f"{table_name}_etsf_observed_data.png")
                plt.savefig(observed_img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(observed_img_path)

            # Trend Plot
            def plot_trend():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(decomposition.trend)
                ax.set_title(f'Trend Component - {value_column}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Trend')
                plt.xticks(rotation=45)
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_trend)
            if result is not None:
                fig, _ = result
                trend_img_path = os.path.join(self.output_folder, f"{table_name}_etsf_trend.png")
                plt.savefig(trend_img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(trend_img_path)

            # Seasonal Plot
            def plot_seasonal():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(decomposition.seasonal)
                ax.set_title(f'Seasonal Component - {value_column} (Period: {seasonal_period})')
                ax.set_xlabel('Date')
                ax.set_ylabel('Seasonality')
                plt.xticks(rotation=45)
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_seasonal)
            if result is not None:
                fig, _ = result
                seasonal_img_path = os.path.join(self.output_folder, f"{table_name}_etsf_seasonal.png")
                plt.savefig(seasonal_img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(seasonal_img_path)

            # Residual Plot
            def plot_residual():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(decomposition.resid)
                ax.set_title(f'Residual Component - {value_column}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Residual')
                plt.xticks(rotation=45)
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_residual)
            if result is not None:
                fig, _ = result
                residual_img_path = os.path.join(self.output_folder, f"{table_name}_etsf_residual.png")
                plt.savefig(residual_img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(residual_img_path)

            # Split the data into train and test sets
            train_size = int(len(df_ts) * 0.8)
            train, test = df_ts[:train_size], df_ts[train_size:]

            # Fit ARIMA model with exogenous variables
            try:
                model = ARIMA(train[value_column], order=(1,1,1), 
                            exog=train[['lag_1', 'lag_7', 'fourier_sin', 'fourier_cos']])
                results = model.fit()

                # Generate predictions
                predictions = results.forecast(steps=len(test), 
                                            exog=test[['lag_1', 'lag_7', 'fourier_sin', 'fourier_cos']])

                # Forecast Plot
                def plot_forecast():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    
                    # Plot full dataset
                    ax.plot(df_ts.index, df_ts[value_column], label='Actual', alpha=0.7)
                    
                    # Plot train/test split
                    ax.axvline(x=train.index[-1], color='gray', linestyle='--', alpha=0.7, 
                              label='Train/Test Split')
                    
                    # Plot forecast
                    ax.plot(test.index, predictions, label='Forecast', color='red')
                    
                    ax.set_title(f'ETSF Forecast vs Actual - {value_column}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel(value_column)
                    ax.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_forecast)
                if result is not None:
                    fig, _ = result
                    forecast_img_path = os.path.join(self.output_folder, f"{table_name}_etsf_forecast.png")
                    plt.savefig(forecast_img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(forecast_img_path)

                # Calculate error metrics
                mse = mean_squared_error(test[value_column], predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(test[value_column], predictions)
                
                # Calculate forecast for future periods
                future_periods = min(30, int(len(df_ts) * 0.1))  # Forecast 10% or up to 30 periods
                
                # Prepare exogenous data for future forecast
                # This is simplified - in a real scenario, you would need to properly handle the lag values
                future_exog = test[['lag_1', 'lag_7', 'fourier_sin', 'fourier_cos']].iloc[-future_periods:].reset_index(drop=True)
                
                # Generate future forecast
                future_forecast = results.forecast(steps=future_periods, exog=future_exog)
                
                # Generate future dates
                last_date = df_ts.index[-1]
                if isinstance(last_date, pd.Timestamp):
                    freq = df_ts.index.freq or pd.infer_freq(df_ts.index) or 'D'
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_periods, freq=freq)
                    future_dates_str = [date.strftime('%Y-%m-%d') for date in future_dates]
                else:
                    future_dates_str = [f"Future {i+1}" for i in range(future_periods)]

                model_results = {
                    "adf_result": adf_result,
                    "is_stationary": adf_result['p-value'] < 0.05,
                    "seasonal_period": seasonal_period,
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "entity_time_series": list(entity_time_series.keys()),
                    "future_forecast": future_forecast.tolist(),
                    "future_dates": future_dates_str,
                    "date_column": date_column,
                    "value_column": value_column,
                    "image_paths": [
                        ("ETSF Analysis - Observed Data", observed_img_path),
                        ("ETSF Analysis - Trend", trend_img_path),
                        ("ETSF Analysis - Seasonal", seasonal_img_path),
                        ("ETSF Analysis - Residual", residual_img_path),
                        ("ETSF Analysis - Forecast", forecast_img_path)
                    ]
                }
                
                self.interpret_results("ETSF Analysis", model_results, table_name)
                
            except Exception as e:
                error_message = f"Error in ARIMA modeling: {str(e)}"
                print(warning(error_message))
                
                # Try a simpler model if ARIMA fails
                try:
                    # Use simple exponential smoothing
                    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
                    
                    model = SimpleExpSmoothing(train[value_column])
                    results = model.fit()
                    
                    # Generate predictions
                    predictions = results.forecast(len(test))
                    
                    # Calculate error metrics
                    mse = mean_squared_error(test[value_column], predictions)
                    rmse = np.sqrt(mse)
                    
                    # Plot forecast with simpler model
                    def plot_simple_forecast():
                        fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                        ax.plot(df_ts.index, df_ts[value_column], label='Actual')
                        ax.plot(test.index, predictions, label='Forecast', color='red')
                        ax.set_title(f'ETSF Forecast vs Actual (Simple Model) - {value_column}')
                        ax.set_xlabel('Date')
                        ax.set_ylabel(value_column)
                        ax.legend()
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        return fig, ax

                    result = self.generate_plot(plot_simple_forecast)
                    if result is not None:
                        fig, _ = result
                        forecast_img_path = os.path.join(self.output_folder, f"{table_name}_etsf_simple_forecast.png")
                        plt.savefig(forecast_img_path, dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        image_paths.append(forecast_img_path)
                    
                    results = {
                        "adf_result": adf_result,
                        "is_stationary": adf_result['p-value'] < 0.05,
                        "seasonal_period": seasonal_period,
                        "model_type": "Simple Exponential Smoothing (fallback)",
                        "mse": float(mse),
                        "rmse": float(rmse),
                        "note": "Used simpler model due to ARIMA failure",
                        "error_in_arima": error_message,
                        "image_paths": [
                            ("ETSF Analysis - Observed Data", observed_img_path),
                            ("ETSF Analysis - Trend", trend_img_path),
                            ("ETSF Analysis - Seasonal", seasonal_img_path),
                            ("ETSF Analysis - Residual", residual_img_path),
                            ("ETSF Analysis - Simple Forecast", forecast_img_path)
                        ]
                    }
                    
                    self.interpret_results("ETSF Analysis", results, table_name)
                    
                except Exception as e:
                    error_message = f"Error in both ARIMA and Simple Exponential Smoothing: {str(e)}"
                    print(error(error_message))
                    
                    results = {
                        "adf_result": adf_result,
                        "is_stationary": adf_result['p-value'] < 0.05,
                        "seasonal_period": seasonal_period,
                        "error": error_message,
                        "image_paths": [
                            ("ETSF Analysis - Observed Data", observed_img_path),
                            ("ETSF Analysis - Trend", trend_img_path),
                            ("ETSF Analysis - Seasonal", seasonal_img_path),
                            ("ETSF Analysis - Residual", residual_img_path)
                        ]
                    }
                    
                    self.interpret_results("ETSF Analysis", results, table_name)

        except Exception as e:
            error_message = f"An error occurred during ETSF Analysis: {str(e)}"
            print(error(error_message))
            self.interpret_results("ETSF Analysis", {'error': error_message}, table_name)
        
        finally:
            self.technique_counter += 1

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
        output_file = os.path.join(self.output_folder, "ida_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def save_results_as_txt(self):
        output_file = os.path.join(self.output_folder, f"ida_{self.table_name}_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(f"Innovative Data Analysis Results for {self.table_name}\n\n")
            
            # Add database description
            if self.database_description:
                f.write(f"Database Description: {self.database_description}\n\n")
                
            # Add entity information
            entity_description = self.format_entity_description()
            if entity_description != "No specific entities identified.":
                f.write(f"Entities in the Data:\n{entity_description}\n\n")
            
            f.write("Key Findings:\n")
            for finding in self.findings:
                f.write(f"- {finding}\n")
            f.write("\nDetailed Analysis Results:\n")
            f.write(self.text_output)
        print(success(f"Results saved as txt file: {output_file}"))

    def generate_pdf_report(self):
        report_title = f"Innovative Data Analysis Report for {self.table_name}"
        
        # Add database description to the report
        if self.database_description:
            report_title += f"\nDatabase Description: {self.database_description}"
        
        pdf_file = self.pdf_generator.create_enhanced_pdf_report(
            self.findings,
            self.pdf_content,
            self.image_data,
            filename=f"ida_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
        else:
            print(error("Failed to generate PDF report"))

# Example usage
if __name__ == "__main__":
    from src.api_model import EragAPI
    
    worker_api = EragAPI("worker_model_name")
    supervisor_api = EragAPI("supervisor_model_name")
    
    db_path = "path/to/your/database.sqlite"
    
    ida = InnovativeDataAnalysis(worker_api, supervisor_api, db_path)
    
    ida.run()