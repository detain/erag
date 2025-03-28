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

from scipy import stats
from scipy.stats import norm, zscore, chi2
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

import networkx as nx
from Bio import Align

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

class AdvancedExploratoryDataAnalysisB4:
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
        print(info("Please provide a description of the database for advanced analysis (Batch 4). This will help the AI models provide better insights."))
        print(info("Describe the purpose, main tables, key data points, and any other relevant information:"))
        self.database_description = input("> ")
        print(success(f"Database description received: {self.database_description}"))

    def get_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]

    def run(self):
        self.prompt_for_database_description()
        print(info(f"Starting Advanced Exploratory Data Analysis (Batch4) on {self.db_path}"))
        
        tables = self.get_tables()
        for table in tables:
            self.analyze_table(table)
        
        self.save_text_output()
        self.generate_pdf_report()
        print(success(f"Advanced Exploratory Data Analysis (Batch4) completed. Results saved in {self.output_folder}"))

    def analyze_table(self, table_name):
        self.table_name = table_name
        self.output_folder = os.path.join(settings.output_folder, f"axda_b4_{self.table_name}")
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, self.table_name)
        
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            print(info(f"Loaded full dataset with {len(df)} rows and {len(df.columns)} columns"))

        analysis_methods = [
            self.matrix_profile,
            self.ensemble_anomaly_detection,
            self.gaussian_mixture_models,
            self.expectation_maximization,
            self.statistical_process_control,
            self.z_score_analysis,
            self.mahalanobis_distance,
            self.box_cox_transformation,
            self.grubbs_test,
            self.chauvenet_criterion,
            self.benfords_law_analysis,
            self.forensic_accounting,
            self.network_analysis_fraud_detection,
            self.sequence_alignment,
            self.conformal_anomaly_detection
        ]

        for method in analysis_methods:
            try:
                # Check if execution is paused
                self.check_if_paused()
                self.technique_counter += 1
                method(df, table_name)
            except Exception as e:
                error_message = f"An error occurred during {method.__name__}: {str(e)}"
                print(error(error_message))
                self.text_output += f"\n{error_message}\n"
                
                # Write error to method-specific output file
                method_name = method.__name__
                with open(os.path.join(self.output_folder, f"{method_name}_results.txt"), "w", encoding='utf-8') as f:
                    f.write(error_message)
                
                self.pdf_content.append((method.__name__, [], error_message))

    def matrix_profile(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Matrix Profile"))
        image_paths = []
        results = {}
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            # Extract actual data for context
            try:
                # Select the first numerical column for analysis
                target_column = numerical_columns[0]
                data = df[target_column].dropna().astype(float).values
                
                # Set appropriate window size
                window_size = min(len(data) // 4, 100)
                
                # Calculate basic statistics
                results["analysis_info"] = {
                    "target_column": str(target_column),
                    "window_size": window_size,
                    "data_points": len(data),
                    "data_statistics": {
                        "mean": float(np.mean(data)),
                        "std": float(np.std(data)),
                        "min": float(np.min(data)),
                        "max": float(np.max(data)),
                        "median": float(np.median(data))
                    }
                }
            except Exception as e:
                results["error"] = f"Error extracting matrix profile context: {str(e)}"
                print(f"Error extracting matrix profile context: {str(e)}")
            
            def plot_matrix_profile():
                from stumpy import stump
                
                # Select the first numerical column for demonstration
                data = df[numerical_columns[0]].dropna().astype(float).values  # Convert to float
                window_size = min(len(data) // 4, 100)  # Adjust window size as needed
                
                mp = stump(data, m=window_size)
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.calculate_figure_size())
                ax1.plot(data)
                ax1.set_title(f"Original Data: {numerical_columns[0]}")
                ax2.plot(mp[:, 0])
                ax2.set_title("Matrix Profile")
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_matrix_profile)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_matrix_profile.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Matrix Profile", img_path))
            else:
                print("Skipping Matrix Profile plot due to error in plot generation.")
                results["error"] = "Failed to generate Matrix Profile plot"
        else:
            results["error"] = "No numerical columns found for Matrix Profile analysis"
            print("No numerical columns found for Matrix Profile analysis.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Matrix Profile", results, table_name)

    def ensemble_anomaly_detection(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Ensemble Anomaly Detection"))
        image_paths = []
        results = {}
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            # Extract actual anomaly detection information for context
            try:
                X = df[numerical_columns].dropna()
                
                # Ensemble of anomaly detection methods
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                elliptic_env = EllipticEnvelope(contamination=0.1, random_state=42)
                
                # Fit and predict
                iso_forest_pred = iso_forest.fit_predict(X)
                elliptic_env_pred = elliptic_env.fit_predict(X)
                
                # Combine predictions (1 for inlier, -1 for outlier)
                ensemble_pred = np.mean([iso_forest_pred, elliptic_env_pred], axis=0)
                
                # Extract anomaly statistics
                anomaly_count = sum(ensemble_pred < 0)
                
                # Get actual anomalous records
                anomaly_indices = np.where(ensemble_pred < 0)[0]
                anomalies = []
                
                for idx in anomaly_indices[:10]:  # Limit to first 10 for brevity
                    record = {}
                    for col in numerical_columns:
                        record[str(col)] = float(X.iloc[idx][col])
                    anomalies.append(record)
                
                results["analysis_info"] = {
                    "variables_analyzed": numerical_columns.tolist(),
                    "anomaly_detection": {
                        "total_records": len(X),
                        "anomalies_detected": int(anomaly_count),
                        "anomaly_percentage": float(anomaly_count / len(X) * 100),
                        "sample_anomalies": anomalies
                    },
                    "detection_methods": ["Isolation Forest", "Elliptic Envelope"]
                }
            except Exception as e:
                results["error"] = f"Error in ensemble anomaly detection analysis: {str(e)}"
                print(f"Error extracting ensemble anomaly detection details: {str(e)}")
            
            def plot_ensemble_anomaly_detection():
                X = df[numerical_columns].dropna()
                
                # Ensemble of anomaly detection methods
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                elliptic_env = EllipticEnvelope(contamination=0.1, random_state=42)
                
                # Fit and predict
                iso_forest_pred = iso_forest.fit_predict(X)
                elliptic_env_pred = elliptic_env.fit_predict(X)
                
                # Combine predictions (1 for inlier, -1 for outlier)
                ensemble_pred = np.mean([iso_forest_pred, elliptic_env_pred], axis=0)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=ensemble_pred, cmap='viridis')
                ax.set_xlabel(numerical_columns[0])
                ax.set_ylabel(numerical_columns[1])
                ax.set_title("Ensemble Anomaly Detection")
                plt.colorbar(scatter, label='Anomaly Score')
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_ensemble_anomaly_detection)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_ensemble_anomaly_detection.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Ensemble Anomaly Detection", img_path))
            else:
                print("Skipping Ensemble Anomaly Detection plot due to error in plot generation.")
                results["error"] = "Failed to generate Ensemble Anomaly Detection plot"
        else:
            results["error"] = "Not enough numerical columns for Ensemble Anomaly Detection analysis"
            print("Not enough numerical columns for Ensemble Anomaly Detection analysis.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Ensemble Anomaly Detection", results, table_name)

    def gaussian_mixture_models(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Gaussian Mixture Models (GMM)"))
        image_paths = []
        results = {}
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            # Extract GMM information for context
            try:
                X = df[numerical_columns].dropna()
                
                # Fit GMM
                n_components = 3  # Number of clusters
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(X)
                
                # Predict
                labels = gmm.predict(X)
                
                # Calculate cluster statistics
                clusters = []
                for i in range(n_components):
                    cluster_data = X[labels == i]
                    if len(cluster_data) > 0:
                        cluster = {
                            "cluster_id": i,
                            "size": len(cluster_data),
                            "percentage": float(len(cluster_data) / len(X) * 100),
                            "features": {}
                        }
                        
                        # Calculate statistics for each feature in this cluster
                        for j, col in enumerate(numerical_columns):
                            feature_data = cluster_data[col]
                            cluster["features"][str(col)] = {
                                "mean": float(feature_data.mean()),
                                "std": float(feature_data.std()),
                                "min": float(feature_data.min()),
                                "max": float(feature_data.max())
                            }
                        
                        clusters.append(cluster)
                
                results["analysis_info"] = {
                    "variables_analyzed": numerical_columns.tolist(),
                    "gmm_analysis": {
                        "number_of_components": n_components,
                        "bic_score": float(gmm.bic(X)),
                        "aic_score": float(gmm.aic(X)),
                        "clusters": clusters
                    }
                }
            except Exception as e:
                results["error"] = f"Error in GMM analysis: {str(e)}"
                print(f"Error extracting GMM details: {str(e)}")
            
            def plot_gmm():
                X = df[numerical_columns].dropna()
                
                # Fit GMM
                gmm = GaussianMixture(n_components=3, random_state=42)
                gmm.fit(X)
                
                # Predict
                labels = gmm.predict(X)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis')
                ax.set_xlabel(numerical_columns[0])
                ax.set_ylabel(numerical_columns[1])
                ax.set_title("Gaussian Mixture Models (GMM)")
                plt.colorbar(scatter, label='Cluster')
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_gmm)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_gmm.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Gaussian Mixture Models", img_path))
            else:
                print("Skipping Gaussian Mixture Models plot due to error in plot generation.")
                results["error"] = "Failed to generate GMM plot"
        else:
            results["error"] = "Not enough numerical columns for GMM analysis"
            print("Not enough numerical columns for Gaussian Mixture Models analysis.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Gaussian Mixture Models (GMM)", results, table_name)

    def expectation_maximization(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Expectation-Maximization Algorithm"))
        image_paths = []
        results = {}
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            # Extract EM information for context (similar to GMM as they're implemented the same in sklearn)
            try:
                X = df[numerical_columns].dropna()
                
                # EM is essentially the same as GMM in sklearn
                n_components = 3  # Number of clusters
                em = GaussianMixture(n_components=n_components, random_state=42)
                em.fit(X)
                
                # Predict
                labels = em.predict(X)
                
                # Calculate cluster statistics
                clusters = []
                for i in range(n_components):
                    cluster_data = X[labels == i]
                    if len(cluster_data) > 0:
                        cluster = {
                            "cluster_id": i,
                            "size": len(cluster_data),
                            "percentage": float(len(cluster_data) / len(X) * 100),
                            "features": {}
                        }
                        
                        # Calculate statistics for each feature in this cluster
                        for j, col in enumerate(numerical_columns):
                            feature_data = cluster_data[col]
                            cluster["features"][str(col)] = {
                                "mean": float(feature_data.mean()),
                                "std": float(feature_data.std()),
                                "min": float(feature_data.min()),
                                "max": float(feature_data.max())
                            }
                        
                        clusters.append(cluster)
                
                results["analysis_info"] = {
                    "variables_analyzed": numerical_columns.tolist(),
                    "em_analysis": {
                        "number_of_components": n_components,
                        "log_likelihood": float(em.score(X) * len(X)),
                        "iterations": int(em.n_iter_),
                        "clusters": clusters
                    }
                }
            except Exception as e:
                results["error"] = f"Error in EM analysis: {str(e)}"
                print(f"Error extracting EM details: {str(e)}")
            
            def plot_em():
                X = df[numerical_columns].dropna()
                
                # EM is essentially the same as GMM in sklearn
                em = GaussianMixture(n_components=3, random_state=42)
                em.fit(X)
                
                # Predict
                labels = em.predict(X)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis')
                ax.set_xlabel(numerical_columns[0])
                ax.set_ylabel(numerical_columns[1])
                ax.set_title("Expectation-Maximization Algorithm")
                plt.colorbar(scatter, label='Cluster')
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_em)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_em.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Expectation-Maximization", img_path))
            else:
                print("Skipping Expectation-Maximization plot due to error in plot generation.")
                results["error"] = "Failed to generate EM plot"
        else:
            results["error"] = "Not enough numerical columns for EM analysis"
            print("Not enough numerical columns for Expectation-Maximization analysis.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Expectation-Maximization Algorithm", results, table_name)

    def statistical_process_control(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Statistical Process Control (SPC) Charts"))
        image_paths = []
        results = {}
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            # Extract SPC information for context
            try:
                target_column = numerical_columns[0]
                data = df[target_column].dropna()
                
                # Calculate control limits
                mean = data.mean()
                std = data.std()
                ucl = mean + 3 * std  # Upper control limit
                lcl = mean - 3 * std  # Lower control limit
                
                # Find points out of control
                out_of_control = data[(data > ucl) | (data < lcl)]
                
                # Extract actual out-of-control points
                out_of_control_points = []
                for idx, value in out_of_control.items():
                    out_of_control_points.append({
                        "index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                        "value": float(value),
                        "deviation": float(value - mean),
                        "deviation_sigmas": float((value - mean) / std)
                    })
                
                results["analysis_info"] = {
                    "target_column": str(target_column),
                    "process_statistics": {
                        "mean": float(mean),
                        "std": float(std),
                        "ucl": float(ucl),
                        "lcl": float(lcl)
                    },
                    "control_analysis": {
                        "total_points": len(data),
                        "points_out_of_control": len(out_of_control),
                        "percentage_out_of_control": float(len(out_of_control) / len(data) * 100),
                        "out_of_control_examples": out_of_control_points[:10]  # Limit to first 10 for brevity
                    }
                }
            except Exception as e:
                results["error"] = f"Error in SPC analysis: {str(e)}"
                print(f"Error extracting SPC details: {str(e)}")
            
            def plot_spc():
                data = df[numerical_columns[0]].dropna()
                
                mean = data.mean()
                std = data.std()
                ucl = mean + 3 * std
                lcl = mean - 3 * std
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(data, marker='o', linestyle='-', linewidth=0.5, markersize=5)
                ax.axhline(mean, color='g', linestyle='--', label='Mean')
                ax.axhline(ucl, color='r', linestyle='--', label='UCL')
                ax.axhline(lcl, color='r', linestyle='--', label='LCL')
                
                # Highlight out of control points
                out_indices = np.where((data > ucl) | (data < lcl))[0]
                ax.scatter(out_indices, data.iloc[out_indices], color='red', s=80, label='Out of Control')
                
                ax.set_title(f"Control Chart for {numerical_columns[0]}")
                ax.set_ylabel("Value")
                ax.set_xlabel("Sample")
                ax.legend()
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_spc)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_spc.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Statistical Process Control", img_path))
            else:
                print("Skipping Statistical Process Control plot due to error in plot generation.")
                results["error"] = "Failed to generate SPC plot"
        else:
            results["error"] = "No numerical columns found for SPC analysis"
            print("No numerical columns found for Statistical Process Control analysis.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Statistical Process Control (SPC) Charts", results, table_name)

    def z_score_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Z-Score and Modified Z-Score"))
        image_paths = []
        results = {}
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            # Extract Z-score information for context
            try:
                target_column = numerical_columns[0]
                data = df[target_column].dropna()
                
                # Calculate Z-scores and modified Z-scores
                z_scores = zscore(data)
                modified_z_scores = 0.6745 * (data - np.median(data)) / np.median(np.abs(data - np.median(data)))
                
                # Find outliers using both methods
                z_outliers = np.abs(z_scores) > 3
                mod_z_outliers = np.abs(modified_z_scores) > 3.5
                
                # Get actual outlier values
                z_outlier_values = data[z_outliers]
                mod_z_outlier_values = data[mod_z_outliers]
                
                # Extract some example outliers
                z_outlier_examples = []
                for idx, value in z_outlier_values.items():
                    z_outlier_examples.append({
                        "index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                        "value": float(value),
                        "z_score": float(z_scores[idx])
                    })
                
                mod_z_outlier_examples = []
                for idx, value in mod_z_outlier_values.items():
                    mod_z_outlier_examples.append({
                        "index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                        "value": float(value),
                        "modified_z_score": float(modified_z_scores[idx])
                    })
                
                results["analysis_info"] = {
                    "target_column": str(target_column),
                    "z_score_analysis": {
                        "total_points": len(data),
                        "z_score_outliers": int(np.sum(z_outliers)),
                        "z_score_percentage": float(np.sum(z_outliers) / len(data) * 100),
                        "z_score_examples": z_outlier_examples[:5]  # Limit to first 5 for brevity
                    },
                    "modified_z_score_analysis": {
                        "total_points": len(data),
                        "mod_z_outliers": int(np.sum(mod_z_outliers)),
                        "mod_z_percentage": float(np.sum(mod_z_outliers) / len(data) * 100),
                        "mod_z_examples": mod_z_outlier_examples[:5]  # Limit to first 5 for brevity
                    }
                }
            except Exception as e:
                results["error"] = f"Error in Z-score analysis: {str(e)}"
                print(f"Error extracting Z-score details: {str(e)}")
            
            def plot_z_score():
                data = df[numerical_columns[0]].dropna()
                
                z_scores = zscore(data)
                modified_z_scores = 0.6745 * (data - np.median(data)) / np.median(np.abs(data - np.median(data)))
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.calculate_figure_size())
                ax1.scatter(range(len(data)), z_scores)
                ax1.axhline(y=3, color='r', linestyle='--')
                ax1.axhline(y=-3, color='r', linestyle='--')
                ax1.set_title(f"Z-Scores for {numerical_columns[0]}")
                ax1.set_ylabel("Z-Score")
                
                ax2.scatter(range(len(data)), modified_z_scores)
                ax2.axhline(y=3.5, color='r', linestyle='--')
                ax2.axhline(y=-3.5, color='r', linestyle='--')
                ax2.set_title(f"Modified Z-Scores for {numerical_columns[0]}")
                ax2.set_ylabel("Modified Z-Score")
                ax2.set_xlabel("Sample")
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_z_score)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_z_score.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Z-Score Analysis", img_path))
            else:
                print("Skipping Z-Score plot due to error in plot generation.")
                results["error"] = "Failed to generate Z-score plot"
        else:
            results["error"] = "No numerical columns found for Z-score analysis"
            print("No numerical columns found for Z-Score analysis.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Z-Score and Modified Z-Score", results, table_name)

    def mahalanobis_distance(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Mahalanobis Distance"))
        image_paths = []
        results = {}
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            # Extract Mahalanobis distance information for context
            try:
                X = df[numerical_columns].dropna()
                
                # Remove constant columns
                X = X.loc[:, (X != X.iloc[0]).any()]
                
                if X.shape[1] < 2:
                    results["error"] = "Not enough variable columns for Mahalanobis Distance analysis"
                    print("Not enough variable columns for Mahalanobis Distance analysis.")
                else:
                    # Calculate mean and covariance
                    mean = np.mean(X, axis=0)
                    cov = np.cov(X, rowvar=False)
                    
                    try:
                        # Calculate Mahalanobis distance
                        inv_cov = np.linalg.inv(cov)
                        diff = X - mean
                        left = np.dot(diff, inv_cov)
                        mahalanobis = np.sqrt(np.sum(left * diff, axis=1))
                        
                        # Identify outliers (e.g., top 5%)
                        threshold = np.percentile(mahalanobis, 95)
                        outliers = mahalanobis > threshold
                        
                        # Extract actual outlier records
                        outlier_examples = []
                        for idx in np.where(outliers)[0][:10]:  # Limit to first 10 for brevity
                            record = {
                                "index": int(idx),
                                "mahalanobis_distance": float(mahalanobis[idx]),
                                "values": {}
                            }
                            for col in X.columns:
                                record["values"][str(col)] = float(X.iloc[idx][col])
                            outlier_examples.append(record)
                        
                        results["analysis_info"] = {
                            "variables_analyzed": X.columns.tolist(),
                            "mahalanobis_analysis": {
                                "total_points": len(X),
                                "outliers_detected": int(np.sum(outliers)),
                                "outlier_percentage": float(np.sum(outliers) / len(X) * 100),
                                "threshold_used": float(threshold),
                                "outlier_examples": outlier_examples
                            }
                        }
                    except np.linalg.LinAlgError:
                        results["error"] = "Singular matrix encountered. Could not calculate Mahalanobis distances."
                        print("Singular matrix encountered. Skipping Mahalanobis Distance analysis.")
            except Exception as e:
                results["error"] = f"Error in Mahalanobis distance analysis: {str(e)}"
                print(f"Error extracting Mahalanobis distance details: {str(e)}")
            
            def plot_mahalanobis():
                X = df[numerical_columns].dropna()
                
                # Remove constant columns
                X = X.loc[:, (X != X.iloc[0]).any()]
                
                if X.shape[1] < 2:
                    print("Not enough variable columns for Mahalanobis Distance analysis.")
                    return None

                # Calculate mean and covariance
                mean = np.mean(X, axis=0)
                cov = np.cov(X, rowvar=False)
                
                try:
                    # Calculate Mahalanobis distance
                    inv_cov = np.linalg.inv(cov)
                    diff = X - mean
                    left = np.dot(diff, inv_cov)
                    mahalanobis = np.sqrt(np.sum(left * diff, axis=1))
                    
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=mahalanobis, cmap='viridis')
                    ax.set_xlabel(X.columns[0])
                    ax.set_ylabel(X.columns[1])
                    ax.set_title("Mahalanobis Distance")
                    plt.colorbar(scatter, label='Mahalanobis Distance')
                    plt.tight_layout()
                    return fig, ax
                except np.linalg.LinAlgError:
                    print("Singular matrix encountered. Skipping Mahalanobis Distance analysis.")
                    return None

            result = self.generate_plot(plot_mahalanobis)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_mahalanobis.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Mahalanobis Distance", img_path))
            else:
                print("Skipping Mahalanobis Distance plot due to error in plot generation.")
                if "error" not in results:
                    results["error"] = "Failed to generate Mahalanobis distance plot"
        else:
            results["error"] = "Not enough numerical columns for Mahalanobis Distance analysis"
            print("Not enough numerical columns for Mahalanobis Distance analysis.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Mahalanobis Distance", results, table_name)

    def box_cox_transformation(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Box-Cox Transformation"))
        image_paths = []
        results = {}
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            # Extract Box-Cox transformation information for context
            try:
                from scipy.stats import boxcox
                
                target_column = numerical_columns[0]
                data = df[target_column].dropna()
                
                # Ensure all values are positive
                if np.min(data) <= 0:
                    data = data - np.min(data) + 1
                
                transformed_data, lambda_param = boxcox(data)
                
                # Calculate statistics for original and transformed data
                orig_stats = {
                    "mean": float(np.mean(data)),
                    "median": float(np.median(data)),
                    "std": float(np.std(data)),
                    "skewness": float(stats.skew(data)),
                    "kurtosis": float(stats.kurtosis(data))
                }
                
                trans_stats = {
                    "mean": float(np.mean(transformed_data)),
                    "median": float(np.median(transformed_data)),
                    "std": float(np.std(transformed_data)),
                    "skewness": float(stats.skew(transformed_data)),
                    "kurtosis": float(stats.kurtosis(transformed_data))
                }
                
                results["analysis_info"] = {
                    "target_column": str(target_column),
                    "box_cox_lambda": float(lambda_param),
                    "original_data": orig_stats,
                    "transformed_data": trans_stats,
                    "normality_improvement": {
                        "skewness_reduction": float(abs(stats.skew(data)) - abs(stats.skew(transformed_data))),
                        "kurtosis_improvement": float(abs(stats.kurtosis(data) - 0) - abs(stats.kurtosis(transformed_data) - 0))
                    }
                }
            except Exception as e:
                results["error"] = f"Error in Box-Cox transformation analysis: {str(e)}"
                print(f"Error extracting Box-Cox transformation details: {str(e)}")
            
            def plot_box_cox():
                from scipy.stats import boxcox
                
                data = df[numerical_columns[0]].dropna()
                
                # Ensure all values are positive
                if np.min(data) <= 0:
                    data = data - np.min(data) + 1
                
                transformed_data, lambda_param = boxcox(data)
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.calculate_figure_size())
                ax1.hist(data, bins=30)
                ax1.set_title(f"Original Distribution of {numerical_columns[0]}")
                ax1.set_ylabel("Frequency")
                
                ax2.hist(transformed_data, bins=30)
                ax2.set_title(f"Box-Cox Transformed Distribution (λ = {lambda_param:.2f})")
                ax2.set_ylabel("Frequency")
                ax2.set_xlabel("Value")
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_box_cox)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_box_cox.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Box-Cox Transformation", img_path))
            else:
                print("Skipping Box-Cox Transformation plot due to error in plot generation.")
                if "error" not in results:
                    results["error"] = "Failed to generate Box-Cox transformation plot"
        else:
            results["error"] = "No numerical columns found for Box-Cox transformation analysis"
            print("No numerical columns found for Box-Cox Transformation analysis.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Box-Cox Transformation", results, table_name)

    def grubbs_test(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Grubbs' Test"))
        image_paths = []
        results = {}
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            # Extract Grubbs' test information for context
            try:
                from scipy import stats
                
                target_column = numerical_columns[0]
                data = df[target_column].dropna()
                
                def grubbs_test(data, alpha=0.05):
                    n = len(data)
                    mean = np.mean(data)
                    std = np.std(data)
                    z = (data - mean) / std
                    G = np.max(np.abs(z))
                    max_idx = np.argmax(np.abs(z))
                    t_value = stats.t.ppf(1 - alpha / (2 * n), n - 2)
                    G_critical = ((n - 1) * np.sqrt(t_value**2 / (n - 2 + t_value**2))) / np.sqrt(n)
                    return G > G_critical, G, G_critical, max_idx, data[max_idx]
                
                is_outlier, G, G_critical, max_idx, outlier_value = grubbs_test(data)
                
                results["analysis_info"] = {
                    "target_column": str(target_column),
                    "grubbs_test": {
                        "sample_size": len(data),
                        "grubbs_statistic": float(G),
                        "critical_value": float(G_critical),
                        "is_outlier_present": bool(is_outlier),
                        "significance_level": 0.05,
                        "outlier_details": {
                            "index": int(max_idx) if isinstance(max_idx, (int, np.integer)) else str(max_idx),
                            "value": float(outlier_value),
                            "z_score": float((outlier_value - np.mean(data)) / np.std(data))
                        } if is_outlier else None
                    }
                }
            except Exception as e:
                results["error"] = f"Error in Grubbs' test analysis: {str(e)}"
                print(f"Error extracting Grubbs' test details: {str(e)}")
            
            def plot_grubbs():
                from scipy import stats
                
                data = df[numerical_columns[0]].dropna()
                
                def grubbs_test(data, alpha=0.05):
                    n = len(data)
                    mean = np.mean(data)
                    std = np.std(data)
                    z = (data - mean) / std
                    G = np.max(np.abs(z))
                    t_value = stats.t.ppf(1 - alpha / (2 * n), n - 2)
                    G_critical = ((n - 1) * np.sqrt(t_value**2 / (n - 2 + t_value**2))) / np.sqrt(n)
                    return G > G_critical, G, G_critical
                
                is_outlier, G, G_critical = grubbs_test(data)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.boxplot(data)
                ax.set_title(f"Grubbs' Test for {numerical_columns[0]}")
                ax.set_ylabel("Value")
                plt.figtext(0.5, 0.01, f"Grubbs' statistic: {G:.2f}, Critical value: {G_critical:.2f}, Outlier Present: {is_outlier}", ha="center")
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_grubbs)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_grubbs.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Grubbs' Test", img_path))
            else:
                print("Skipping Grubbs' Test plot due to error in plot generation.")
                if "error" not in results:
                    results["error"] = "Failed to generate Grubbs' test plot"
        else:
            results["error"] = "No numerical columns found for Grubbs' test analysis"
            print("No numerical columns found for Grubbs' Test analysis.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Grubbs' Test", results, table_name)

    def chauvenet_criterion(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Chauvenet's Criterion"))
        image_paths = []
        results = {}
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            # Extract Chauvenet's criterion information for context
            try:
                target_column = numerical_columns[0]
                data = df[target_column].dropna()
                
                def chauvenet(data):
                    mean = np.mean(data)
                    std = np.std(data)
                    N = len(data)
                    criterion = 1.0 / (2 * N)
                    d = abs(data - mean) / std
                    prob = 2 * (1 - norm.cdf(d))
                    return prob < criterion
                
                is_outlier = chauvenet(data)
                outlier_count = np.sum(is_outlier)
                
                # Extract actual outlier values
                outlier_indices = np.where(is_outlier)[0]
                outlier_examples = []
                
                for idx in outlier_indices[:10]:  # Limit to first 10 for brevity
                    outlier_examples.append({
                        "index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                        "value": float(data.iloc[idx]),
                        "deviation": float(data.iloc[idx] - np.mean(data)),
                        "deviation_sigmas": float((data.iloc[idx] - np.mean(data)) / np.std(data))
                    })
                
                results["analysis_info"] = {
                    "target_column": str(target_column),
                    "chauvenet_analysis": {
                        "sample_size": len(data),
                        "outliers_detected": int(outlier_count),
                        "outlier_percentage": float(outlier_count / len(data) * 100),
                        "outlier_examples": outlier_examples
                    }
                }
            except Exception as e:
                results["error"] = f"Error in Chauvenet's criterion analysis: {str(e)}"
                print(f"Error extracting Chauvenet's criterion details: {str(e)}")
            
            def plot_chauvenet():
                data = df[numerical_columns[0]].dropna()
                
                def chauvenet(data):
                    mean = np.mean(data)
                    std = np.std(data)
                    N = len(data)
                    criterion = 1.0 / (2 * N)
                    d = abs(data - mean) / std
                    prob = 2 * (1 - norm.cdf(d))
                    return prob < criterion
                
                is_outlier = chauvenet(data)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.scatter(range(len(data)), data, c=['r' if x else 'b' for x in is_outlier])
                ax.set_title(f"Chauvenet's Criterion for {numerical_columns[0]}")
                ax.set_ylabel("Value")
                ax.set_xlabel("Sample")
                
                # Add a legend
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Outlier'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='Normal')
                ]
                ax.legend(handles=legend_elements)
                
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_chauvenet)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_chauvenet.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Chauvenet's Criterion", img_path))
            else:
                print("Skipping Chauvenet's Criterion plot due to error in plot generation.")
                if "error" not in results:
                    results["error"] = "Failed to generate Chauvenet's criterion plot"
        else:
            results["error"] = "No numerical columns found for Chauvenet's criterion analysis"
            print("No numerical columns found for Chauvenet's Criterion analysis.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Chauvenet's Criterion", results, table_name)

    def benfords_law_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Benford's Law Analysis"))
        image_paths = []
        results = {}
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            # Extract Benford's Law analysis information for context
            try:
                target_column = numerical_columns[0]
                data = df[target_column].dropna()
                
                def get_first_digit(n):
                    return int(str(abs(n)).strip('0.')[0])
                
                first_digits = data.apply(get_first_digit)
                observed_freq = first_digits.value_counts().sort_index() / len(first_digits)
                expected_freq = pd.Series([np.log10(1 + 1/d) for d in range(1, 10)], index=range(1, 10))
                
                # Calculate chi-square statistic
                observed_counts = first_digits.value_counts().sort_index()
                expected_counts = expected_freq * len(first_digits)
                
                chi2, p_value = stats.chisquare(observed_counts, expected_counts)
                conforms_to_benford = p_value > 0.05  # Using 5% significance level
                
                # Format the digit distribution for analysis
                digit_distribution = []
                for digit in range(1, 10):
                    observed = float(observed_freq.get(digit, 0))
                    expected = float(expected_freq.get(digit, 0))
                    
                    digit_distribution.append({
                        "digit": digit,
                        "observed_frequency": observed,
                        "expected_frequency": expected,
                        "difference": float(observed - expected),
                        "difference_percentage": float((observed - expected) / expected * 100) if expected != 0 else None
                    })
                
                results["analysis_info"] = {
                    "target_column": str(target_column),
                    "benfords_law_analysis": {
                        "sample_size": len(data),
                        "chi_square_statistic": float(chi2),
                        "p_value": float(p_value),
                        "conforms_to_benford": bool(conforms_to_benford),
                        "digit_distribution": digit_distribution
                    }
                }
            except Exception as e:
                results["error"] = f"Error in Benford's Law analysis: {str(e)}"
                print(f"Error extracting Benford's Law details: {str(e)}")
            
            def plot_benfords_law():
                data = df[numerical_columns[0]].dropna()
                
                def get_first_digit(n):
                    return int(str(abs(n)).strip('0.')[0])
                
                first_digits = data.apply(get_first_digit)
                observed_freq = first_digits.value_counts().sort_index() / len(first_digits)
                
                expected_freq = pd.Series([np.log10(1 + 1/d) for d in range(1, 10)], index=range(1, 10))
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                
                # Bar plot
                observed_freq.plot(kind='bar', ax=ax1, alpha=0.5, label='Observed')
                expected_freq.plot(kind='line', ax=ax1, color='r', label='Expected (Benford\'s Law)')
                ax1.set_title(f"Benford's Law Analysis for {numerical_columns[0]}")
                ax1.set_xlabel("First Digit")
                ax1.set_ylabel("Frequency")
                ax1.legend()
                
                # Pie chart
                ax2.pie(observed_freq, labels=observed_freq.index, autopct='%1.1f%%', startangle=90)
                ax2.set_title("Distribution of First Digits")
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_benfords_law)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_benfords_law.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Benford's Law Analysis", img_path))
            else:
                print("Skipping Benford's Law Analysis plot due to error in plot generation.")
                if "error" not in results:
                    results["error"] = "Failed to generate Benford's Law analysis plot"
        else:
            results["error"] = "No numerical columns found for Benford's Law analysis"
            print("No numerical columns found for Benford's Law Analysis.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Benford's Law Analysis", results, table_name)

    def forensic_accounting(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Forensic Accounting Techniques"))
        image_paths = []
        results = {}
    
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            # Extract forensic accounting information for context
            try:
                X = df[numerical_columns].dropna()
                
                # Remove constant columns
                X = X.loc[:, (X != X.iloc[0]).any()]
                
                if X.shape[1] < 2:
                    results["error"] = "Not enough variable columns for Forensic Accounting analysis"
                    print("Not enough variable columns for Forensic Accounting analysis.")
                else:
                    try:
                        # Calculate Mahalanobis distance
                        mean = np.mean(X, axis=0)
                        cov = np.cov(X, rowvar=False)
                        inv_cov = np.linalg.inv(cov)
                        diff = X - mean
                        left = np.dot(diff, inv_cov)
                        mahalanobis = np.sqrt(np.sum(left * diff, axis=1))
                        
                        # Define threshold for anomalies (e.g., top 5% of Mahalanobis distances)
                        threshold = np.percentile(mahalanobis, 95)
                        anomalies = mahalanobis > threshold
                        
                        # Extract suspicious records
                        suspicious_records = []
                        for idx in np.where(anomalies)[0][:10]:  # Limit to 10 for brevity
                            record = {
                                "index": int(idx),
                                "anomaly_score": float(mahalanobis[idx]),
                                "values": {}
                            }
                            for col in X.columns:
                                record["values"][str(col)] = float(X.iloc[idx][col])
                            suspicious_records.append(record)
                        
                        results["analysis_info"] = {
                            "variables_analyzed": X.columns.tolist(),
                            "forensic_analysis": {
                                "total_records": len(X),
                                "suspicious_records": int(np.sum(anomalies)),
                                "suspicious_percentage": float(np.sum(anomalies) / len(X) * 100),
                                "detection_method": "Mahalanobis Distance",
                                "threshold_used": float(threshold),
                                "suspicious_examples": suspicious_records
                            }
                        }
                    except np.linalg.LinAlgError:
                        # Alternative analysis: Simple outlier detection using Z-score
                        z_scores = np.abs(stats.zscore(X))
                        outliers = (z_scores > 3).any(axis=1)
                        
                        # Extract suspicious records
                        suspicious_records = []
                        for idx in np.where(outliers)[0][:10]:  # Limit to 10 for brevity
                            record = {
                                "index": int(idx),
                                "max_z_score": float(z_scores.iloc[idx].max()),
                                "values": {}
                            }
                            for col in X.columns:
                                record["values"][str(col)] = float(X.iloc[idx][col])
                            suspicious_records.append(record)
                        
                        results["analysis_info"] = {
                            "variables_analyzed": X.columns.tolist(),
                            "forensic_analysis": {
                                "total_records": len(X),
                                "suspicious_records": int(np.sum(outliers)),
                                "suspicious_percentage": float(np.sum(outliers) / len(X) * 100),
                                "detection_method": "Z-score (fallback method)",
                                "threshold_used": 3.0,  # Standard threshold for Z-score
                                "suspicious_examples": suspicious_records
                            }
                        }
            except Exception as e:
                results["error"] = f"Error in Forensic Accounting analysis: {str(e)}"
                print(f"Error extracting Forensic Accounting details: {str(e)}")
            
            def plot_forensic_accounting():
                X = df[numerical_columns].dropna()
                
                # Remove constant columns
                X = X.loc[:, (X != X.iloc[0]).any()]
                
                if X.shape[1] < 2:
                    print("Not enough variable columns for Forensic Accounting analysis.")
                    return None

                try:
                    # Calculate Mahalanobis distance
                    mean = np.mean(X, axis=0)
                    cov = np.cov(X, rowvar=False)
                    inv_cov = np.linalg.inv(cov)
                    diff = X - mean
                    left = np.dot(diff, inv_cov)
                    mahalanobis = np.sqrt(np.sum(left * diff, axis=1))
                    
                    # Define threshold for anomalies (e.g., top 5% of Mahalanobis distances)
                    threshold = np.percentile(mahalanobis, 95)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                    scatter = ax1.scatter(X.iloc[:, 0], X.iloc[:, 1], c=mahalanobis, cmap='viridis')
                    ax1.set_xlabel(X.columns[0])
                    ax1.set_ylabel(X.columns[1])
                    ax1.set_title("Forensic Accounting: Potential Anomalies")
                    plt.colorbar(scatter, ax=ax1, label='Mahalanobis Distance')
                    
                    # Highlight potential anomalies
                    anomalies = X[mahalanobis > threshold]
                    ax1.scatter(anomalies.iloc[:, 0], anomalies.iloc[:, 1], color='red', s=100, facecolors='none', edgecolors='r', label='Potential Anomalies')
                    
                    ax1.legend()
                    
                    # Pie chart
                    anomaly_count = len(anomalies)
                    normal_count = len(X) - anomaly_count
                    ax2.pie([normal_count, anomaly_count], labels=['Normal', 'Anomalies'], autopct='%1.1f%%', startangle=90)
                    ax2.set_title("Distribution of Anomalies")
                    
                    plt.tight_layout()
                    return fig, (ax1, ax2)
                except np.linalg.LinAlgError:
                    print("Singular matrix encountered. Performing alternative analysis.")
                    
                    # Alternative analysis: Simple outlier detection using Z-score
                    z_scores = np.abs(stats.zscore(X))
                    outliers = (z_scores > 3).any(axis=1)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                    ax1.scatter(X.iloc[:, 0], X.iloc[:, 1], c='blue', label='Normal')
                    ax1.scatter(X[outliers].iloc[:, 0], X[outliers].iloc[:, 1], c='red', label='Potential Anomalies')
                    ax1.set_xlabel(X.columns[0])
                    ax1.set_ylabel(X.columns[1])
                    ax1.set_title("Forensic Accounting: Potential Anomalies (Z-score method)")
                    ax1.legend()
                    
                    # Pie chart for alternative analysis
                    anomaly_count = outliers.sum()
                    normal_count = len(X) - anomaly_count
                    ax2.pie([normal_count, anomaly_count], labels=['Normal', 'Anomalies'], autopct='%1.1f%%', startangle=90)
                    ax2.set_title("Distribution of Anomalies (Z-score method)")
                    
                    plt.tight_layout()
                    return fig, (ax1, ax2)

            result = self.generate_plot(plot_forensic_accounting)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_forensic_accounting.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Forensic Accounting", img_path))
            else:
                print("Skipping Forensic Accounting plot due to error in plot generation.")
                if "error" not in results:
                    results["error"] = "Failed to generate Forensic Accounting plot"
        else:
            results["error"] = "Not enough numerical columns for Forensic Accounting analysis"
            print("Not enough numerical columns for Forensic Accounting analysis.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Forensic Accounting Techniques", results, table_name)

    def network_analysis_fraud_detection(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Network Analysis for Fraud Detection"))
        image_paths = []
        results = {}
        
        # For this example, we'll assume we have columns for 'source', 'target', and 'amount'
        if all(col in df.columns for col in ['source', 'target', 'amount']):
            # Extract network analysis information for context
            try:
                G = nx.from_pandas_edgelist(df, 'source', 'target', ['amount'])
                
                # Calculate degree centrality
                degree_centrality = nx.degree_centrality(G)
                
                # Identify potential fraudulent nodes (e.g., nodes with high degree centrality)
                threshold = np.percentile(list(degree_centrality.values()), 95)
                suspicious_nodes = [node for node, centrality in degree_centrality.items() if centrality > threshold]
                
                # Extract network statistics
                node_stats = []
                for node, centrality in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]:  # Top 10 by centrality
                    node_stats.append({
                        "node_id": str(node),
                        "degree_centrality": float(centrality),
                        "neighbors": len(list(G.neighbors(node))),
                        "connections": [str(neighbor) for neighbor in G.neighbors(node)][:5]  # Limit to first 5 for brevity
                    })
                
                results["analysis_info"] = {
                    "network_analysis": {
                        "total_nodes": len(G.nodes()),
                        "total_edges": len(G.edges()),
                        "average_degree": float(sum(dict(G.degree()).values()) / len(G.nodes())) if len(G.nodes()) > 0 else 0,
                        "suspicious_nodes": len(suspicious_nodes),
                        "suspicious_percentage": float(len(suspicious_nodes) / len(G.nodes()) * 100) if len(G.nodes()) > 0 else 0,
                        "detection_method": "Degree Centrality",
                        "threshold_used": float(threshold),
                        "top_nodes_by_centrality": node_stats
                    }
                }
            except Exception as e:
                results["error"] = f"Error in network analysis: {str(e)}"
                print(f"Error extracting network analysis details: {str(e)}")
            
            def plot_network_analysis():
                G = nx.from_pandas_edgelist(df, 'source', 'target', ['amount'])
                
                # Calculate degree centrality
                degree_centrality = nx.degree_centrality(G)
                
                # Identify potential fraudulent nodes (e.g., nodes with high degree centrality)
                threshold = np.percentile(list(degree_centrality.values()), 95)
                suspicious_nodes = [node for node, centrality in degree_centrality.items() if centrality > threshold]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                
                # Network plot
                pos = nx.spring_layout(G, seed=42)  # Use seed for reproducibility
                
                # Draw normal nodes
                nx.draw_networkx_nodes(G, pos, 
                                      nodelist=[n for n in G.nodes() if n not in suspicious_nodes],
                                      node_color='lightblue', 
                                      node_size=500, 
                                      ax=ax1)
                
                # Draw suspicious nodes
                nx.draw_networkx_nodes(G, pos, 
                                      nodelist=suspicious_nodes,
                                      node_color='red', 
                                      node_size=700, 
                                      ax=ax1)
                
                # Draw edges
                nx.draw_networkx_edges(G, pos, ax=ax1)
                
                # Draw labels
                nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)
                
                ax1.set_title("Network Analysis for Fraud Detection")
                ax1.axis('off')
                
                # Add a legend
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Normal'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Suspicious')
                ]
                ax1.legend(handles=legend_elements)
                
                # Pie chart
                suspicious_count = len(suspicious_nodes)
                normal_count = len(G.nodes()) - suspicious_count
                ax2.pie([normal_count, suspicious_count], labels=['Normal', 'Suspicious'], autopct='%1.1f%%', startangle=90)
                ax2.set_title("Distribution of Suspicious Nodes")
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_network_analysis)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_network_analysis.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Network Analysis", img_path))
            else:
                print("Skipping Network Analysis plot due to error in plot generation.")
                if "error" not in results:
                    results["error"] = "Failed to generate Network Analysis plot"
        else:
            print("Required columns (source, target, amount) not found. Performing alternative analysis.")
            
            numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numerical_columns) >= 2:
                # Extract correlation network information for context
                try:
                    X = df[numerical_columns].dropna()
                    
                    # Perform a simple correlation analysis
                    corr = X.corr()
                    
                    # Extract strong correlations
                    strong_correlations = []
                    for i in range(len(corr.columns)):
                        for j in range(i+1, len(corr.columns)):
                            if abs(corr.iloc[i, j]) > 0.7:  # Strong correlation threshold
                                strong_correlations.append({
                                    "variable_1": str(corr.columns[i]),
                                    "variable_2": str(corr.columns[j]),
                                    "correlation": float(corr.iloc[i, j]),
                                    "strength": "very strong" if abs(corr.iloc[i, j]) > 0.9 else "strong"
                                })
                    
                    results["analysis_info"] = {
                        "alternative_network_analysis": {
                            "variables_analyzed": numerical_columns.tolist(),
                            "correlation_analysis": {
                                "total_variable_pairs": int(len(numerical_columns) * (len(numerical_columns) - 1) / 2),
                                "strong_correlations": len(strong_correlations),
                                "strong_correlation_examples": sorted(strong_correlations, key=lambda x: abs(x["correlation"]), reverse=True)[:10]  # Top 10 by abs correlation
                            }
                        }
                    }
                except Exception as e:
                    results["error"] = f"Error in alternative network analysis: {str(e)}"
                    print(f"Error extracting alternative network analysis details: {str(e)}")
                
                def plot_alternative_analysis():
                    X = df[numerical_columns].dropna()
                    
                    # Perform a simple correlation analysis
                    corr = X.corr()
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                    
                    # Heatmap
                    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax1)
                    ax1.set_title("Correlation Heatmap")
                    
                    # Count high correlations
                    high_corr = (np.abs(corr) > 0.7).sum().sum() / 2  # Divide by 2 to avoid counting twice
                    low_corr = (corr.size - corr.shape[0]) / 2 - high_corr  # Subtract diagonal and high correlations
                    
                    # Pie chart for correlation distribution
                    ax2.pie([high_corr, low_corr], labels=['High Correlation', 'Low Correlation'], autopct='%1.1f%%', startangle=90)
                    ax2.set_title("Distribution of Correlations")
                    
                    plt.tight_layout()
                    return fig, (ax1, ax2)

                result = self.generate_plot(plot_alternative_analysis)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_alternative_network_analysis.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(("Alternative Network Analysis", img_path))
                else:
                    print("Skipping alternative analysis plot due to error in plot generation.")
                    if "error" not in results:
                        results["error"] = "Failed to generate alternative network analysis plot"
            else:
                results["error"] = "Not enough numerical columns for alternative network analysis"
                print("Not enough numerical columns for alternative analysis.")
            
            self.interpret_results("Alternative Network Analysis", results, table_name)
            return
        
        results['image_paths'] = image_paths
        self.interpret_results("Network Analysis for Fraud Detection", results, table_name)

    def sequence_alignment(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Sequence Alignment and Matching"))
        image_paths = []
        results = {}
        
        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            # Extract sequence alignment information for context
            try:
                # Select the first text column
                target_column = text_columns[0]
                sequences = df[target_column].dropna().head(10).tolist()  # Limit to 10 sequences for simplicity
                
                if len(sequences) < 2:
                    results["error"] = "Not enough sequences for alignment analysis"
                    print("Not enough sequences for alignment analysis.")
                else:
                    # Calculate all pairwise alignments
                    try:
                        # Try using Bio module first
                        from Bio import Align
                        
                        # Create PairwiseAligner object
                        aligner = Align.PairwiseAligner()
                        aligner.mode = 'global'
                        
                        # Perform pairwise alignments
                        alignments = []
                        for i in range(len(sequences)):
                            for j in range(i+1, len(sequences)):
                                try:
                                    alignment = aligner.align(sequences[i], sequences[j])
                                    alignments.append({
                                        "sequence_1": str(sequences[i]),
                                        "sequence_2": str(sequences[j]),
                                        "similarity_score": float(alignment.score),
                                        "alignment_method": "Biopython"
                                    })
                                except Exception:
                                    # If alignment fails (e.g., too long sequences), try a simpler approach
                                    similarity = 1 - abs(len(sequences[i]) - len(sequences[j])) / max(len(sequences[i]), len(sequences[j]))
                                    alignments.append({
                                        "sequence_1": str(sequences[i]),
                                        "sequence_2": str(sequences[j]),
                                        "similarity_score": float(similarity),
                                        "alignment_method": "Length-based"
                                    })
                        
                        # Find most similar and most different sequences
                        alignments_sorted = sorted(alignments, key=lambda x: x["similarity_score"], reverse=True)
                        most_similar = alignments_sorted[0] if alignments_sorted else None
                        most_different = alignments_sorted[-1] if alignments_sorted else None
                        
                        results["analysis_info"] = {
                            "target_column": str(target_column),
                            "sequence_alignment": {
                                "sequences_analyzed": len(sequences),
                                "pairwise_alignments": len(alignments),
                                "most_similar_pair": most_similar,
                                "most_different_pair": most_different,
                                "average_similarity": float(np.mean([a["similarity_score"] for a in alignments])) if alignments else None
                            }
                        }
                    except ImportError:
                        # If Bio module not available, use simple text similarity
                        similarities = []
                        for i in range(len(sequences)):
                            for j in range(i+1, len(sequences)):
                                similarity = 1 - abs(len(sequences[i]) - len(sequences[j])) / max(len(sequences[i]), len(sequences[j]))
                                similarities.append({
                                    "sequence_1": str(sequences[i]),
                                    "sequence_2": str(sequences[j]),
                                    "similarity_score": float(similarity),
                                    "alignment_method": "Length-based"
                                })
                        
                        # Find most similar and most different sequences
                        similarities_sorted = sorted(similarities, key=lambda x: x["similarity_score"], reverse=True)
                        most_similar = similarities_sorted[0] if similarities_sorted else None
                        most_different = similarities_sorted[-1] if similarities_sorted else None
                        
                        results["analysis_info"] = {
                            "target_column": str(target_column),
                            "sequence_alignment": {
                                "sequences_analyzed": len(sequences),
                                "pairwise_alignments": len(similarities),
                                "most_similar_pair": most_similar,
                                "most_different_pair": most_different,
                                "average_similarity": float(np.mean([s["similarity_score"] for s in similarities])) if similarities else None,
                                "method": "Length-based similarity (Bio module not available)"
                            }
                        }
            except Exception as e:
                results["error"] = f"Error in sequence alignment analysis: {str(e)}"
                print(f"Error extracting sequence alignment details: {str(e)}")
            
            def plot_sequence_alignment():
                try:
                    from Bio import Align
                    
                    # Select the first text column
                    sequences = df[text_columns[0]].dropna().head(10).tolist()  # Limit to 10 sequences for simplicity
                    
                    # Create PairwiseAligner object
                    aligner = Align.PairwiseAligner()
                    aligner.mode = 'global'
                    
                    # Perform pairwise alignments
                    alignments = []
                    for i in range(len(sequences)):
                        for j in range(i+1, len(sequences)):
                            alignment = aligner.align(sequences[i], sequences[j])
                            alignments.append((i, j, alignment.score))  # Store indices and alignment score
                    
                    # Create a similarity matrix
                    similarity_matrix = np.zeros((len(sequences), len(sequences)))
                    for i, j, score in alignments:
                        similarity_matrix[i, j] = similarity_matrix[j, i] = score
                    
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    im = ax.imshow(similarity_matrix, cmap='viridis')
                    ax.set_title("Sequence Alignment Similarity Matrix")
                    ax.set_xlabel("Sequence Index")
                    ax.set_ylabel("Sequence Index")
                    plt.colorbar(im, label='Alignment Score')
                    plt.tight_layout()
                    return fig, ax
                
                except ImportError:
                    print("Bio module not found. Performing alternative text analysis.")
                    
                    # Select the first text column
                    text_data = df[text_columns[0]].dropna().head(10).tolist()  # Limit to 10 texts for simplicity
                    
                    # Calculate simple text similarity based on character count difference
                    similarity_matrix = np.zeros((len(text_data), len(text_data)))
                    for i in range(len(text_data)):
                        for j in range(i+1, len(text_data)):
                            similarity = 1 - abs(len(text_data[i]) - len(text_data[j])) / max(len(text_data[i]), len(text_data[j]))
                            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
                    
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    im = ax.imshow(similarity_matrix, cmap='viridis')
                    ax.set_title("Text Similarity Matrix (Based on Length)")
                    ax.set_xlabel("Text Index")
                    ax.set_ylabel("Text Index")
                    plt.colorbar(im, label='Similarity Score')
                    plt.tight_layout()
                    return fig, ax

            result = self.generate_plot(plot_sequence_alignment)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_sequence_alignment.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Sequence Alignment", img_path))
            else:
                print("Skipping Sequence Alignment plot due to error in plot generation.")
                if "error" not in results:
                    results["error"] = "Failed to generate sequence alignment plot"
        else:
            results["error"] = "No text columns found for Sequence Alignment analysis"
            print("No text columns found for Sequence Alignment analysis.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Sequence Alignment and Matching", results, table_name)

    def conformal_anomaly_detection(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Conformal Anomaly Detection"))
        image_paths = []
        results = {}
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            # Extract conformal anomaly detection information for context
            try:
                # Make a simpler alternative implementation since nonconformist might not be available
                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import RandomForestRegressor
                
                X = df[numerical_columns].dropna()
                y = X.iloc[:, 0]  # Use the first column as the target
                X = X.iloc[:, 1:]  # Use the remaining columns as features
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Create and fit underlying model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate absolute errors
                errors = np.abs(y_test - y_pred)
                
                # Determine threshold for anomalies (e.g., errors > 90th percentile)
                threshold = np.percentile(errors, 90)
                anomalies = errors > threshold
                
                # Extract actual anomalies
                anomaly_examples = []
                for i, is_anomaly in enumerate(anomalies):
                    if is_anomaly:
                        example = {
                            "true_value": float(y_test.iloc[i]),
                            "predicted_value": float(y_pred[i]),
                            "absolute_error": float(errors[i]),
                            "features": {}
                        }
                        for col in X_test.columns:
                            example["features"][str(col)] = float(X_test.iloc[i][col])
                        anomaly_examples.append(example)
                
                results["analysis_info"] = {
                    "variables_analyzed": {
                        "target": str(numerical_columns[0]),
                        "features": numerical_columns[1:].tolist()
                    },
                    "conformal_analysis": {
                        "total_test_points": len(y_test),
                        "anomalies_detected": int(np.sum(anomalies)),
                        "anomaly_percentage": float(np.sum(anomalies) / len(y_test) * 100),
                        "error_threshold": float(threshold),
                        "anomaly_examples": anomaly_examples[:5]  # Limit to first 5 for brevity
                    }
                }
            except Exception as e:
                results["error"] = f"Error in conformal anomaly detection analysis: {str(e)}"
                print(f"Error extracting conformal anomaly detection details: {str(e)}")
            
            def plot_conformal_anomaly_detection():
                try:
                    # Make a simpler alternative implementation
                    from sklearn.model_selection import train_test_split
                    from sklearn.ensemble import RandomForestRegressor
                    
                    X = df[numerical_columns].dropna()
                    y = X.iloc[:, 0]  # Use the first column as the target
                    X = X.iloc[:, 1:]  # Use the remaining columns as features
                    
                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Create and fit underlying model
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate absolute errors
                    errors = np.abs(y_test - y_pred)
                    
                    # Determine threshold for anomalies (e.g., errors > 90th percentile)
                    threshold = np.percentile(errors, 90)
                    anomalies = errors > threshold
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                    
                    # Scatter plot of true vs predicted
                    ax1.scatter(y_test[~anomalies], y_pred[~anomalies], c='blue', alpha=0.5, label='Normal')
                    ax1.scatter(y_test[anomalies], y_pred[anomalies], c='red', alpha=0.5, label='Anomaly')
                    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                    ax1.set_xlabel("True Values")
                    ax1.set_ylabel("Predicted Values")
                    ax1.set_title("Conformal Anomaly Detection")
                    ax1.legend()
                    
                    # Pie chart of anomalies
                    ax2.pie([len(y_test) - sum(anomalies), sum(anomalies)], 
                           labels=['Normal', 'Anomaly'], 
                           autopct='%1.1f%%')
                    ax2.set_title("Distribution of Anomalies")
                    
                    plt.tight_layout()
                    return fig, (ax1, ax2)
                except Exception as e:
                    print(f"Error in conformal anomaly detection plot: {str(e)}")
                    return None

            result = self.generate_plot(plot_conformal_anomaly_detection)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_conformal_anomaly_detection.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Conformal Anomaly Detection", img_path))
            else:
                print("Skipping Conformal Anomaly Detection plot due to error in plot generation.")
                if "error" not in results:
                    results["error"] = "Failed to generate conformal anomaly detection plot"
        else:
            results["error"] = "Not enough numerical columns for Conformal Anomaly Detection analysis"
            print("Not enough numerical columns for Conformal Anomaly Detection analysis.")
        
        results['image_paths'] = image_paths
        self.interpret_results("Conformal Anomaly Detection", results, table_name)

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
        output_file = os.path.join(self.output_folder, "axda_b4_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        report_title = f"Advanced Exploratory Data Analysis (Batch 4) Report for {self.table_name}"
        
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
            filename=f"axda_b4_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
            return pdf_file
        else:
            print(error("Failed to generate PDF report"))
            return None