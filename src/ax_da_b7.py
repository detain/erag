# Standard library imports
import os
import sqlite3
import threading
import time
import signal
import sys
from functools import wraps
from datetime import datetime, timedelta
import warnings
import concurrent.futures

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
from scipy.signal import periodogram
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from pmdarima import auto_arima
import statsmodels.tsa.stattools

# Local imports
from src.api_model import EragAPI
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight
from src.print_pdf import PDFReportGenerator
from src.helper_da import get_technique_info

class TimeoutException(Exception):
    pass

class AdvancedExploratoryDataAnalysisB7:
    def __init__(self, worker_erag_api, supervisor_erag_api, db_path):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.db_path = db_path
        self.technique_counter = 0
        self.total_techniques = 16
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
        print(info("Describe the purpose, main tables, key data points, time periods, and any other relevant information:"))
        self.database_description = input("> ")
        print(success(f"Database description received: {self.database_description}"))

    def identify_entity_names(self, df):
        """
        Identify actual names of entities in the dataset instead of using generic names
        """
        entity_names = {}
        
        # Try to identify companies, products, regions, etc. in the time series data
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_values = df[col].unique()
            if 2 <= len(unique_values) <= 20:  # Reasonable number of categories
                entity_names[col.lower()] = unique_values.tolist()
        
        # Check for specific column names that might indicate entities
        for col in df.columns:
            col_lower = col.lower()
            if 'company' in col_lower or 'stock' in col_lower or 'ticker' in col_lower:
                entity_names['companies'] = df[col].unique().tolist()
            elif 'product' in col_lower or 'item' in col_lower:
                entity_names['products'] = df[col].unique().tolist()
            elif 'region' in col_lower or 'country' in col_lower or 'location' in col_lower:
                entity_names['regions'] = df[col].unique().tolist()
                
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
        print(info(f"Starting Advanced Exploratory Data Analysis (Batch 7) on {self.db_path}"))
        
        # Ask for database description before starting analysis
        self.prompt_for_database_description()
        
        tables = self.get_tables()
        for table in tables:
            self.analyze_table(table)
        
        self.save_text_output()
        self.generate_pdf_report()
        print(success(f"Advanced Exploratory Data Analysis (Batch 7) completed. Results saved in {self.output_folder}"))

    def preprocess_date_column(self, df):
        # Check if 'Date' column exists
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            date_col = date_cols[0]
        else:
            # Try to find and convert a date column
            potential_date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if potential_date_cols:
                date_col = potential_date_cols[0]
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    print(info(f"Converted column '{date_col}' to datetime format"))
                except:
                    raise ValueError("No date column could be converted to datetime format. Analysis cannot proceed.")
            else:
                raise ValueError("No 'Date' column found in the dataset. Analysis cannot proceed.")

        # Check if any conversion failed
        if df[date_col].isnull().any():
            print(warning("Some dates could not be converted to datetime format. These will be treated as missing values."))

        # Remove rows with null dates
        df = df.dropna(subset=[date_col])
        print(info(f"Using '{date_col}' as date column with {len(df)} valid rows"))

        # Sort the dataframe by date
        df = df.sort_values(date_col)

        # Check for duplicate dates
        if df[date_col].duplicated().any():
            print(warning(f"Duplicate dates found in '{date_col}'. Aggregating data by date."))
            # Group by date and aggregate
            # For numeric columns, take the mean
            # For non-numeric columns, take the first value
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            agg_dict = {col: 'mean' if col in numeric_columns else 'first' for col in df.columns if col != date_col}
            df = df.groupby(date_col).agg(agg_dict).reset_index()

        # Identify the most common frequency
        freq = pd.infer_freq(df[date_col])
        if freq is None:
            # If frequency can't be inferred, try to determine based on date differences
            date_diffs = df[date_col].diff().dropna()
            if len(date_diffs) > 0:
                most_common_diff = date_diffs.mode().iloc[0]
                if most_common_diff.days == 1:
                    freq = 'D'  # Daily
                elif 28 <= most_common_diff.days <= 31:
                    freq = 'M'  # Monthly
                elif 90 <= most_common_diff.days <= 92:
                    freq = 'Q'  # Quarterly
                elif 365 <= most_common_diff.days <= 366:
                    freq = 'Y'  # Yearly
                else:
                    freq = 'D'  # Default to daily if can't determine
            else:
                freq = 'D'  # Default to daily
            print(warning(f"Date frequency could not be inferred. Assuming {freq} frequency."))

        # Create a complete date range
        date_range = pd.date_range(start=df[date_col].min(), end=df[date_col].max(), freq=freq)

        # Reindex the dataframe with the complete date range
        df = df.set_index(date_col).reindex(date_range).reset_index()
        df = df.rename(columns={'index': date_col})

        print(info(f"Date column preprocessed. Data frequency: {freq}, Date range: {df[date_col].min()} to {df[date_col].max()}"))
        return df, date_col

    def analyze_table(self, table_name):
        self.table_name = table_name
        self.output_folder = os.path.join(settings.output_folder, f"axda_b7_{self.table_name}")
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, self.table_name)
        
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        
        # Load the full dataset
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            print(info(f"Loaded full dataset with {len(df)} rows and {len(df.columns)} columns"))
            
        if df.empty:
            error_message = f"The table {table_name} is empty."
            print(error(error_message))
            self.text_output += f"\n{error_message}\n"
            return

        # Preprocess the date column
        try:
            df, date_col = self.preprocess_date_column(df)
        except ValueError as e:
            print(error(str(e)))
            return

        # Identify entity names in the dataset
        self.entity_names_mapping = self.identify_entity_names(df)
        entity_description = self.format_entity_description()
        
        print(info(f"Identified entities in the data: {entity_description}"))

        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values using interpolation and forward/backward fill
        df = df.interpolate().fillna(method='bfill').fillna(method='ffill')
        
        analysis_methods = [
            self.arima_analysis,
            self.auto_arimax_analysis,
            self.exponential_smoothing,
            self.moving_average,
            self.linear_regression_trend,
            self.seasonal_decomposition_analysis,
            self.holt_winters_method,
            self.sarimax_analysis,
            self.gradient_boosting_time_series,
            self.lstm_time_series,
            self.fourier_analysis,
            self.trend_extraction,
            self.cross_sectional_regression,
            self.ensemble_time_series,
            self.bootstrapping_time_series,
            self.theta_method
        ]

        for method in analysis_methods:
            # Check if script is paused
            self.check_if_paused()
                
            try:
                method(df.copy(), table_name)
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

    @staticmethod
    def model_fit_with_timeout(model, timeout):
        def fit_func():
            return model.fit()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(fit_func)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                return None

    def arima_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - ARIMA Analysis"))
        image_paths = []
        arima_results = {}

        # Ensure we have a datetime column
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            date_col = date_cols[0]
            df.set_index(date_col, inplace=True)
        else:
            # If no datetime column, create a date range index
            df.index = pd.date_range(start='1/1/2000', periods=len(df))
            date_col = 'Date'

        # Ensure the date index has a frequency
        if df.index.freq is None:
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq is not None:
                df = df.asfreq(inferred_freq)
            else:
                # Default to daily frequency if cannot be inferred
                df = df.asfreq('D')
                print(warning("Date frequency could not be inferred. Assuming daily frequency."))

        # Get numeric columns for analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(info(f"Found {len(numeric_cols)} numeric columns for ARIMA analysis"))
        
        for col in numeric_cols:
            if df[col].isna().all():
                continue

            # Handle missing values
            df[col] = df[col].interpolate().bfill().ffill()

            try:
                # Determine optimal ARIMA parameters
                p, d, q = self.determine_arima_parameters(df[col])
                print(info(f"Optimal ARIMA parameters for {col}: ({p}, {d}, {q})"))

                # Fit the ARIMA model with a timeout
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    
                    model = ARIMA(df[col], order=(p, d, q))
                    results = self.model_fit_with_timeout(model, timeout=30)  # 30 seconds timeout

                    if results is None:
                        raise TimeoutError("ARIMA model fitting timed out")

                    # Check for convergence warning
                    if any("convergence" in str(warn.message).lower() for warn in w):
                        print(warning(f"Warning: ARIMA model for {col} did not converge. Trying alternative parameters."))
                        # Try alternative parameters
                        for alternative_order in [(1,1,1), (1,1,0), (0,1,1)]:
                            model = ARIMA(df[col], order=alternative_order)
                            results = self.model_fit_with_timeout(model, timeout=30)
                            if results is not None and not any("convergence" in str(warn.message).lower() for warn in w):
                                print(info(f"Alternative ARIMA parameters {alternative_order} converged for {col}"))
                                p, d, q = alternative_order
                                break
                        else:
                            raise ValueError("Could not find converging ARIMA parameters")

                def plot_arima():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df.index, df[col], label='Observed')
                    ax.plot(df.index, results.fittedvalues, color='red', label='ARIMA Fit')
                    ax.set_title(f'ARIMA Analysis: {col} (Order: {p},{d},{q})')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value')
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_arima)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_arima_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                # Perform forecasting
                forecast_steps = min(30, int(len(df) * 0.1))  # Forecast 10% of data length or 30 steps, whichever is smaller
                forecast = results.forecast(steps=forecast_steps)
                
                # Calculate error metrics
                mse = mean_squared_error(df[col][-forecast_steps:], results.fittedvalues[-forecast_steps:])
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(df[col][-forecast_steps:], results.fittedvalues[-forecast_steps:])
                
                # Generate forecast dates for better context
                last_date = df.index[-1]
                freq = df.index.freq or pd.infer_freq(df.index) or 'D'
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq=freq)
                
                arima_results[col] = {
                    'aic': float(results.aic),
                    'bic': float(results.bic),
                    'order': (p, d, q),
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'forecast_values': forecast.tolist(),
                    'forecast_dates': [date.strftime('%Y-%m-%d') for date in forecast_dates]
                }

            except Exception as e:
                print(error(f"Error in ARIMA analysis for column {col}: {str(e)}"))
                arima_results[col] = {'error': str(e)}

        results = {
            'image_paths': image_paths,
            'arima_results': arima_results,
            'date_column': date_col,
            'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            'data_frequency': str(df.index.freq) if df.index.freq else 'Unknown'
        }
        
        self.interpret_results("ARIMA Analysis", results, table_name)

    def determine_arima_parameters(self, series):
        # Determine 'd' (differencing term)
        d = 0
        while d < 2 and not self.is_stationary(series):
            series = series.diff().dropna()
            d += 1

        # Use auto_arima to determine optimal p and q
        try:
            model = auto_arima(series, d=d, start_p=0, start_q=0, max_p=5, max_q=5, 
                            seasonal=False, stepwise=True, suppress_warnings=True, 
                            error_action="ignore", max_order=None, trace=False)
            
            return model.order[0], d, model.order[2]
        except:
            # Fallback to default parameters if auto_arima fails
            return 1, d, 1

    def is_stationary(self, series):
        try:
            return statsmodels.tsa.stattools.adfuller(series, autolag='AIC')[1] <= 0.05
        except:
            return False
           
    def auto_arimax_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Auto ARIMAX Analysis"))
        image_paths = []
        
        # Ensure we have a datetime column
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            date_col = date_cols[0]
            df.set_index(date_col, inplace=True)
        else:
            # If no datetime column, create a date range index
            df.index = pd.date_range(start='1/1/2000', periods=len(df))
            date_col = 'Date'

        # Ensure the date index has a frequency
        if df.index.freq is None:
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq is not None:
                df = df.asfreq(inferred_freq)
            else:
                df = df.asfreq('D')

        # Get numeric columns for analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            print(warning("At least 2 numeric columns needed for ARIMAX analysis."))
            return
        
        auto_arimax_results = {}
        
        # For each numeric column, use other numeric columns as exogenous variables
        for target_col in numeric_cols:
            try:
                feature_cols = [col for col in numeric_cols if col != target_col]
                
                # Fill missing values
                df_filled = df.copy()
                for col in df_filled.columns:
                    df_filled[col] = df_filled[col].interpolate().bfill().ffill()
                
                # Split data into train and test
                train_size = int(len(df_filled) * 0.8)
                train = df_filled.iloc[:train_size]
                test = df_filled.iloc[train_size:]
                
                if len(train) < 10 or len(test) < 5:  # Ensure enough data
                    continue
                
                # Fit Auto ARIMA model
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = auto_arima(train[target_col], exogenous=train[feature_cols], 
                                      trace=False, error_action="ignore", suppress_warnings=True,
                                      max_p=3, max_q=3, max_d=2)
                
                # Get model order
                order = model.order
                seasonal_order = model.seasonal_order
                
                # Fit final model
                model.fit(train[target_col], exogenous=train[feature_cols])
                
                # Make predictions for test set
                predictions = model.predict(n_periods=len(test), exogenous=test[feature_cols])
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(test[target_col], predictions))
                mae = mean_absolute_error(test[target_col], predictions)
                
                # Plot results
                def plot_arimax():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df_filled.index, df_filled[target_col], label='Observed')
                    ax.plot(test.index, predictions, color='red', label='ARIMAX Predictions')
                    ax.axvline(x=train.index[-1], color='gray', linestyle='--', 
                              label='Train/Test Split')
                    ax.set_title(f'Auto ARIMAX: {target_col} (Order: {order}, Seasonal: {seasonal_order})')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value')
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax
                
                result = self.generate_plot(plot_arimax)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_auto_arimax_{target_col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)
                
                # Forecast future values
                future_steps = min(30, int(len(df) * 0.1))
                future_exog = test[feature_cols].iloc[-future_steps:].reset_index(drop=True)
                future_forecast = model.predict(n_periods=future_steps, exogenous=future_exog)
                
                auto_arimax_results[target_col] = {
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'exogenous_variables': feature_cols,
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'forecast': future_forecast.tolist()
                }
                
            except Exception as e:
                print(error(f"Error in Auto ARIMAX analysis for column {target_col}: {str(e)}"))
                auto_arimax_results[target_col] = {'error': str(e)}
        
        results = {
            'image_paths': image_paths,
            'auto_arimax_results': auto_arimax_results,
            'date_column': date_col,
            'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            'data_frequency': str(df.index.freq) if df.index.freq else 'Unknown'
        }
        
        self.interpret_results("Auto ARIMAX Analysis", results, table_name)
        
    def exponential_smoothing(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Exponential Smoothing"))
        image_paths = []
        exp_smoothing_results = {}

        # Prepare data
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            date_col = date_cols[0]
            df.set_index(date_col, inplace=True)
        else:
            df.index = pd.date_range(start='1/1/2000', periods=len(df))
            date_col = 'Date'

        if df.index.freq is None:
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq is not None:
                df = df.asfreq(inferred_freq)
            else:
                df = df.asfreq('D')

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(info(f"Found {len(numeric_cols)} numeric columns for Exponential Smoothing analysis"))
        
        for col in numeric_cols:
            if df[col].isna().all():
                continue

            df[col] = df[col].interpolate().bfill().ffill()

            try:
                # Determine if we should use seasonal exponential smoothing
                # Check if we have enough data points (at least 2 seasons)
                seasonal = False
                seasonal_periods = 1
                
                if len(df) >= 24:  # Need enough data for seasonal analysis
                    # Try to determine seasonality
                    if df.index.freq == 'D' or str(df.index.freq).startswith('D'):
                        seasonal = True
                        seasonal_periods = 7  # Weekly seasonality for daily data
                    elif df.index.freq == 'M' or str(df.index.freq).startswith('M'):
                        seasonal = True
                        seasonal_periods = 12  # Yearly seasonality for monthly data
                    elif df.index.freq == 'Q' or str(df.index.freq).startswith('Q'):
                        seasonal = True
                        seasonal_periods = 4  # Yearly seasonality for quarterly data
                
                if seasonal:
                    print(info(f"Using seasonal exponential smoothing for {col} with period {seasonal_periods}"))
                    model = ExponentialSmoothing(
                        df[col], 
                        seasonal_periods=seasonal_periods,
                        trend='add',
                        seasonal='add'
                    )
                else:
                    print(info(f"Using simple exponential smoothing for {col}"))
                    model = ExponentialSmoothing(df[col])
                
                results = self.model_fit_with_timeout(model, timeout=30)

                if results is None:
                    raise TimeoutError("Exponential Smoothing model fitting timed out")

                def plot_exp_smoothing():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df.index, df[col], label='Observed')
                    ax.plot(df.index, results.fittedvalues, color='red', label='Exponential Smoothing')
                    ax.set_title(f'Exponential Smoothing: {col}' + 
                               (f' (Seasonal: {seasonal_periods})' if seasonal else ''))
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value')
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_exp_smoothing)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_exp_smoothing_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                # Generate forecast
                forecast_steps = min(30, int(len(df) * 0.1))
                forecast = results.forecast(steps=forecast_steps)
                
                # Generate forecast dates
                last_date = df.index[-1]
                freq = df.index.freq or pd.infer_freq(df.index) or 'D'
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq=freq)
                
                # Calculate forecast plot
                def plot_exp_smoothing_forecast():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df.index, df[col], label='Observed')
                    ax.plot(df.index, results.fittedvalues, color='red', label='Fitted')
                    ax.plot(forecast_dates, forecast, color='green', label='Forecast')
                    ax.set_title(f'Exponential Smoothing Forecast: {col}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value')
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax
                
                forecast_result = self.generate_plot(plot_exp_smoothing_forecast)
                if forecast_result is not None:
                    fig, _ = forecast_result
                    img_path = os.path.join(self.output_folder, f"{table_name}_exp_smoothing_forecast_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)
                
                exp_smoothing_results[col] = {
                    'aic': float(results.aic),
                    'bic': float(results.bic),
                    'mse': float(mean_squared_error(df[col][-forecast_steps:], results.fittedvalues[-forecast_steps:])),
                    'seasonal': seasonal,
                    'seasonal_periods': seasonal_periods if seasonal else None,
                    'forecast_values': forecast.tolist(),
                    'forecast_dates': [date.strftime('%Y-%m-%d') for date in forecast_dates]
                }

            except Exception as e:
                print(error(f"Error in Exponential Smoothing analysis for column {col}: {str(e)}"))
                exp_smoothing_results[col] = {'error': str(e)}

        results = {
            'image_paths': image_paths,
            'exp_smoothing_results': exp_smoothing_results,
            'date_column': date_col,
            'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            'data_frequency': str(df.index.freq) if df.index.freq else 'Unknown'
        }
        
        self.interpret_results("Exponential Smoothing", results, table_name)

    def moving_average(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Moving Average"))
        image_paths = []
        moving_average_results = {}

        try:
            # Prepare data
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                date_col = date_cols[0]
                df.set_index(date_col, inplace=True)
            else:
                df.index = pd.date_range(start='1/1/2000', periods=len(df))
                date_col = 'Date'

            if df.index.freq is None:
                inferred_freq = pd.infer_freq(df.index)
                if inferred_freq is not None:
                    df = df.asfreq(inferred_freq)
                else:
                    df = df.asfreq('D')

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            print(info(f"Found {len(numeric_cols)} numeric columns for Moving Average analysis"))

            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in the dataset")

            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[col].isna().all():
                    continue

                df[col] = df[col].interpolate().bfill().ffill()

                # Calculate window sizes based on data frequency
                if df.index.freq == 'D' or str(df.index.freq).startswith('D'):
                    window_sizes = [7, 30, 90]  # Weekly, monthly, quarterly for daily data
                    window_labels = ['Weekly', 'Monthly', 'Quarterly']
                elif df.index.freq == 'M' or str(df.index.freq).startswith('M'):
                    window_sizes = [3, 6, 12]  # Quarterly, semi-annual, annual for monthly data
                    window_labels = ['Quarterly', 'Semi-Annual', 'Annual']
                elif df.index.freq == 'Q' or str(df.index.freq).startswith('Q'):
                    window_sizes = [2, 4, 8]  # Semi-annual, annual, bi-annual for quarterly data
                    window_labels = ['Semi-Annual', 'Annual', 'Bi-Annual']
                else:
                    # Default window sizes
                    window_sizes = [
                        min(7, len(df) // 4),
                        min(30, len(df) // 2),
                        min(90, len(df) - 1)
                    ]
                    window_labels = ['Short', 'Medium', 'Long']
                
                # Filter out window sizes larger than the dataset
                valid_windows = [(size, label) for size, label in zip(window_sizes, window_labels) if size < len(df)]
                if not valid_windows:
                    # Use a single window size if none of the predefined ones work
                    valid_windows = [(max(2, len(df) // 2), 'Half')]
                
                window_sizes, window_labels = zip(*valid_windows)
                print(info(f"Using window sizes {window_sizes} for {col}"))
                
                # Calculate moving averages
                moving_avgs = {}
                for window_size, window_label in zip(window_sizes, window_labels):
                    ma = df[col].rolling(window=window_size).mean()
                    moving_avgs[window_label] = ma

                def plot_moving_average():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df.index, df[col], label='Original', alpha=0.5)
                    
                    for window_label, ma in moving_avgs.items():
                        ax.plot(df.index, ma, label=f'{window_label} MA')
                    
                    ax.set_title(f'Moving Average Analysis: {col}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value')
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_moving_average)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_moving_average_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                ma_results = {}
                for window_label, ma in moving_avgs.items():
                    last_value = ma.iloc[-1] if not ma.empty else None
                    trend_direction = "up" if ma.iloc[-1] > ma.iloc[-min(10, len(ma))] else "down" if ma.iloc[-1] < ma.iloc[-min(10, len(ma))] else "flat"
                    
                    ma_results[window_label] = {
                        'window_size': window_sizes[window_labels.index(window_label)],
                        'last_value': float(last_value) if last_value is not None else None,
                        'trend_direction': trend_direction
                    }
                
                moving_average_results[col] = ma_results

            if not moving_average_results:
                raise ValueError("No valid data for moving average analysis")

        except Exception as e:
            print(error(f"Error in Moving Average analysis: {str(e)}"))
            moving_average_results = {'error': str(e)}

        results = {
            'image_paths': image_paths,
            'moving_average_results': moving_average_results,
            'date_column': date_col,
            'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            'data_frequency': str(df.index.freq) if df.index.freq else 'Unknown'
        }
        
        self.interpret_results("Moving Average Analysis", results, table_name)

    def linear_regression_trend(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Linear Regression Trend"))
        image_paths = []
        linear_trend_results = {}

        try:
            # Prepare data
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                date_col = date_cols[0]
                df.set_index(date_col, inplace=True)
            else:
                df.index = pd.date_range(start='1/1/2000', periods=len(df))
                date_col = 'Date'

            if df.index.freq is None:
                inferred_freq = pd.infer_freq(df.index)
                if inferred_freq is not None:
                    df = df.asfreq(inferred_freq)
                else:
                    df = df.asfreq('D')

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            print(info(f"Found {len(numeric_cols)} numeric columns for Linear Regression Trend analysis"))

            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in the dataset")

            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[col].isna().all():
                    continue

                df[col] = df[col].interpolate().bfill().ffill()

                # Create a feature for days since start
                days_since_start = (df.index - df.index[0]).days.values.reshape(-1, 1)
                y = df[col].values

                model = LinearRegression()
                model.fit(days_since_start, y)

                # Generate predictions
                trend = model.predict(days_since_start)

                # Calculate forecast
                forecast_steps = min(30, int(len(df) * 0.1))
                last_day = days_since_start[-1][0]
                forecast_days = np.array(range(last_day + 1, last_day + forecast_steps + 1)).reshape(-1, 1)
                forecast = model.predict(forecast_days)
                
                # Generate forecast dates
                last_date = df.index[-1]
                freq = df.index.freq or pd.infer_freq(df.index) or 'D'
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq=freq)

                def plot_linear_trend():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df.index, y, label='Original')
                    ax.plot(df.index, trend, color='red', label='Linear Trend')
                    ax.plot(forecast_dates, forecast, color='green', linestyle='--', label='Forecast')
                    ax.set_title(f'Linear Regression Trend: {col}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value')
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_linear_trend)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_linear_trend_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                linear_trend_results[col] = {
                    'slope': float(model.coef_[0]),
                    'intercept': float(model.intercept_),
                    'r_squared': float(model.score(days_since_start, y)),
                    'trend_direction': 'upward' if model.coef_[0] > 0 else 'downward' if model.coef_[0] < 0 else 'flat',
                    'daily_change': float(model.coef_[0]),
                    'monthly_change': float(model.coef_[0] * 30),  # Approximation
                    'annual_change': float(model.coef_[0] * 365),  # Approximation
                    'forecast_values': forecast.tolist(),
                    'forecast_dates': [date.strftime('%Y-%m-%d') for date in forecast_dates]
                }

            if not linear_trend_results:
                raise ValueError("No valid data for linear regression trend analysis")

        except Exception as e:
            print(error(f"Error in Linear Regression Trend analysis: {str(e)}"))
            linear_trend_results = {'error': str(e)}

        results = {
            'image_paths': image_paths,
            'linear_trend_results': linear_trend_results,
            'date_column': date_col,
            'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            'data_frequency': str(df.index.freq) if df.index.freq else 'Unknown'
        }
        
        self.interpret_results("Linear Regression Trend", results, table_name)

    def seasonal_decomposition_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Seasonal Decomposition"))
        image_paths = []
        seasonal_decomposition_results = {}

        try:
            # Prepare data
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                date_col = date_cols[0]
                df.set_index(date_col, inplace=True)
            else:
                df.index = pd.date_range(start='1/1/2000', periods=len(df))
                date_col = 'Date'

            if df.index.freq is None:
                inferred_freq = pd.infer_freq(df.index)
                if inferred_freq is not None:
                    df = df.asfreq(inferred_freq)
                else:
                    df = df.asfreq('D')

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            print(info(f"Found {len(numeric_cols)} numeric columns for Seasonal Decomposition analysis"))

            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in the dataset")

            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[col].isna().all():
                    continue

                df[col] = df[col].interpolate().bfill().ffill()

                # Ensure we have enough data for seasonal decomposition
                if len(df) < 2:
                    raise ValueError(f"Not enough data points for seasonal decomposition in column {col}")

                # Determine the period for seasonal decomposition
                if df.index.freq == 'D' or str(df.index.freq).startswith('D'):
                    period = 7  # Weekly seasonality for daily data
                    period_label = "Weekly"
                elif df.index.freq == 'M' or str(df.index.freq).startswith('M'):
                    period = 12  # Yearly seasonality for monthly data
                    period_label = "Yearly"
                elif df.index.freq == 'Q' or str(df.index.freq).startswith('Q'):
                    period = 4  # Yearly seasonality for quarterly data
                    period_label = "Yearly"
                else:
                    period = min(7, len(df) // 2)  # Default to 7 or half the data length if small
                    period_label = "Default"
                
                print(info(f"Using {period} ({period_label}) period for seasonal decomposition of {col}"))

                # Perform seasonal decomposition
                result = seasonal_decompose(df[col], model='additive', period=period)

                def plot_seasonal_decomposition():
                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(self.calculate_figure_size()[0], self.calculate_figure_size()[1]*2))
                    result.observed.plot(ax=ax1)
                    ax1.set_title('Observed')
                    result.trend.plot(ax=ax2)
                    ax2.set_title('Trend')
                    result.seasonal.plot(ax=ax3)
                    ax3.set_title(f'Seasonal (Period: {period})')
                    result.resid.plot(ax=ax4)
                    ax4.set_title('Residual')
                    plt.tight_layout()
                    return fig, (ax1, ax2, ax3, ax4)

                result_plot = self.generate_plot(plot_seasonal_decomposition)
                if result_plot is not None:
                    fig, _ = result_plot
                    img_path = os.path.join(self.output_folder, f"{table_name}_seasonal_decomposition_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                # Calculate seasonal strength metrics
                observed_var = np.var(result.observed)
                trend_var = np.var(result.trend[~np.isnan(result.trend)])
                seasonal_var = np.var(result.seasonal[~np.isnan(result.seasonal)])
                resid_var = np.var(result.resid[~np.isnan(result.resid)])
                
                # Find peak seasonal periods
                seasonal_series = pd.Series(result.seasonal)
                seasonal_strength = seasonal_var / observed_var if observed_var > 0 else 0
                
                # For daily data, identify day of week patterns
                day_of_week_effect = None
                if df.index.freq == 'D' or str(df.index.freq).startswith('D'):
                    # Group by day of week and calculate mean seasonal effect
                    day_of_week_effect = seasonal_series.groupby(seasonal_series.index.dayofweek).mean().to_dict()
                
                # For monthly data, identify month of year patterns
                month_of_year_effect = None
                if df.index.freq == 'M' or str(df.index.freq).startswith('M'):
                    # Group by month and calculate mean seasonal effect
                    month_of_year_effect = seasonal_series.groupby(seasonal_series.index.month).mean().to_dict()

                seasonal_decomposition_results[col] = {
                    'period': period,
                    'period_label': period_label,
                    'trend_strength': float(trend_var / observed_var) if observed_var > 0 else 0,
                    'seasonality_strength': float(seasonal_strength),
                    'residual_strength': float(resid_var / observed_var) if observed_var > 0 else 0,
                    'day_of_week_effect': day_of_week_effect,
                    'month_of_year_effect': month_of_year_effect,
                    'has_significant_seasonality': seasonal_strength > 0.3
                }

            if not seasonal_decomposition_results:
                raise ValueError("No valid data for seasonal decomposition analysis")

        except Exception as e:
            print(error(f"Error in Seasonal Decomposition analysis: {str(e)}"))
            seasonal_decomposition_results = {'error': str(e)}

        results = {
            'image_paths': image_paths,
            'seasonal_decomposition_results': seasonal_decomposition_results,
            'date_column': date_col,
            'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            'data_frequency': str(df.index.freq) if df.index.freq else 'Unknown'
        }
        
        self.interpret_results("Seasonal Decomposition", results, table_name)

    def holt_winters_method(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Holt-Winters Method"))
        image_paths = []
        holt_winters_results = {}
        
        # Prepare data
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            date_col = date_cols[0]
            df.set_index(date_col, inplace=True)
        else:
            df.index = pd.date_range(start='1/1/2000', periods=len(df))
            date_col = 'Date'

        if df.index.freq is None:
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq is not None:
                df = df.asfreq(inferred_freq)
            else:
                df = df.asfreq('D')

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(info(f"Found {len(numeric_cols)} numeric columns for Holt-Winters analysis"))
        
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            if df[col].isna().all():
                continue

            df[col] = df[col].interpolate().bfill().ffill()

            try:
                # Determine seasonal period based on data frequency
                if df.index.freq == 'D' or str(df.index.freq).startswith('D'):
                    seasonal_periods = 7  # Weekly seasonality for daily data
                elif df.index.freq == 'M' or str(df.index.freq).startswith('M'):
                    seasonal_periods = 12  # Yearly seasonality for monthly data
                elif df.index.freq == 'Q' or str(df.index.freq).startswith('Q'):
                    seasonal_periods = 4  # Yearly seasonality for quarterly data
                else:
                    # Default to 7 or a reasonable period based on data length
                    seasonal_periods = min(7, len(df) // 4)
                
                # Need at least 2 full seasonal periods of data
                if len(df) < 2 * seasonal_periods:
                    print(warning(f"Not enough data for seasonal Holt-Winters on {col}. Using non-seasonal model."))
                    model = ExponentialSmoothing(df[col], trend='add')
                    seasonal = False
                else:
                    print(info(f"Using seasonal Holt-Winters for {col} with period {seasonal_periods}"))
                    model = ExponentialSmoothing(
                        df[col], 
                        seasonal_periods=seasonal_periods,
                        trend='add',
                        seasonal='add'
                    )
                    seasonal = True
                
                results = self.model_fit_with_timeout(model, timeout=30)

                if results is None:
                    raise TimeoutError("Holt-Winters model fitting timed out")

                def plot_holt_winters():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df.index, df[col], label='Observed')
                    ax.plot(df.index, results.fittedvalues, color='red', label='Holt-Winters')
                    ax.set_title(f'Holt-Winters Method: {col}' +
                               (f' (Seasonal Period: {seasonal_periods})' if seasonal else ''))
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value')
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_holt_winters)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_holt_winters_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                # Calculate forecast
                forecast_steps = min(30, int(len(df) * 0.1))
                forecast = results.forecast(steps=forecast_steps)
                
                # Generate forecast dates
                last_date = df.index[-1]
                freq = df.index.freq or pd.infer_freq(df.index) or 'D'
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq=freq)
                
                # Plot forecast
                def plot_holt_winters_forecast():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df.index, df[col], label='Observed')
                    ax.plot(df.index, results.fittedvalues, color='red', label='Fitted')
                    ax.plot(forecast_dates, forecast, color='green', label='Forecast')
                    ax.set_title(f'Holt-Winters Forecast: {col}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value')
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax
                
                forecast_result = self.generate_plot(plot_holt_winters_forecast)
                if forecast_result is not None:
                    fig, _ = forecast_result
                    img_path = os.path.join(self.output_folder, f"{table_name}_holt_winters_forecast_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)
                
                # Calculate error metrics
                mse = mean_squared_error(df[col], results.fittedvalues)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(df[col], results.fittedvalues)
                
                holt_winters_results[col] = {
                    'aic': float(results.aic),
                    'bic': float(results.bic),
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'seasonal': seasonal,
                    'seasonal_periods': seasonal_periods if seasonal else None,
                    'forecast_values': forecast.tolist(),
                    'forecast_dates': [date.strftime('%Y-%m-%d') for date in forecast_dates],
                    'smoothing_level': float(results.params['smoothing_level']),
                    'smoothing_trend': float(results.params['smoothing_trend']) if 'smoothing_trend' in results.params else None,
                    'smoothing_seasonal': float(results.params['smoothing_seasonal']) if 'smoothing_seasonal' in results.params else None
                }

            except Exception as e:
                print(error(f"Error in Holt-Winters analysis for column {col}: {str(e)}"))
                holt_winters_results[col] = {'error': str(e)}

        results = {
            'image_paths': image_paths,
            'holt_winters_results': holt_winters_results,
            'date_column': date_col,
            'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            'data_frequency': str(df.index.freq) if df.index.freq else 'Unknown'
        }
        
        self.interpret_results("Holt-Winters Method", results, table_name)

    def gradient_boosting_time_series(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Gradient Boosting for Time Series"))
        image_paths = []
        gb_results = {}

        try:
            # Prepare data
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                date_col = date_cols[0]
                df.set_index(date_col, inplace=True)
            else:
                df.index = pd.date_range(start='1/1/2000', periods=len(df))
                date_col = 'Date'

            if df.index.freq is None:
                inferred_freq = pd.infer_freq(df.index)
                if inferred_freq is not None:
                    df = df.asfreq(inferred_freq)
                else:
                    df = df.asfreq('D')

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            print(info(f"Found {len(numeric_cols)} numeric columns for Gradient Boosting analysis"))

            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in the dataset")

            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[col].isna().all():
                    continue

                df[col] = df[col].interpolate().bfill().ffill()

                series = df[col]
                
                # Create lag features for time series forecasting
                n_lags = min(10, len(series) // 5)  # Use up to 10 lags
                X = pd.DataFrame(index=series.index)
                
                # Add time-based features
                X['dayofyear'] = X.index.dayofyear
                X['month'] = X.index.month
                X['dayofweek'] = X.index.dayofweek
                X['quarter'] = X.index.quarter
                
                # Add lag features
                for i in range(1, n_lags + 1):
                    X[f'lag_{i}'] = series.shift(i)
                
                # Drop rows with NaN values (from lagging)
                X = X.dropna()
                y = series.loc[X.index]
                
                if len(X) < 2:
                    raise ValueError(f"Not enough data points for Gradient Boosting in column {col}")

                # Split data into train and test sets
                train_size = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
                y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

                # Train the model
                model = GradientBoostingRegressor(random_state=42)
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)
                
                # Evaluate performance
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)

                def plot_gradient_boosting():
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.calculate_figure_size()[0], self.calculate_figure_size()[1] * 1.5))
                    
                    # Plot full series with train/test split
                    ax1.plot(series.index, series, label='Original')
                    ax1.axvline(X_train.index[-1], color='black', linestyle='--', label='Train/Test Split')
                    ax1.set_title(f'Gradient Boosting Time Series: {col} - Full Data')
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Value')
                    ax1.legend()
                    
                    # Plot test set predictions
                    ax2.plot(y_test.index, y_test, label='Actual')
                    ax2.plot(y_test.index, y_pred, color='red', label='Predicted')
                    ax2.set_title(f'Gradient Boosting Time Series: {col} - Test Set')
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel('Value')
                    ax2.legend()
                    
                    plt.tight_layout()
                    return fig, (ax1, ax2)

                result = self.generate_plot(plot_gradient_boosting)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_gradient_boosting_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)
                
                # Calculate feature importance
                feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                
                # Plot feature importance
                def plot_feature_importance():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    feature_importance.plot(kind='bar', ax=ax)
                    ax.set_title(f'Feature Importance: {col}')
                    ax.set_xlabel('Feature')
                    ax.set_ylabel('Importance')
                    plt.tight_layout()
                    return fig, ax
                
                importance_result = self.generate_plot(plot_feature_importance)
                if importance_result is not None:
                    fig, _ = importance_result
                    img_path = os.path.join(self.output_folder, f"{table_name}_gradient_boosting_importance_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                gb_results[col] = {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'feature_importance': feature_importance.to_dict(),
                    'top_features': feature_importance.head(3).to_dict()
                }

            if not gb_results:
                raise ValueError("No valid data for Gradient Boosting analysis")

        except Exception as e:
            print(error(f"Error in Gradient Boosting analysis: {str(e)}"))
            gb_results = {'error': str(e)}

        results = {
            'image_paths': image_paths,
            'gb_results': gb_results,
            'date_column': date_col,
            'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            'data_frequency': str(df.index.freq) if df.index.freq else 'Unknown'
        }
        
        self.interpret_results("Gradient Boosting for Time Series", results, table_name)
    
    
    def lstm_time_series(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - SARIMAX Time Series (replacing LSTM)"))
        image_paths = []
        sarimax_results = {}

        try:
            # Prepare data
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                date_col = date_cols[0]
                df.set_index(date_col, inplace=True)
            else:
                df.index = pd.date_range(start='1/1/2000', periods=len(df))
                date_col = 'Date'

            if df.index.freq is None:
                inferred_freq = pd.infer_freq(df.index)
                if inferred_freq is not None:
                    df = df.asfreq(inferred_freq)
                else:
                    df = df.asfreq('D')

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            print(info(f"Found {len(numeric_cols)} numeric columns for SARIMAX analysis"))

            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in the dataset")

            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[col].isna().all():
                    continue

                df[col] = df[col].interpolate().bfill().ffill()
                
                # Determine seasonal period based on data frequency
                if df.index.freq == 'D' or str(df.index.freq).startswith('D'):
                    seasonal_periods = 7  # Weekly seasonality for daily data
                elif df.index.freq == 'M' or str(df.index.freq).startswith('M'):
                    seasonal_periods = 12  # Yearly seasonality for monthly data
                elif df.index.freq == 'Q' or str(df.index.freq).startswith('Q'):
                    seasonal_periods = 4  # Yearly seasonality for quarterly data
                else:
                    seasonal_periods = 1  # No seasonality as default
                
                # For small datasets, use simpler model
                if len(df) < 4 * seasonal_periods:
                    print(warning(f"Limited data for seasonal modeling. Using simpler SARIMAX model for {col}"))
                    order = (1, 1, 1)
                    seasonal_order = (0, 0, 0, 0)
                else:
                    order = (1, 1, 1)
                    seasonal_order = (1, 1, 1, seasonal_periods)

                # Fit SARIMAX model with timeout
                try:
                    model = SARIMAX(df[col], order=order, seasonal_order=seasonal_order)
                    results = self.model_fit_with_timeout(model, timeout=30)
                    
                    if results is None:
                        # Try simpler model if timeout
                        print(warning(f"SARIMAX model timed out. Trying simpler model for {col}"))
                        model = SARIMAX(df[col], order=(1, 1, 0), seasonal_order=(0, 0, 0, 0))
                        results = self.model_fit_with_timeout(model, timeout=30)
                        
                        if results is None:
                            raise TimeoutError("SARIMAX model fitting timed out")
                except:
                    # Try one more time with even simpler model
                    model = SARIMAX(df[col], order=(1, 1, 0), seasonal_order=(0, 0, 0, 0))
                    results = self.model_fit_with_timeout(model, timeout=30)
                    
                    if results is None:
                        raise TimeoutError("SARIMAX model fitting timed out")

                # Generate in-sample predictions
                predictions = results.predict(start=0, end=-1)

                # Generate out-of-sample forecast
                forecast_steps = min(30, int(len(df) * 0.1))
                forecast = results.forecast(steps=forecast_steps)
                
                # Generate forecast dates
                last_date = df.index[-1]
                freq = df.index.freq or pd.infer_freq(df.index) or 'D'
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq=freq)

                def plot_sarimax():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df.index, df[col], label='Observed')
                    ax.plot(df.index, predictions, color='red', label='SARIMAX')
                    ax.plot(forecast_dates, forecast, color='green', label='Forecast')
                    ax.set_title(f'SARIMAX Time Series: {col} (Order: {order}, Seasonal: {seasonal_order})')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value')
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_sarimax)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_sarimax_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                # Calculate error metrics
                mse = mean_squared_error(df[col], predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(df[col], predictions)
                
                sarimax_results[col] = {
                    'aic': float(results.aic),
                    'bic': float(results.bic),
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'forecast_values': forecast.tolist(),
                    'forecast_dates': [date.strftime('%Y-%m-%d') for date in forecast_dates]
                }

            if not sarimax_results:
                raise ValueError("No valid data for SARIMAX analysis")

        except Exception as e:
            print(error(f"Error in SARIMAX analysis: {str(e)}"))
            sarimax_results = {'error': str(e)}

        results = {
            'image_paths': image_paths,
            'sarimax_results': sarimax_results,
            'date_column': date_col,
            'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            'data_frequency': str(df.index.freq) if df.index.freq else 'Unknown'
        }
        
        self.interpret_results("SARIMAX Time Series (replacing LSTM)", results, table_name)

    def fourier_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Fourier Analysis"))
        image_paths = []
        fourier_results = {}

        try:
            # Prepare data
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                date_col = date_cols[0]
                df.set_index(date_col, inplace=True)
            else:
                df.index = pd.date_range(start='1/1/2000', periods=len(df))
                date_col = 'Date'

            if df.index.freq is None:
                inferred_freq = pd.infer_freq(df.index)
                if inferred_freq is not None:
                    df = df.asfreq(inferred_freq)
                else:
                    df = df.asfreq('D')

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            print(info(f"Found {len(numeric_cols)} numeric columns for Fourier analysis"))

            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in the dataset")

            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[col].isna().all():
                    continue

                df[col] = df[col].interpolate().bfill().ffill()

                series = df[col].values
                if len(series) < 10:  # Ensure we have enough data points for Fourier analysis
                    print(warning(f"Not enough data points for Fourier analysis in column {col}"))
                    continue

                # Compute periodogram
                f, Pxx_den = periodogram(series)
                
                # Find the dominant frequencies - the indices of the peak values
                peak_indices = np.argsort(Pxx_den)[-5:][::-1]  # Top 5 peaks
                peak_frequencies = f[peak_indices]
                peak_powers = Pxx_den[peak_indices]
                
                # Determine if there are significant periodic components
                # (power of highest peak is significantly larger than median power)
                median_power = np.median(Pxx_den)
                has_periodicity = peak_powers[0] > median_power * 5
                
                # For daily data, convert frequencies to days
                if df.index.freq == 'D' or str(df.index.freq).startswith('D'):
                    if peak_frequencies[0] > 0:  # Avoid division by zero
                        cycle_period_days = 1 / peak_frequencies[0]
                    else:
                        cycle_period_days = 0
                else:
                    cycle_period_days = None

                def plot_fourier():
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                    
                    # Original time series
                    ax1.plot(df.index, series)
                    ax1.set_title(f'Time Series: {col}')
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Value')
                    
                    # Power spectral density
                    ax2.semilogy(f, Pxx_den)
                    for freq, power in zip(peak_frequencies, peak_powers):
                        ax2.scatter(freq, power, color='red')
                        if cycle_period_days and freq == peak_frequencies[0]:
                            ax2.annotate(f"{cycle_period_days:.1f} days", 
                                     xy=(freq, power), xytext=(10, 10),
                                     textcoords='offset points')
                    
                    ax2.set_title(f'Fourier Analysis: {col}')
                    ax2.set_xlabel('Frequency')
                    ax2.set_ylabel('Power Spectral Density')
                    
                    plt.tight_layout()
                    return fig, (ax1, ax2)

                result = self.generate_plot(plot_fourier)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_fourier_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                # Format results for interpretation
                peak_data = []
                for i, (freq, power) in enumerate(zip(peak_frequencies, peak_powers)):
                    period = 1/freq if freq > 0 else float('inf')
                    if df.index.freq == 'D' or str(df.index.freq).startswith('D'):
                        period_desc = f"{period:.1f} days"
                    elif df.index.freq == 'M' or str(df.index.freq).startswith('M'):
                        period_desc = f"{period:.1f} months"
                    else:
                        period_desc = f"{period:.1f} periods"
                        
                    peak_data.append({
                        'frequency': float(freq),
                        'power': float(power),
                        'period': float(period),
                        'period_description': period_desc,
                        'significance': float(power / median_power)
                    })

                fourier_results[col] = {
                    'dominant_frequency': float(peak_frequencies[0]) if len(peak_frequencies) > 0 else None,
                    'max_power': float(peak_powers[0]) if len(peak_powers) > 0 else None,
                    'has_significant_periodicity': has_periodicity,
                    'cycle_period_days': float(cycle_period_days) if cycle_period_days else None,
                    'peak_data': peak_data
                }

            if not fourier_results:
                raise ValueError("No valid data for Fourier analysis")

        except Exception as e:
            print(error(f"Error in Fourier analysis: {str(e)}"))
            fourier_results = {'error': str(e)}

        results = {
            'image_paths': image_paths,
            'fourier_results': fourier_results,
            'date_column': date_col,
            'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            'data_frequency': str(df.index.freq) if df.index.freq else 'Unknown'
        }
        
        self.interpret_results("Fourier Analysis", results, table_name)

    def trend_extraction(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Trend Extraction"))
        image_paths = []
        trend_results = {}

        try:
            # Prepare data
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                date_col = date_cols[0]
                df.set_index(date_col, inplace=True)
            else:
                df.index = pd.date_range(start='1/1/2000', periods=len(df))
                date_col = 'Date'

            if df.index.freq is None:
                inferred_freq = pd.infer_freq(df.index)
                if inferred_freq is not None:
                    df = df.asfreq(inferred_freq)
                else:
                    df = df.asfreq('D')

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            print(info(f"Found {len(numeric_cols)} numeric columns for Trend Extraction analysis"))

            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in the dataset")

            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[col].isna().all():
                    continue

                df[col] = df[col].interpolate().bfill().ffill()

                series = df[col].values
                if len(series) < 10:  # Ensure we have enough data points for trend extraction
                    print(warning(f"Not enough data points for trend extraction in column {col}"))
                    continue

                # Use HP filter to extract trend and cycle components
                try:
                    cycle, trend = hpfilter(series, lamb=1600)
                except:
                    # If HP filter fails, try a simple moving average for trend
                    window_size = min(30, len(series) // 3)
                    trend = pd.Series(series).rolling(window=window_size, center=True).mean().values
                    cycle = series - trend
                    
                # Calculate first and last trend values to determine overall direction
                trend_start = trend[~np.isnan(trend)][0]
                trend_end = trend[~np.isnan(trend)][-1]
                trend_direction = "upward" if trend_end > trend_start else "downward" if trend_end < trend_start else "flat"
                
                # Calculate growth rate
                if trend_start != 0:
                    growth_rate = ((trend_end - trend_start) / trend_start) * 100
                else:
                    growth_rate = np.nan

                def plot_trend():
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.calculate_figure_size()[0], self.calculate_figure_size()[1] * 1.5))
                    
                    # Plot original series with extracted trend
                    ax1.plot(df.index, series, label='Original')
                    ax1.plot(df.index, trend, color='red', label='Trend')
                    ax1.set_title(f'Trend Extraction: {col}')
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Value')
                    ax1.legend()
                    
                    # Plot cyclical component
                    ax2.plot(df.index, cycle, color='green')
                    ax2.set_title(f'Cyclical Component: {col}')
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel('Value')
                    ax2.axhline(y=0, color='black', linestyle='--')
                    
                    plt.tight_layout()
                    return fig, (ax1, ax2)

                result = self.generate_plot(plot_trend)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_trend_extraction_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                # Calculate various trend metrics
                trend_var = np.var(trend[~np.isnan(trend)])
                cycle_var = np.var(cycle[~np.isnan(cycle)])
                series_var = np.var(series)
                
                trend_strength = trend_var / series_var if series_var > 0 else 0
                cycle_strength = cycle_var / series_var if series_var > 0 else 0
                
                # Identify trend change points
                trend_diff = np.diff(trend[~np.isnan(trend)])
                sign_changes = np.where(np.diff(np.signbit(trend_diff)))[0]
                
                # Get change points in the original time index
                change_points = []
                if len(sign_changes) > 0:
                    valid_indices = np.where(~np.isnan(trend))[0]
                    change_indices = valid_indices[sign_changes + 1]  # +1 because diff reduces length by 1
                    change_points = df.index[change_indices].tolist()
                    # Convert to string for serialization
                    change_points = [point.strftime('%Y-%m-%d') for point in change_points]

                trend_results[col] = {
                    'trend_direction': trend_direction,
                    'growth_rate_pct': float(growth_rate) if not np.isnan(growth_rate) else None,
                    'trend_strength': float(trend_strength),
                    'cycle_strength': float(cycle_strength),
                    'has_strong_trend': trend_strength > 0.5,
                    'trend_change_points': change_points,
                    'num_trend_changes': len(change_points)
                }

            if not trend_results:
                raise ValueError("No valid data for trend extraction analysis")

        except Exception as e:
            print(error(f"Error in Trend Extraction analysis: {str(e)}"))
            trend_results = {'error': str(e)}

        results = {
            'image_paths': image_paths,
            'trend_results': trend_results,
            'date_column': date_col,
            'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            'data_frequency': str(df.index.freq) if df.index.freq else 'Unknown'
        }
        
        self.interpret_results("Trend Extraction", results, table_name)

    def cross_sectional_regression(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Cross-Sectional Regression"))
        image_paths = []
        xsection_results = {}
        
        try:
            # Identify numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            print(info(f"Found {len(numeric_cols)} numeric columns for Cross-Sectional Regression analysis"))
            
            if len(numeric_cols) < 2:
                raise ValueError("Not enough numeric columns for cross-sectional regression.")
            
            # Try to find a suitable target column
            # First look for columns that might represent output/performance
            potential_targets = [col for col in numeric_cols if any(
                term in col.lower() for term in 
                ['performance', 'output', 'result', 'target', 'sales', 'revenue', 'profit']
            )]
            
            # If no obvious targets, use the last numeric column
            if not potential_targets:
                target_col = numeric_cols[-1]
            else:
                target_col = potential_targets[0]
            
            # Feature columns are all numeric columns except the target
            feature_cols = [col for col in numeric_cols if col != target_col]
            
            # Fill missing values
            df_clean = df.copy()
            for col in numeric_cols:
                df_clean[col] = df_clean[col].interpolate().bfill().ffill()
            
            # Drop any remaining rows with NaN
            df_clean = df_clean.dropna(subset=numeric_cols)
            
            if len(df_clean) < 10:  # Need enough data points
                raise ValueError("Not enough data for cross-sectional regression after handling missing values.")
            
            # Fit linear regression model
            X = df_clean[feature_cols]
            y = df_clean[target_col]
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Make predictions
            predictions = model.predict(X)
            
            # Calculate performance metrics
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, predictions)
            r2 = model.score(X, y)
            
            # Get coefficients
            coefficients = {}
            for feature, coef in zip(feature_cols, model.coef_):
                coefficients[feature] = float(coef)
            
            # Identify most important features
            feature_importance = pd.Series(coefficients).abs().sort_values(ascending=False)
            
            def plot_cross_sectional():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                
                # Actual vs Predicted
                ax1.scatter(y, predictions, alpha=0.5)
                ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
                ax1.set_title('Actual vs Predicted Values')
                ax1.set_xlabel(f'Actual {target_col}')
                ax1.set_ylabel(f'Predicted {target_col}')
                
                # Feature importance
                top_features = feature_importance.head(min(5, len(feature_importance)))
                top_features.plot(kind='barh', ax=ax2)
                ax2.set_title('Top Feature Importance')
                ax2.set_xlabel('Absolute Coefficient Value')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_cross_sectional)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_cross_sectional_regression.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            
            # Store regression results
            xsection_results = {
                'target_column': target_col,
                'feature_columns': feature_cols,
                'coefficients': coefficients,
                'intercept': float(model.intercept_),
                'r_squared': float(r2),
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'top_features': feature_importance.head(5).to_dict(),
                'equation': f"{target_col} = {model.intercept_:.4f} + " + " + ".join([f"{coef:.4f}*{feat}" for feat, coef in zip(feature_cols, model.coef_)])
            }
            
        except Exception as e:
            print(error(f"Error in Cross-Sectional Regression analysis: {str(e)}"))
            xsection_results = {'error': str(e)}

        results = {
            'image_paths': image_paths,
            'cross_sectional_results': xsection_results
        }
        
        self.interpret_results("Cross-Sectional Regression", results, table_name)

    def ensemble_time_series(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Ensemble Time Series"))
        image_paths = []
        ensemble_results = {}
        
        try:
            # Prepare data
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                date_col = date_cols[0]
                df.set_index(date_col, inplace=True)
            else:
                df.index = pd.date_range(start='1/1/2000', periods=len(df))
                date_col = 'Date'

            if df.index.freq is None:
                inferred_freq = pd.infer_freq(df.index)
                if inferred_freq is not None:
                    df = df.asfreq(inferred_freq)
                else:
                    df = df.asfreq('D')

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            print(info(f"Found {len(numeric_cols)} numeric columns for Ensemble Time Series analysis"))
            
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[col].isna().all():
                    continue

                df[col] = df[col].interpolate().bfill().ffill()
                
                if len(df) < 20:  # Need sufficient data for multiple models
                    print(warning(f"Insufficient data for ensemble modeling on {col}"))
                    continue
                
                # Split data for training and evaluation
                train_size = int(len(df) * 0.8)
                train = df.iloc[:train_size]
                test = df.iloc[train_size:]
                
                series = df[col]
                train_series = train[col]
                test_series = test[col]
                
                # Initialize model forecasts dictionary
                forecasts = {}
                
                # 1. ARIMA model
                try:
                    arima_model = ARIMA(train_series, order=(1, 1, 1))
                    arima_results = arima_model.fit()
                    arima_pred = arima_results.predict(start=len(train), end=len(df)-1)
                    forecasts['ARIMA'] = arima_pred
                except Exception as e:
                    print(warning(f"ARIMA model failed: {str(e)}"))
                
                # 2. Exponential Smoothing
                try:
                    exp_model = ExponentialSmoothing(train_series)
                    exp_results = exp_model.fit()
                    exp_pred = exp_results.forecast(len(test))
                    forecasts['ExponentialSmoothing'] = exp_pred
                except Exception as e:
                    print(warning(f"Exponential Smoothing model failed: {str(e)}"))
                
                # 3. SARIMAX model - only if we have enough data
                if len(train) >= 30:
                    try:
                        sarimax_model = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                        sarimax_results = sarimax_model.fit()
                        sarimax_pred = sarimax_results.predict(start=len(train), end=len(df)-1)
                        forecasts['SARIMAX'] = sarimax_pred
                    except Exception as e:
                        print(warning(f"SARIMAX model failed: {str(e)}"))
                
                # Create ensemble forecast if we have at least 2 models
                if len(forecasts) >= 2:
                    # Convert forecasts to DataFrame for easier handling
                    forecast_df = pd.DataFrame(forecasts)
                    # Create simple average ensemble
                    forecast_df['Ensemble'] = forecast_df.mean(axis=1)
                    
                    # Calculate performance metrics for each model
                    model_metrics = {}
                    for model_name, forecast in forecasts.items():
                        mse = mean_squared_error(test_series, forecast)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(test_series, forecast)
                        model_metrics[model_name] = {
                            'mse': float(mse),
                            'rmse': float(rmse),
                            'mae': float(mae)
                        }
                    
                    # Calculate metrics for ensemble
                    ensemble_mse = mean_squared_error(test_series, forecast_df['Ensemble'])
                    ensemble_rmse = np.sqrt(ensemble_mse)
                    ensemble_mae = mean_absolute_error(test_series, forecast_df['Ensemble'])
                    model_metrics['Ensemble'] = {
                        'mse': float(ensemble_mse),
                        'rmse': float(ensemble_rmse),
                        'mae': float(ensemble_mae)
                    }
                    
                    # Generate future forecast steps
                    forecast_steps = min(30, int(len(df) * 0.1))
                    future_forecasts = {}
                    
                    # Get forecasts from each model
                    if 'ARIMA' in forecasts:
                        future_forecasts['ARIMA'] = arima_results.forecast(steps=forecast_steps)
                    
                    if 'ExponentialSmoothing' in forecasts:
                        future_forecasts['ExponentialSmoothing'] = exp_results.forecast(steps=forecast_steps)
                    
                    if 'SARIMAX' in forecasts:
                        future_forecasts['SARIMAX'] = sarimax_results.forecast(steps=forecast_steps)
                    
                    # Create ensemble of future forecasts
                    if len(future_forecasts) >= 2:
                        future_df = pd.DataFrame(future_forecasts)
                        future_ensemble = future_df.mean(axis=1)
                    else:
                        future_ensemble = list(future_forecasts.values())[0]
                    
                    # Generate forecast dates
                    last_date = df.index[-1]
                    freq = df.index.freq or pd.infer_freq(df.index) or 'D'
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq=freq)
                    
                    def plot_ensemble():
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.calculate_figure_size()[0], self.calculate_figure_size()[1] * 1.5))
                        
                        # Plot test set predictions
                        ax1.plot(train.index, train_series, label='Training Data')
                        ax1.plot(test.index, test_series, label='Test Data')
                        
                        for model_name, forecast in forecasts.items():
                            if model_name != 'Ensemble':
                                ax1.plot(test.index, forecast, linestyle='--', alpha=0.7, label=f'{model_name}')
                        
                        ax1.plot(test.index, forecast_df['Ensemble'], linewidth=2, color='red', label='Ensemble')
                        ax1.set_title(f'Ensemble Model Performance: {col}')
                        ax1.set_xlabel('Date')
                        ax1.set_ylabel('Value')
                        ax1.legend()
                        
                        # Plot future forecasts
                        ax2.plot(df.index, series, label='Historical Data')
                        
                        for model_name, forecast in future_forecasts.items():
                            ax2.plot(forecast_dates, forecast, linestyle='--', alpha=0.7, label=f'{model_name} Forecast')
                        
                        ax2.plot(forecast_dates, future_ensemble, linewidth=2, color='red', label='Ensemble Forecast')
                        ax2.set_title(f'Ensemble Forecast: {col}')
                        ax2.set_xlabel('Date')
                        ax2.set_ylabel('Value')
                        ax2.legend()
                        
                        plt.tight_layout()
                        return fig, (ax1, ax2)

                    result = self.generate_plot(plot_ensemble)
                    if result is not None:
                        fig, _ = result
                        img_path = os.path.join(self.output_folder, f"{table_name}_ensemble_{col}.png")
                        plt.savefig(img_path, dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        image_paths.append(img_path)
                    
                    # Save results
                    ensemble_results[col] = {
                        'models_used': list(forecasts.keys()),
                        'model_metrics': model_metrics,
                        'ensemble_forecast': future_ensemble.tolist() if hasattr(future_ensemble, 'tolist') else future_ensemble,
                        'forecast_dates': [date.strftime('%Y-%m-%d') for date in forecast_dates],
                        'best_individual_model': min(model_metrics.items(), key=lambda x: x[1]['rmse'] if x[0] != 'Ensemble' else float('inf'))[0]
                    }
                else:
                    print(warning(f"Not enough successful models for ensemble on {col}"))
            
            if not ensemble_results:
                raise ValueError("No valid data for ensemble time series analysis")
                
        except Exception as e:
            print(error(f"Error in Ensemble Time Series analysis: {str(e)}"))
            ensemble_results = {'error': str(e)}

        results = {
            'image_paths': image_paths,
            'ensemble_results': ensemble_results,
            'date_column': date_col,
            'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            'data_frequency': str(df.index.freq) if df.index.freq else 'Unknown'
        }
        
        self.interpret_results("Ensemble Time Series", results, table_name)

    def bootstrapping_time_series(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Bootstrapping Time Series"))
        image_paths = []
        bootstrap_results = {}
        
        try:
            # Prepare data
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                date_col = date_cols[0]
                df.set_index(date_col, inplace=True)
            else:
                df.index = pd.date_range(start='1/1/2000', periods=len(df))
                date_col = 'Date'

            if df.index.freq is None:
                inferred_freq = pd.infer_freq(df.index)
                if inferred_freq is not None:
                    df = df.asfreq(inferred_freq)
                else:
                    df = df.asfreq('D')

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            print(info(f"Found {len(numeric_cols)} numeric columns for Bootstrapping analysis"))
            
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[col].isna().all():
                    continue

                df[col] = df[col].interpolate().bfill().ffill()
                
                if len(df) < 30:  # Need sufficient data for bootstrapping
                    print(warning(f"Insufficient data for bootstrapping on {col}"))
                    continue
                
                # Parameters for bootstrapping
                series = df[col].values
                n_bootstraps = 1000  # Number of bootstrap samples
                
                # Determine block size based on data frequency
                if df.index.freq == 'D' or str(df.index.freq).startswith('D'):
                    block_size = 7  # One week for daily data
                elif df.index.freq == 'M' or str(df.index.freq).startswith('M'):
                    block_size = 3  # Three months for monthly data
                elif df.index.freq == 'Q' or str(df.index.freq).startswith('Q'):
                    block_size = 2  # Two quarters for quarterly data
                else:
                    block_size = max(2, len(series) // 10)  # Default is 10% of data length
                
                # Ensure block size is valid
                block_size = min(block_size, len(series) // 2)
                block_size = max(block_size, 2)
                
                print(info(f"Using block size {block_size} for bootstrapping {col}"))
                
                # Create bootstrap samples using block bootstrap
                bootstrapped_series = []
                
                for _ in range(n_bootstraps):
                    bootstrap_sample = []
                    
                    # Generate blocks until we have a full sample
                    while len(bootstrap_sample) < len(series):
                        # Randomly select block start
                        start_idx = np.random.randint(0, len(series) - block_size + 1)
                        # Add block
                        bootstrap_sample.extend(series[start_idx:start_idx + block_size])
                    
                    # Trim to original length
                    bootstrapped_series.append(bootstrap_sample[:len(series)])
                
                # Convert to array for easier calculation
                bootstrapped_array = np.array(bootstrapped_series)
                
                # Calculate statistics across bootstrap samples
                bootstrap_mean = np.mean(bootstrapped_array, axis=0)
                bootstrap_median = np.median(bootstrapped_array, axis=0)
                bootstrap_lower = np.percentile(bootstrapped_array, 2.5, axis=0)
                bootstrap_upper = np.percentile(bootstrapped_array, 97.5, axis=0)
                
                # Generate forecast
                forecast_steps = min(30, int(len(df) * 0.1))
                
                # Use ARIMA on each bootstrap sample to generate forecasts
                bootstrap_forecasts = []
                
                # Use a subset of bootstrap samples for forecasting to save time
                forecast_samples = 50
                for i in range(min(forecast_samples, n_bootstraps)):
                    try:
                        model = ARIMA(bootstrapped_series[i], order=(1, 1, 0))
                        results = model.fit()
                        forecast = results.forecast(steps=forecast_steps)
                        bootstrap_forecasts.append(forecast)
                    except:
                        pass  # Skip if ARIMA fails on a particular bootstrap sample
                
                if len(bootstrap_forecasts) > 0:
                    # Convert forecasts to array
                    forecast_array = np.array(bootstrap_forecasts)
                    
                    # Calculate forecast statistics
                    forecast_mean = np.mean(forecast_array, axis=0)
                    forecast_median = np.median(forecast_array, axis=0)
                    forecast_lower = np.percentile(forecast_array, 2.5, axis=0)
                    forecast_upper = np.percentile(forecast_array, 97.5, axis=0)
                    
                    # Generate forecast dates
                    last_date = df.index[-1]
                    freq = df.index.freq or pd.infer_freq(df.index) or 'D'
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq=freq)
                    
                    def plot_bootstrap():
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.calculate_figure_size()[0], self.calculate_figure_size()[1] * 1.5))
                        
                        # Plot historical data with bootstrap confidence intervals
                        ax1.plot(df.index, series, label='Observed', color='blue')
                        ax1.plot(df.index, bootstrap_mean, color='red', label='Bootstrap Mean')
                        ax1.fill_between(df.index, bootstrap_lower, bootstrap_upper, color='blue', alpha=0.2, label='95% CI')
                        ax1.set_title(f'Bootstrapping Time Series: {col}')
                        ax1.set_xlabel('Date')
                        ax1.set_ylabel('Value')
                        ax1.legend()
                        
                        # Plot forecast with confidence intervals
                        ax2.plot(df.index, series, label='Historical', color='blue')
                        ax2.plot(forecast_dates, forecast_mean, color='red', label='Forecast Mean')
                        ax2.fill_between(forecast_dates, forecast_lower, forecast_upper, color='red', alpha=0.2, label='95% CI')
                        ax2.set_title(f'Bootstrap Forecast: {col}')
                        ax2.set_xlabel('Date')
                        ax2.set_ylabel('Value')
                        ax2.legend()
                        
                        plt.tight_layout()
                        return fig, (ax1, ax2)

                    result = self.generate_plot(plot_bootstrap)
                    if result is not None:
                        fig, _ = result
                        img_path = os.path.join(self.output_folder, f"{table_name}_bootstrap_{col}.png")
                        plt.savefig(img_path, dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        image_paths.append(img_path)
                    
                    # Prepare results
                    bootstrap_results[col] = {
                        'n_bootstraps': n_bootstraps,
                        'block_size': block_size,
                        'mean_forecast': forecast_mean.tolist(),
                        'lower_ci': forecast_lower.tolist(),
                        'upper_ci': forecast_upper.tolist(),
                        'forecast_dates': [date.strftime('%Y-%m-%d') for date in forecast_dates],
                        'forecast_uncertainty': float(np.mean(forecast_upper - forecast_lower))
                    }
                else:
                    print(warning(f"Failed to generate forecasts for {col}"))
            
            if not bootstrap_results:
                raise ValueError("No valid data for bootstrapping time series analysis")
                
        except Exception as e:
            print(error(f"Error in Bootstrapping Time Series analysis: {str(e)}"))
            bootstrap_results = {'error': str(e)}

        results = {
            'image_paths': image_paths,
            'bootstrap_results': bootstrap_results,
            'date_column': date_col,
            'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            'data_frequency': str(df.index.freq) if df.index.freq else 'Unknown'
        }
        
        self.interpret_results("Bootstrapping Time Series", results, table_name)

    def sarimax_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - SARIMAX Analysis"))
        image_paths = []
        sarimax_results = {}
        
        try:
            # Prepare data
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                date_col = date_cols[0]
                df.set_index(date_col, inplace=True)
            else:
                df.index = pd.date_range(start='1/1/2000', periods=len(df))
                date_col = 'Date'

            if df.index.freq is None:
                inferred_freq = pd.infer_freq(df.index)
                if inferred_freq is not None:
                    df = df.asfreq(inferred_freq)
                else:
                    df = df.asfreq('D')

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            print(info(f"Found {len(numeric_cols)} numeric columns for SARIMAX analysis"))
            
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[col].isna().all():
                    continue

                df[col] = df[col].interpolate().bfill().ffill()
                
                # Determine seasonal period based on data frequency
                if df.index.freq == 'D' or str(df.index.freq).startswith('D'):
                    seasonal_periods = 7  # Weekly seasonality for daily data
                elif df.index.freq == 'M' or str(df.index.freq).startswith('M'):
                    seasonal_periods = 12  # Yearly seasonality for monthly data
                elif df.index.freq == 'Q' or str(df.index.freq).startswith('Q'):
                    seasonal_periods = 4  # Yearly seasonality for quarterly data
                else:
                    seasonal_periods = 1  # No seasonality as default
                
                # For small datasets, use simpler model
                if len(df) < 4 * seasonal_periods:
                    print(warning(f"Limited data for seasonal modeling. Using simpler SARIMAX model for {col}"))
                    order = (1, 1, 1)
                    seasonal_order = (0, 0, 0, 0)
                else:
                    # Use auto_arima to find best parameters if dataset isn't too large
                    if len(df) < 1000:
                        try:
                            print(info(f"Running auto_arima to find optimal SARIMAX parameters for {col}"))
                            auto_model = auto_arima(
                                df[col], 
                                seasonal=True, 
                                m=seasonal_periods,
                                suppress_warnings=True,
                                error_action="ignore",
                                max_order=None,
                                trace=False
                            )
                            order = auto_model.order
                            seasonal_order = auto_model.seasonal_order
                            print(info(f"Optimal SARIMAX parameters for {col}: order={order}, seasonal_order={seasonal_order}"))
                        except Exception as e:
                            print(warning(f"Auto ARIMA failed: {str(e)}. Using default parameters."))
                            order = (1, 1, 1)
                            seasonal_order = (1, 1, 0, seasonal_periods)
                    else:
                        # For large datasets, use reasonable defaults
                        order = (1, 1, 1)
                        seasonal_order = (1, 1, 0, seasonal_periods)

                # Fit SARIMAX model with timeout
                try:
                    model = SARIMAX(df[col], order=order, seasonal_order=seasonal_order)
                    results = self.model_fit_with_timeout(model, timeout=30)
                    
                    if results is None:
                        # Try simpler model if timeout
                        print(warning(f"SARIMAX model timed out. Trying simpler model for {col}"))
                        model = SARIMAX(df[col], order=(1, 1, 0), seasonal_order=(0, 0, 0, 0))
                        results = self.model_fit_with_timeout(model, timeout=30)
                        
                        if results is None:
                            raise TimeoutError("SARIMAX model fitting timed out")
                except Exception as e:
                    # Try one more time with even simpler model
                    print(warning(f"SARIMAX model failed: {str(e)}. Trying simplest model."))
                    model = SARIMAX(df[col], order=(1, 1, 0), seasonal_order=(0, 0, 0, 0))
                    results = self.model_fit_with_timeout(model, timeout=30)
                    
                    if results is None:
                        raise TimeoutError("SARIMAX model fitting timed out")
                
                # Split data for in-sample evaluation
                train_size = int(len(df) * 0.8)
                train_index = df.index[:train_size]
                test_index = df.index[train_size:]
                
                # Generate in-sample predictions
                predictions = results.predict()
                
                # Generate out-of-sample forecast
                forecast_steps = min(30, int(len(df) * 0.1))
                forecast = results.forecast(steps=forecast_steps)
                
                # Generate forecast dates
                last_date = df.index[-1]
                freq = df.index.freq or pd.infer_freq(df.index) or 'D'
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq=freq)

                def plot_sarimax():
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.calculate_figure_size()[0], self.calculate_figure_size()[1] * 1.5))
                    
                    # Plot training/testing split with predictions
                    ax1.plot(train_index, df.loc[train_index, col], label='Training Data')
                    ax1.plot(test_index, df.loc[test_index, col], label='Test Data')
                    ax1.plot(df.index, predictions, color='red', label='SARIMAX Fit')
                    ax1.set_title(f'SARIMAX Time Series: {col} (Order: {order}, Seasonal: {seasonal_order})')
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Value')
                    ax1.legend()
                    
                    # Plot forecast
                    ax2.plot(df.index, df[col], label='Historical Data')
                    ax2.plot(forecast_dates, forecast, color='green', label='Forecast')
                    ax2.set_title(f'SARIMAX Forecast: {col}')
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel('Value')
                    ax2.legend()
                    
                    plt.tight_layout()
                    return fig, (ax1, ax2)

                result = self.generate_plot(plot_sarimax)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_sarimax_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                # Calculate error metrics
                test_predictions = predictions.loc[test_index]
                test_actuals = df.loc[test_index, col]
                
                mse = mean_squared_error(test_actuals, test_predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(test_actuals, test_predictions)
                
                sarimax_results[col] = {
                    'aic': float(results.aic),
                    'bic': float(results.bic),
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'seasonal_periods': seasonal_periods,
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'forecast_values': forecast.tolist(),
                    'forecast_dates': [date.strftime('%Y-%m-%d') for date in forecast_dates]
                }
            
            if not sarimax_results:
                raise ValueError("No valid data for SARIMAX analysis")
                
        except Exception as e:
            print(error(f"Error in SARIMAX analysis: {str(e)}"))
            sarimax_results = {'error': str(e)}

        results = {
            'image_paths': image_paths,
            'sarimax_results': sarimax_results,
            'date_column': date_col,
            'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            'data_frequency': str(df.index.freq) if df.index.freq else 'Unknown'
        }
        
        self.interpret_results("SARIMAX Analysis", results, table_name)

    def theta_method(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Theta Method"))
        image_paths = []
        theta_results = {}
        
        try:
            # Prepare data
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                date_col = date_cols[0]
                df.set_index(date_col, inplace=True)
            else:
                df.index = pd.date_range(start='1/1/2000', periods=len(df))
                date_col = 'Date'

            if df.index.freq is None:
                inferred_freq = pd.infer_freq(df.index)
                if inferred_freq is not None:
                    df = df.asfreq(inferred_freq)
                else:
                    df = df.asfreq('D')

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            print(info(f"Found {len(numeric_cols)} numeric columns for Theta Method analysis"))
            
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[col].isna().all():
                    continue

                df[col] = df[col].interpolate().bfill().ffill()
                
                if len(df) < 10:  # Need sufficient data
                    print(warning(f"Insufficient data for theta method on {col}"))
                    continue
                
                # Apply Theta method
                series = df[col].values
                n = len(series)
                theta = 2  # Classical theta method uses theta=2
                
                # Step 1: Fit linear trend
                X = np.arange(1, n+1).reshape(-1, 1)
                y = series.reshape(-1, 1)
                model = LinearRegression()
                model.fit(X, y)
                
                # Get trend components
                slope = model.coef_[0][0]
                intercept = model.intercept_[0]
                trend = model.predict(X).flatten()
                
                # Step 2: Calculate detrended series
                detrended = series - trend
                
                # Step 3: Apply SES (Simple Exponential Smoothing) to detrended series
                alpha = 0.5  # Smoothing parameter
                ses = np.zeros(n)
                ses[0] = detrended[0]
                for i in range(1, n):
                    ses[i] = alpha * detrended[i] + (1-alpha) * ses[i-1]
                
                # Step 4: Combine trend and SES components to get fitted values
                theta_fitted = trend + theta * ses
                
                # Step 5: Generate forecast
                forecast_steps = min(30, int(len(df) * 0.1))
                X_future = np.arange(n+1, n+forecast_steps+1).reshape(-1, 1)
                future_trend = model.predict(X_future).flatten()
                
                # SES forecast is the last SES value
                future_ses = np.repeat(ses[-1], forecast_steps)
                
                theta_forecast = future_trend + theta * future_ses
                
                # Generate forecast dates
                last_date = df.index[-1]
                freq = df.index.freq or pd.infer_freq(df.index) or 'D'
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq=freq)
                
                # Calculate error metrics
                mse = mean_squared_error(series, theta_fitted)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(series, theta_fitted)
                
                def plot_theta():
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.calculate_figure_size()[0], self.calculate_figure_size()[1] * 1.5))
                    
                    # Plot data with fitted values
                    ax1.plot(df.index, series, label='Observed')
                    ax1.plot(df.index, trend, color='green', label='Trend')
                    ax1.plot(df.index, theta_fitted, color='red', label='Theta Method')
                    ax1.set_title(f'Theta Method Components: {col}')
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Value')
                    ax1.legend()
                    
                    # Plot forecast
                    ax2.plot(df.index, series, label='Historical Data')
                    ax2.plot(forecast_dates, theta_forecast, color='red', label='Theta Forecast')
                    ax2.set_title(f'Theta Method Forecast: {col}')
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel('Value')
                    ax2.legend()
                    
                    plt.tight_layout()
                    return fig, (ax1, ax2)

                result = self.generate_plot(plot_theta)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_theta_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)
                
                theta_results[col] = {
                    'theta_value': theta,
                    'trend_slope': float(slope),
                    'trend_intercept': float(intercept),
                    'smoothing_parameter': alpha,
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'forecast_values': theta_forecast.tolist(),
                    'forecast_dates': [date.strftime('%Y-%m-%d') for date in forecast_dates]
                }
            
            if not theta_results:
                raise ValueError("No valid data for theta method analysis")
                
        except Exception as e:
            print(error(f"Error in Theta Method analysis: {str(e)}"))
            theta_results = {'error': str(e)}

        results = {
            'image_paths': image_paths,
            'theta_results': theta_results,
            'date_column': date_col,
            'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            'data_frequency': str(df.index.freq) if df.index.freq else 'Unknown'
        }
        
        self.interpret_results("Theta Method", results, table_name)

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
        
        When referring to entities in the data (like companies, products, regions, etc.), use their actual names from the data rather than generic terms like "Company A" or "Region 1".

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
        
        When referring to entities in the data (like companies, products, regions, etc.), use their actual names from the data rather than generic terms like "Company A" or "Region 1".
        
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
        output_file = os.path.join(self.output_folder, "axda_b7_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        report_title = f"Advanced Exploratory Data Analysis (Batch 7) Report for {self.table_name}"
        
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
            formatted_image_data,
            filename=f"axda_b7_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
            return pdf_file
        else:
            print(error("Failed to generate PDF report"))
            return None