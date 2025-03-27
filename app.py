import ollama
import pandas as pd
import numpy as np
import streamlit as st
import os
import time
from datetime import datetime
import logging
from typing import Tuple, Optional, Dict
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_cleaner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_FILE_TYPES = ["csv", "xlsx", "json", "parquet"]
MAX_FILE_SIZE_MB = 50
MAX_ROWS_FOR_ANALYSIS = 50000
MODEL_HISTORY_FILE = "model_history.json"

class DataCleanerApp:
    def __init__(self):
        self.base_dir = self._setup_workspace()
        self._init_session_state()
        self.model_history = self._load_model_history()
        
    def _setup_workspace(self) -> Path:
        """Create workspace directory with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(f"data_cleaner_workspace_{timestamp}")
        base_dir.mkdir(exist_ok=True)
        return base_dir
    
    def _init_session_state(self):
        """Initialize Streamlit session state"""
        defaults = {
            'processing_stage': 0,  # 0: not started, 1: uploaded, 2: cleaned, 3: validated
            'df': None,
            'model_metrics': None,
            'processing_time': {},
            'error_messages': [],
            'llm_analysis': "",
            'target_variable': None,
            'baseline_accuracy': None,
            'final_accuracy': None,
            'model_path': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _load_model_history(self) -> Dict:
        """Load previous model training history"""
        history_file = Path(MODEL_HISTORY_FILE)
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load model history: {str(e)}")
        return {}

    def _save_model_history(self):
        """Save model training history"""
        try:
            with open(MODEL_HISTORY_FILE, 'w') as f:
                json.dump(self.model_history, f)
        except Exception as e:
            logger.error(f"Could not save model history: {str(e)}")

    def _validate_file(self, uploaded_file) -> bool:
        """Validate the uploaded file"""
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File size exceeds maximum limit of {MAX_FILE_SIZE_MB}MB")
            return False
        
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext not in SUPPORTED_FILE_TYPES:
            st.error(f"Unsupported file type. Supported types: {', '.join(SUPPORTED_FILE_TYPES)}")
            return False
            
        return True

    def _load_data(self, file_path: Path, file_ext: str) -> Optional[pd.DataFrame]:
        """Load data based on file type with robust error handling"""
        try:
            if file_ext == 'csv':
                return pd.read_csv(file_path)
            elif file_ext == 'xlsx':
                return pd.read_excel(file_path)
            elif file_ext == 'json':
                return pd.read_json(file_path)
            elif file_ext == 'parquet':
                return pd.read_parquet(file_path)
        except Exception as e:
            error_msg = f"Error loading {file_ext} file: {str(e)}"
            logger.error(error_msg)
            st.session_state.error_messages.append(error_msg)
        return None

    def _clean_price_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean price column by removing $ and converting to float"""
        if 'price' in df.columns:
            df['price'] = df['price'].replace('[\$,]', '', regex=True)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        return df

    def _detect_target_variable(self, df: pd.DataFrame) -> Optional[str]:
        """Intelligently detect the best target variable"""
        # Skip columns with too many unique values
        for col in df.columns:
            unique_ratio = len(df[col].unique()) / len(df)
            
            # Skip ID-like columns
            if unique_ratio > 0.9:
                continue
                
            # Prefer columns with 2-10 unique values for classification
            if 1 < len(df[col].unique()) <= 10:
                return col
                
            # For regression, look for numeric columns with reasonable distribution
            if pd.api.types.is_numeric_dtype(df[col]) and unique_ratio > 0.1:
                return col
                
        return None

    def analyze_with_llm(self, df: pd.DataFrame) -> str:
        """Analyze dataset with LLM and get recommendations"""
        start_time = time.time()
        
        # Sample data for analysis to save tokens
        sample_size = min(100, len(df))
        sample_df = df.sample(sample_size) if len(df) > sample_size else df
        
        dataset_json = sample_df.head(5).to_json()
        
        prompt = f"""
        Analyze this dataset and provide specific, actionable recommendations:
        {dataset_json}
        
        1. Identify all missing values and their columns
        2. Suggest appropriate methods to fill missing values (mean/median/mode/drop)
        3. Detect any inconsistent data formats
        4. Identify potential categorical variables
        5. Recommend any necessary data transformations
        
        Provide your response in clear bullet points with technical explanations.
        """
        
        try:
            response = ollama.chat(
                model="deepseek-r1:1.5b", 
                messages=[{"role": "user", "content": prompt}]
            )
            analysis = response["message"]["content"]
            
            st.session_state.processing_time['llm_analysis'] = time.time() - start_time
            logger.info(f"LLM analysis completed in {st.session_state.processing_time['llm_analysis']:.2f}s")
            
            return analysis
        except Exception as e:
            error_msg = f"LLM Analysis failed: {str(e)}"
            logger.error(error_msg)
            st.session_state.error_messages.append(error_msg)
            return error_msg

    def auto_clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Automatically clean data with LLM guidance"""
        start_time = time.time()
        
        # Clean price column first
        df = self._clean_price_column(df)
        
        # Get LLM recommendations
        analysis = self.analyze_with_llm(df)
        st.session_state.llm_analysis = analysis
        
        cleaned_df = df.copy()
        cleaning_report = []
        
        # Handle missing values
        missing_cols = cleaned_df.columns[cleaned_df.isnull().any()].tolist()
        if missing_cols:
            cleaning_report.append("### Missing Value Treatment")
            
            for col in missing_cols:
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    fill_value = cleaned_df[col].median() if cleaned_df[col].nunique() > 5 else cleaned_df[col].mode()[0]
                    cleaned_df[col].fillna(fill_value, inplace=True)
                    cleaning_report.append(f"- Filled missing values in {col} with {fill_value:.2f}")
                else:
                    fill_value = cleaned_df[col].mode()[0]
                    cleaned_df[col].fillna(fill_value, inplace=True)
                    cleaning_report.append(f"- Filled missing values in {col} with '{fill_value}'")
        
        # Convert string numbers to numeric
        converted_cols = []
        for col in cleaned_df.select_dtypes(include=['object']):
            try:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col])
                converted_cols.append(col)
            except:
                pass
        
        if converted_cols:
            cleaning_report.append("### Data Type Conversions")
            cleaning_report.append(f"- Converted columns to numeric: {', '.join(converted_cols)}")
        
        # Standardize date formats
        date_cols = []
        for col in cleaned_df.select_dtypes(include=['object']):
            try:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col])
                date_cols.append(col)
            except:
                pass
        
        if date_cols:
            cleaning_report.append("### Date Format Standardization")
            cleaning_report.append(f"- Converted to datetime: {', '.join(date_cols)}")
        
        st.session_state.processing_time['data_cleaning'] = time.time() - start_time
        logger.info(f"Data cleaning completed in {st.session_state.processing_time['data_cleaning']:.2f}s")
        
        return cleaned_df, "\n".join(cleaning_report)

    def validate_with_model(self, df: pd.DataFrame) -> Tuple[Optional[float], str]:
        """Validate data quality with model training"""
        start_time = time.time()
        validation_report = []
        try:
            # Auto-detect target column
            target_col = self._detect_target_variable(df)
            if target_col is None:
                return None, "Could not identify a suitable target variable for validation."
            
            st.session_state.target_variable = target_col
            validation_report.append(f"### Validation Target: {target_col}")
            
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Preprocess features
            X = X.select_dtypes(include=[np.number])
            X = X.fillna(X.mean())
            
            # Check if target is continuous or categorical
            unique_values = len(y.unique())
            total_values = len(y)
        
            if unique_values / total_values > 0.1:  # More than 10% unique values -> regression
                from xgboost import XGBRegressor
                from sklearn.metrics import mean_absolute_error, r2_score
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Model training
                model = XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    early_stopping_rounds=10,
                    eval_metric='mae'
                )
                
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=False
                )
                
                # Evaluation
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                validation_report.append("### Regression Validation Metrics")
                validation_report.append(f"- Mean Absolute Error: {mae:.2f}")
                validation_report.append(f"- R² Score: {r2:.2f}")
                
                st.session_state.final_accuracy = r2  # Using R² as our accuracy metric
                
            else:  # Classification
                from xgboost import XGBClassifier
                from sklearn.metrics import accuracy_score
                
                # Encode categorical target
                y = pd.factorize(y)[0]
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42,
                    stratify=y if unique_values < 10 else None
                )
                
                # Model training
                model = XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    early_stopping_rounds=10,
                    eval_metric='logloss'
                )
                
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=False
                )
                
                # Evaluation
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                validation_report.append("### Classification Validation Metrics")
                validation_report.append(f"- Accuracy: {accuracy:.2%}")
                
                st.session_state.final_accuracy = accuracy
        
        # Feature importance
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            validation_report.append("### Feature Importance")
            validation_report.append(importance.head(10).to_markdown())
            
            # Save model
            model_path = self.base_dir / "validation_model.json"
            model.save_model(model_path)
            st.session_state.model_path = model_path
            
            # Update model history
            self.model_history[str(datetime.now())] = {
                'target': target_col,
                'metrics': st.session_state.final_accuracy,
                'features': X.columns.tolist(),
                'model_path': str(model_path)
            }
            self._save_model_history()
            
            st.session_state.processing_time['model_validation'] = time.time() - start_time
            logger.info(f"Model validation completed in {st.session_state.processing_time['model_validation']:.2f}s")
            
            return st.session_state.final_accuracy, "\n".join(validation_report)
        except Exception as e:
            error_msg = f"Model validation failed: {str(e)}"
            logger.error(error_msg)
            st.session_state.error_messages.append(error_msg)
            return None, error_msg
    def _create_sidebar(self):
        """Create the application sidebar"""
        with st.sidebar:
            st.title("Data Quality Validator")
            st.markdown("""
            **Workflow:**
            1. Upload your dataset
            2. Automatic cleaning
            3. Data quality validation
            """)
            
            if st.session_state.error_messages:
                with st.expander("⚠️ Error Log", expanded=False):
                    for error in st.session_state.error_messages:
                        st.error(error)
            
            if st.session_state.processing_time:
                st.markdown("### Processing Times")
                for step, time_taken in st.session_state.processing_time.items():
                    st.write(f"{step.replace('_', ' ').title()}: {time_taken:.2f}s")

    def run_streamlit_app(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="Professional Data Validator",
            layout="wide",
            page_icon="🔍"
        )
        
        self._create_sidebar()
        
        st.title("🔍 Professional Data Cleaning & Validation")
        st.markdown("""
        A robust system for cleaning and validating dataset quality through model performance.
        """)
        
        # File upload section
        if st.session_state.processing_stage == 0:
            st.subheader("1. Data Upload")
            uploaded_file = st.file_uploader(
                "Choose a dataset file",
                type=SUPPORTED_FILE_TYPES,
                accept_multiple_files=False,
                key="file_uploader"
            )
            
            if uploaded_file:
                if not self._validate_file(uploaded_file):
                    return
                
                with st.spinner("Processing uploaded file..."):
                    try:
                        # Save uploaded file
                        file_ext = uploaded_file.name.split('.')[-1].lower()
                        file_path = self.base_dir / f"uploaded_dataset.{file_ext}"
                        
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Load dataset
                        st.session_state.df = self._load_data(file_path, file_ext)
                        
                        if st.session_state.df is not None:
                            # Downsample if too large
                            if len(st.session_state.df) > MAX_ROWS_FOR_ANALYSIS:
                                st.session_state.df = st.session_state.df.sample(MAX_ROWS_FOR_ANALYSIS)
                                st.warning(f"Dataset too large. Using random sample of {MAX_ROWS_FOR_ANALYSIS} rows.")
                            
                            st.session_state.processing_stage = 1
                            
                            st.success("✅ File uploaded successfully!")
                            with st.expander("Dataset Preview", expanded=True):
                                st.dataframe(st.session_state.df.head())
                            
                            with st.expander("Dataset Statistics", expanded=False):
                                st.json({
                                    "Rows": len(st.session_state.df),
                                    "Columns": len(st.session_state.df.columns),
                                    "Missing Values": int(st.session_state.df.isnull().sum().sum()),
                                    "Memory Usage": f"{st.session_state.df.memory_usage().sum() / 1024:.2f} KB"
                                })
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
                        logger.error(f"File upload error: {str(e)}")
                        return
        
        # Data cleaning stage
        if st.session_state.processing_stage == 1 and st.session_state.df is not None:
            st.subheader("2. Data Cleaning")
            with st.spinner("Analyzing and cleaning data..."):
                try:
                    # Clean data
                    st.session_state.df, cleaning_report = self.auto_clean_data(st.session_state.df)
                    
                    # Save cleaned data
                    cleaned_path = self.base_dir / "cleaned_dataset.csv"
                    st.session_state.df.to_csv(cleaned_path, index=False)
                    
                    st.session_state.processing_stage = 2
                    
                    st.success("✅ Data cleaning completed!")
                    with st.expander("Data Analysis Report", expanded=True):
                        st.markdown(st.session_state.llm_analysis)
                    
                    with st.expander("Cleaning Operations Performed", expanded=False):
                        st.markdown(cleaning_report)
                    
                    with st.expander("Cleaned Dataset Preview", expanded=False):
                        st.dataframe(st.session_state.df.head())
                    
                    st.write("### Cleaning Summary")
                    cols = st.columns(2)
                    cols[0].metric("Missing Values Remaining", 
                                  int(st.session_state.df.isnull().sum().sum()))
                    cols[1].metric("Processing Time", 
                                  f"{st.session_state.processing_time.get('data_cleaning', 0):.2f}s")
                except Exception as e:
                    st.error(f"Data cleaning failed: {str(e)}")
                    logger.error(f"Data cleaning error: {str(e)}")
                    return
        
        # Model validation stage
        if st.session_state.processing_stage == 2 and st.session_state.df is not None:
            st.subheader("3. Data Quality Validation")
            with st.spinner("Validating data quality with model training..."):
                try:
                    # Validate with model
                    accuracy, validation_report = self.validate_with_model(st.session_state.df)
                    st.session_state.model_metrics = accuracy
                    st.session_state.processing_stage = 3
                    
                    if accuracy is not None:
                        st.success(f"✅ Validation completed with accuracy: {accuracy:.2%}")
                        with st.expander("Validation Report", expanded=True):
                            st.markdown(validation_report)
                        
                        st.write("### Validation Summary")
                        cols = st.columns(3)
                        cols[0].metric("Validation Accuracy", f"{accuracy:.2%}")
                        cols[1].metric("Target Variable", 
                                     st.session_state.target_variable)
                        cols[2].metric("Processing Time", 
                                     f"{st.session_state.processing_time.get('model_validation', 0):.2f}s")
                        
                        # Show download options
                        st.subheader("4. Download Results")
                        cols = st.columns(2)
                        
                        cleaned_path = self.base_dir / "cleaned_dataset.csv"
                        if cleaned_path.exists():
                            with open(cleaned_path, "rb") as f:
                                cols[0].download_button(
                                    label="Download Cleaned Data",
                                    data=f.read(),
                                    file_name="cleaned_data.csv",
                                    mime="text/csv"
                                )
                        
                        if st.session_state.model_path and st.session_state.model_path.exists():
                            with open(st.session_state.model_path, "rb") as f:
                                cols[1].download_button(
                                    label="Download Validation Model",
                                    data=f.read(),
                                    file_name="validation_model.json",
                                    mime="application/json"
                                )
                    else:
                        st.warning("Validation completed with notes")
                        st.markdown(validation_report)
                except Exception as e:
                    st.error(f"Validation failed: {str(e)}")
                    logger.error(f"Validation error: {str(e)}")
                    return

if __name__ == "__main__":
    app = DataCleanerApp()
    app.run_streamlit_app()
