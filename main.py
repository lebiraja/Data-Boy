import ollama
import pandas as pd
import numpy as np
import gym
from stable_baselines3 import DQN
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import os
import time

# Set base directory
BASE_DIR = r"C:\Users\Lebi Raja\DataCleaner-Optimizer"
os.makedirs(BASE_DIR, exist_ok=True)

# Initialize session state
if 'processing_stage' not in st.session_state:
    st.session_state.processing_stage = 0  # 0: not started, 1: uploaded, 2: cleaned, 3: optimized, 4: trained
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_accuracy' not in st.session_state:
    st.session_state.model_accuracy = None

# 1. LLM Analysis Function
def analyze_with_llm(df):
    dataset_json = df.head(5).to_json()
    
    prompt = f"""
    Analyze this dataset and provide recommendations:
    {dataset_json}
    
    1. Identify all missing values and their columns
    2. Suggest appropriate methods to fill missing values
    3. Detect any inconsistent data formats
    4. Recommend any necessary data transformations
    
    Provide your response in clear bullet points.
    """
    
    try:
        response = ollama.chat(model="deepseek-r1:1.5b", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        return f"LLM Analysis failed: {str(e)}"

# 2. Automatic Data Cleaning
def auto_clean_data(df):
    # First get LLM recommendations
    analysis = analyze_with_llm(df)
    
    # Basic automatic cleaning based on common patterns
    cleaned_df = df.copy()
    
    # Handle missing values
    for col in cleaned_df.columns:
        if cleaned_df[col].isnull().sum() > 0:
            # For numeric columns
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                if cleaned_df[col].nunique() > 5:  # Continuous variable
                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                else:  # Likely categorical numeric
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
            # For categorical columns
            else:
                cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    
    # Convert string numbers to numeric
    for col in cleaned_df.select_dtypes(include=['object']).columns:
        try:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col])
        except:
            pass
    
    return cleaned_df, analysis

# 3. RL Optimization (Simplified)
class DatasetOptimizationEnv(gym.Env):
    def __init__(self, df):
        super(DatasetOptimizationEnv, self).__init__()
        self.df = df.copy()
        self.current_index = 0
        self.action_space = gym.spaces.Discrete(3)  # 0: keep, 1: modify, 2: remove
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(df.shape[1],), dtype=np.float32)
    
    def reset(self):
        self.current_index = 0
        return self._get_observation()
    
    def _get_observation(self):
        row = self.df.iloc[self.current_index].values
        return (row - np.nanmin(row)) / (np.nanmax(row) - np.nanmin(row) + 1e-6 )
    
    def step(self, action):
        reward = 0
        done = False
        
        if action == 2:  # Remove
            self.df = self.df.drop(self.current_index).reset_index(drop=True)
            reward = 0.1
        elif action == 1:  # Modify
            # Skip LLM for speed, just normalize the row
            self.df.iloc[self.current_index] = self._get_observation()
            reward = 0.3
        
        self.current_index += 1
        if self.current_index >= len(self.df):
            done = True
        
        return self._get_observation(), reward, done, {}

def optimize_with_rl(df):
    try:
        env = DatasetOptimizationEnv(df)
        model = DQN("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=min(1000, len(df)*2))
        optimized_df = env.df
        return optimized_df, "RL optimization completed successfully!"
    except Exception as e:
        return df, f"RL optimization failed: {str(e)}"

# 4. AutoML Training
def train_model(df):
    try:
        # Auto-detect target column (last column by default)
        target_col = df.columns[-1]
        
        # If last column has too many unique values, try to find a better target
        if len(df[target_col].unique()) > len(df) / 2:
            for col in df.columns:
                if 1 < len(df[col].unique()) < len(df) / 10:
                    target_col = col
                    break
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Convert text columns using simple encoding
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.factorize(X[col])[0]
        
        # Convert target if needed
        if y.dtype == 'object':
            y = pd.factorize(y)[0]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        model = XGBClassifier()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        model.save_model(os.path.join(BASE_DIR, "xgboost_model.json"))
        return accuracy, f"Model trained on '{target_col}' as target variable"
    except Exception as e:
        return None, f"Model training failed: {str(e)}"

# Streamlit App
def run_streamlit_app():
    st.set_page_config(page_title="Auto Data Cleaner & Model Trainer", layout="wide")
    st.title("🤖 Auto Data Cleaner & Model Trainer")
    st.markdown("Upload your dataset and let AI automatically clean, optimize, and train a model!")
    
    # Upload dataset
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file and st.session_state.processing_stage == 0:
        with st.spinner("Uploading and analyzing data..."):
            # Save uploaded file
            filepath = os.path.join(BASE_DIR, "uploaded_dataset.csv")
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load dataset
            st.session_state.df = pd.read_csv(filepath)
            st.session_state.processing_stage = 1
            
            # Show uploaded dataset
            st.success("✅ File uploaded successfully!")
            st.write("### Original Dataset Preview")
            st.dataframe(st.session_state.df.head())
            
            # Start automatic processing
            st.session_state.processing_stage = 1
    
    # Automatic processing pipeline
    if st.session_state.processing_stage >= 1 and st.session_state.df is not None:
        if st.session_state.processing_stage == 1:
            with st.spinner("Analyzing and cleaning data with AI..."):
                # Clean data
                st.session_state.df, analysis = auto_clean_data(st.session_state.df)
                
                # Save cleaned data
                cleaned_path = os.path.join(BASE_DIR, "cleaned_dataset.csv")
                st.session_state.df.to_csv(cleaned_path, index=False)
                
                st.session_state.processing_stage = 2
                
                st.success("✅ Data cleaning completed!")
                st.write("### Data Analysis Report")
                st.markdown(analysis)
                st.write("### Cleaned Dataset Preview")
                st.dataframe(st.session_state.df.head())
        
        if st.session_state.processing_stage == 2:
            with st.spinner("Optimizing dataset with Reinforcement Learning..."):
                # Optimize with RL
                st.session_state.df, rl_report = optimize_with_rl(st.session_state.df)
                
                # Save optimized data
                optimized_path = os.path.join(BASE_DIR, "optimized_dataset.csv")
                st.session_state.df.to_csv(optimized_path, index=False)
                
                st.session_state.processing_stage = 3
                
                st.success("✅ Dataset optimization completed!")
                st.write("### Optimized Dataset Preview")
                st.dataframe(st.session_state.df.head())
        
        if st.session_state.processing_stage == 3:
            with st.spinner("Training machine learning model..."):
                # Train model
                accuracy, training_report = train_model(st.session_state.df)
                st.session_state.model_accuracy = accuracy
                st.session_state.processing_stage = 4
                
                if accuracy is not None:
                    st.success(f"✅ Model trained with accuracy: {accuracy:.2%}")
                else:
                    st.warning("Model training completed with warnings")
                st.write("### Training Report")
                st.markdown(training_report)
    
    # Final results
    if st.session_state.processing_stage == 4:
        st.balloons()
        st.write("### Process Completed!")
        
        if st.session_state.model_accuracy is not None:
            st.metric("Model Accuracy", f"{st.session_state.model_accuracy:.2%}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Cleaned Data",
                data=pd.read_csv(os.path.join(BASE_DIR, "cleaned_dataset.csv")).to_csv(index=False),
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
        
        with col2:
            st.download_button(
                label="Download Optimized Data",
                data=pd.read_csv(os.path.join(BASE_DIR, "optimized_dataset.csv")).to_csv(index=False),
                file_name="optimized_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    run_streamlit_app()
