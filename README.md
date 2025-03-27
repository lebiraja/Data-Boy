**AI-Powered Data Cleaner & Validator**  
Automatically clean, validate, and assess dataset quality using LLMs and XGBoost!  

---

**Overview**  
This professional data validation tool provides:  
- LLM-Powered Analysis (Ollama/DeepSeek) → Examines data quality and suggests improvements  
- Automatic Cleaning → Handles missing values, type conversions, and standardization  
- Model-Based Validation → Uses XGBoost to validate dataset quality through predictive performance  
- Comprehensive Reporting → Detailed cleaning logs and validation metrics  

Just upload your dataset and get a full quality assessment!  

---

**Key Features**  
- Smart Data Analysis – LLM examines dataset structure and suggests fixes  
- Auto-Cleaning Pipeline – Handles missing values, type conversions, and standardization  
- Model Validation – XGBoost model tests data quality through predictive performance  
- Professional Reporting – Detailed cleaning logs and validation metrics  
- History Tracking – Saves model validation results for comparison  
- Streamlit UI – Clean, professional interface with real-time progress  

---

**Installation**  

1. Clone the Repository  
```
git clone https://github.com/your-username/data-cleaner-validator.git
cd data-cleaner-validator
```

2. Install Dependencies  
```
pip install -r requirements.txt
```

3. Set Up Ollama (LLM Backend)  
Install Ollama and pull the DeepSeek model:  
```
ollama pull deepseek-r1:1.5b
```

---

**Requirements**  
```
streamlit==1.32.0
pandas==2.1.0
numpy==1.24.0
ollama==0.1.0
xgboost==2.0.0
scikit-learn==1.3.0
```

---

**Usage**  

Run the App  
```
streamlit run app.py
```

**Workflow:**  
1. Upload your dataset (CSV, Excel, JSON, or Parquet)  
2. View LLM analysis of data quality issues  
3. Automatic cleaning operations are performed  
4. Validation model trains to assess data quality  
5. Download cleaned data and validation report  

---

**Code Structure**  

Core Components in `app.py`:  
```
class DataCleanerApp:
    def _load_data()          # Robust data loading with validation
    def analyze_with_llm()    # Get LLM analysis of dataset quality
    def auto_clean_data()     # Perform automatic data cleaning
    def validate_with_model() # Train XGBoost to validate data quality
    def run_streamlit_app()   # Streamlit UI implementation
```

Key Workflow  
- File Upload & Validation → Checks size/type and loads data  
- LLM Analysis → Examines data structure and suggests fixes  
- Auto-Cleaning → Handles missing values, type conversions  
- Model Validation → XGBoost tests data quality  
- Reporting → Generates comprehensive quality report  

---

**Example Outputs**  

| Stage            | Output |
|-----------------|-----------------------------------------------|
| LLM Analysis    | "Found 15% missing values in 'price'. Recommend median imputation." |  
| Data Cleaning   | "Filled missing values, converted 2 columns to numeric" |  
| Validation      | "Classification accuracy: 92.3% with 'status' as target" |  
| Feature Importance | Top predictive features ranked by importance |  

---

**License**  
This project is licensed under the MIT License – Free for personal and commercial use.  

---

**Links**  
- Report Issues: https://github.com/your-username/data-cleaner-validator/issues  
- Contribute: https://github.com/your-username/data-cleaner-validator/pulls  

---

**Why This Project?**  
- Professional-grade data validation  
- LLM-guided cleaning recommendations  
- Model-based quality assessment  
- Full audit trail of all transformations  

Star this repo if you find it useful!  

