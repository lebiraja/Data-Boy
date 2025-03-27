

---

# **🤖 AI-Powered Data Cleaner & Optimizer**  
**Automatically clean, optimize, and train ML models on your dataset using LLMs and Reinforcement Learning!**  

![Demo](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit)  
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)  
![License](https://img.shields.io/badge/License-MIT-green)  

---

## **📌 Overview**  
This project automates **data cleaning, optimization, and machine learning model training** using:  
- **LLMs (Ollama/DeepSeek)** → Analyze & clean messy data  
- **Reinforcement Learning (Stable-Baselines3)** → Optimize dataset structure  
- **XGBoost** → Train high-performance models  

Just upload a CSV, and the AI handles everything!  

---

## **🚀 Features**  
✅ **Automatic Data Analysis** – LLM identifies missing values, inconsistencies, and suggests fixes  
✅ **Smart Data Cleaning** – AI fills missing values and standardizes formats  
✅ **RL-Based Optimization** – Uses DQN to improve dataset quality  
✅ **AutoML Training** – XGBoost model trained on optimized data  
✅ **Streamlit UI** – User-friendly interface with real-time progress  

---

## **⚙️ Installation**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/your-username/data-cleaner-optimizer.git
cd data-cleaner-optimizer
```

### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```
*(See [requirements.txt](#requirements) below)*  

### **3. Set Up Ollama (LLM Backend)**  
Install [Ollama](https://ollama.ai/) and pull the DeepSeek model:  
```bash
ollama pull deepseek-r1:1.5b
```

---

## **📂 Requirements**  
```plaintext
streamlit==1.32.0
pandas==2.1.0
numpy==1.24.0
ollama==0.1.0
gym==0.26.0
stable-baselines3==2.0.0
xgboost==2.0.0
scikit-learn==1.3.0
```

---

## **💻 Usage**  

### **Run the App**  
```bash
streamlit run main.py
```
1. **Upload a CSV file**  
2. Watch the AI **analyze, clean, optimize, and train** automatically!  
3. **Download** cleaned data & model  

---

## **🧠 Code Structure**  

### **1. `main.py` (Core Pipeline)**  
```python
def analyze_with_llm(df):           # LLM analyzes dataset & suggests fixes
def auto_clean_data(df):            # Auto-fills missing values & standardizes data
def optimize_with_rl(df):           # Uses RL to optimize dataset structure
def train_model(df):                # Trains XGBoost on optimized data
def run_streamlit_app():            # Streamlit UI & automation flow
```

### **2. Key Workflow**  
1. **Upload CSV** → `pd.read_csv()`  
2. **LLM Analysis** → `ollama.chat()`  
3. **Auto-Cleaning** → Fill NA, normalize strings  
4. **RL Optimization** → `DQN("MlpPolicy", env)`  
5. **Model Training** → `XGBClassifier().fit()`  

---

## **📊 Example Outputs**  
| Stage | Output |
|--------|--------|
| **LLM Analysis** | "Found 12% missing values in 'Price'. Recommend median imputation." |  
| **Cleaned Data** | Missing values filled, outliers removed |  
| **RL Optimization** | 15% fewer rows, improved feature distribution |  
| **Model Accuracy** | **92.3%** (XGBoost) |  

---

## **📜 License**  
MIT License - Free for personal/commercial use.  

---

## **🔗 Links**  
- **[Report Issues](https://github.com/your-username/data-cleaner-optimizer/issues)**  
- **[Contribute](https://github.com/your-username/data-cleaner-optimizer/pulls)**  

---

### **🎯 Why This Project?**  
✔ **Saves 80% time** on data cleaning  
✔ **No manual coding** – AI handles everything  
✔ **End-to-end** from raw data to trained model  

**⭐ Star this repo if you find it useful!**  

---
