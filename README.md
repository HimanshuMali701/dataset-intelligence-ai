# 🤖 AI Dataset Assistant

An **LLM-powered dataset analysis system** that automatically evaluates data quality, detects issues, suggests preprocessing steps, and enables natural language interaction with datasets.

---

## 🚀 Features

### 📊 Data Profiling
- Dataset shape & structure  
- Data types analysis  
- Missing values summary  

---

### 🧪 Data Quality Analysis
- Missing value detection  
- Duplicate detection  
- Outlier detection (IQR)  
- Class imbalance detection  
- High correlation detection  
- Constant column detection  

---

### 📈 Dataset Health Score

Evaluates dataset quality based on:
- Completeness  
- Duplicates  
- Correlation  
- Outliers  
- Imbalance  

**Example:**
Dataset Health Score: 82 / 100

---

### 💡 AI Recommendations
- Missing value handling  
- Feature selection suggestions  
- Imbalance handling (SMOTE)  
- Outlier treatment  
- Scaling recommendations  

---

### 📊 Auto EDA Dashboard
- Missing value chart  
- Correlation heatmap  
- Target distribution  
- Numeric feature distributions  

---

### 🤖 AI-Powered Features

#### 🧠 Natural Language Q&A
Ask questions like:
- *"Is this dataset good for prediction?"*  
- *"Which column has most missing values?"*  
- *"What preprocessing should I apply?"*  

---

#### 📌 Model Recommendation
Suggests:
- Classification models  
- Regression models  
- Based on dataset characteristics  

---

#### ⚙️ Preprocessing Code Generator
Generates **ready-to-run ML preprocessing code**, including:
- Data cleaning  
- Feature engineering  
- Encoding  
- Scaling  
- Train-test split  

---

#### 💬 Conversational Chat
- Chat history support  
- Context-aware responses  

---

## 🏗️ Project Structure
```
AI-Dataset-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│ ├── data_loader.py
│ ├── data_profiler.py
│ ├── quality_checker.py
│ ├── scoring.py
│ ├── suggestion_engine.py
│ ├── visualization.py
│ └── llm_agent.py
```

---

## 🛠️ Tech Stack

### 🔹 Core
- Python  
- Pandas  
- NumPy  

### 🔹 Visualization
- Matplotlib  
- Seaborn  

### 🔹 AI / LLM
- Ollama (Local LLM)  
- Phi-3 / Mistral  

### 🔹 Framework
- Streamlit  

---
## ⚙️ Installation & Setup

---

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/ai-dataset-assistant.git
cd ai-dataset-assistant
```
### 2️⃣ Create Virtual Environment
```
python -m venv venv
```
Activate Environment
```
venv\Scripts\activate
```
### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
### 4️⃣ Install Ollama

Download and install from:
👉 https://ollama.com

### 5️⃣ Pull LLM Model
```
ollama pull phi3:mini
```
### ▶️ Run the App
```
streamlit run app.py
```