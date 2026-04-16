import ollama

# ---------------- SAFE LLM CALL ----------------
def safe_llm_call(prompt):
    try:
        response = ollama.chat(
            model="phi3:mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    except Exception:
        return "⚠️ LLM features are disabled in deployed version. Run locally to enable AI features."


# ---------------- Q&A ----------------
def ask_dataset_question(question, profile, quality, score):

    context = f"""
Dataset Score: {score}
Rows: {profile['rows']}
Columns: {profile['columns']}
Missing: {quality['missing']}
Duplicates: {quality['duplicates']}
Correlation: {quality['high_correlation']}
Outliers: {quality['outliers']}
"""

    prompt = f"""
You are an expert data scientist.

Rules:
- First line must be YES / NO / PARTIALLY
- Then give short bullet points
- Be specific and clear

Score Guide:
>80 = Good
60-80 = Needs preprocessing
<60 = Poor

Dataset:
{context}

Question:
{question}
"""

    return safe_llm_call(prompt)


# ---------------- MODEL SUGGESTION ----------------
def suggest_ml_model(profile, quality, target_column, score, df):

    if target_column is not None:
        unique_vals = df[target_column].nunique()
        problem_type = "classification" if unique_vals <= 10 else "regression"
    else:
        problem_type = "unknown"

    context = f"""
Problem Type: {problem_type}
Dataset Score: {score}
Target: {target_column}
Imbalance: {quality['imbalance']}
Missing: {quality['missing']}
"""

    prompt = f"""
You are an ML expert.

Rules:
- Suggest 2-3 models
- Short explanation
- Mention preprocessing

Dataset:
{context}
"""

    return safe_llm_call(prompt)


# ---------------- PREPROCESSING CODE ----------------
def generate_preprocessing_code(profile, quality, target_column, df):

    numeric_cols = list(df.select_dtypes(include="number").columns)
    cat_cols = list(df.select_dtypes(include="object").columns)

    # remove target
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    if target_column in cat_cols:
        cat_cols.remove(target_column)

    # detect columns
    id_cols = [c for c in df.columns if "id" in c.lower() or "number" in c.lower()]
    date_cols = [c for c in df.columns if "date" in c.lower()]
    time_cols = [c for c in df.columns if "time" in c.lower()]

    numeric_cols = [c for c in numeric_cols if c not in id_cols]
    cat_cols = [c for c in cat_cols if c not in id_cols]
    cat_cols = [c for c in cat_cols if c not in date_cols + time_cols]

    # ---------------- CELLS ----------------

    imports = """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
"""

    cleaning = f"""
# drop duplicates
df = df.drop_duplicates()

# drop id columns
df = df.drop(columns={id_cols}, errors="ignore")
"""

    date_processing = ""
    for col in date_cols:
        date_processing += f"""
df["{col}"] = pd.to_datetime(df["{col}"], errors="coerce")
df["{col}_year"] = df["{col}"].dt.year
df["{col}_month"] = df["{col}"].dt.month
df["{col}_day"] = df["{col}"].dt.day
"""

    time_processing = ""
    for col in time_cols:
        time_processing += f"""
df["{col}"] = pd.to_datetime(df["{col}"], errors="coerce")
df["{col}_hour"] = df["{col}"].dt.hour
"""

    feature_engineering = f"""
# process date columns
{date_processing}

# process time columns
{time_processing}

# drop original datetime columns
df = df.drop(columns={date_cols + time_cols}, errors="ignore")
"""

    if target_column:
        target_split = f"""
# target column
target = "{target_column}"

X = df.drop(columns=[target])
y = df[target]
"""
    else:
        target_split = "X = df.copy()"

    pipeline = f"""
numeric_features = {numeric_cols}
categorical_features = {cat_cols}

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

X_processed = preprocessor.fit_transform(X)
"""

    if target_column:
        split = """
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42
)
"""
    else:
        split = 'print("Preprocessing complete")'

    return [
        ("1️⃣ Imports", imports),
        ("2️⃣ Data Cleaning", cleaning),
        ("3️⃣ Feature Engineering", feature_engineering),
        ("4️⃣ Target Split", target_split),
        ("5️⃣ Pipeline", pipeline),
        ("6️⃣ Train Test Split", split),
    ]