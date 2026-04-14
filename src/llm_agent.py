import ollama

def ask_dataset_question(question, profile, quality, score):

    context = f"""
Dataset Score: {score}
Rows: {profile['rows']}
Columns: {profile['columns']}

Missing:
{quality['missing']}

Duplicates:
{quality['duplicates']}

Correlation:
{quality['high_correlation']}

Outliers:
{quality['outliers']}
"""

    prompt = f"""
You are an expert data scientist.

Answer clearly and decisively.

Rules:
- First line must be YES / NO / PARTIALLY
- Then give bullet points
- Be specific
- Avoid generic statements

Decision Guidelines:
Score > 80 → Good dataset
Score 60–80 → Needs preprocessing
Score < 60 → Poor dataset

Dataset:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="phi3:mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

def suggest_ml_model(profile, quality, target_column, score, df):

    # detect problem type
    if target_column is not None:
        unique_vals = df[target_column].nunique()

        if unique_vals <= 10:
            problem_type = "classification"
        else:
            problem_type = "regression"
    else:
        problem_type = "unknown"

    context = f"""
Problem Type: {problem_type}
Dataset Score: {score}
Rows: {profile['rows']}
Columns: {profile['columns']}
Target Column: {target_column}

Imbalance:
{quality['imbalance']}

Missing:
{quality['missing']}
"""

    prompt = f"""
You are an ML expert.

Suggest best models.

Rules:
- Give 2-3 models only
- Explain why
- Mention preprocessing if needed
- Avoid long text

Dataset:
{context}
"""

    response = ollama.chat(
        model="phi3:mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]    

def generate_preprocessing_code(profile, quality, target_column, df):

    numeric_cols = list(df.select_dtypes(include="number").columns)
    cat_cols = list(df.select_dtypes(include="object").columns)

    # remove target
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    if target_column in cat_cols:
        cat_cols.remove(target_column)

    # detect special columns
    id_cols = [c for c in df.columns if "id" in c.lower() or "number" in c.lower()]
    date_cols = [c for c in df.columns if "date" in c.lower()]
    time_cols = [c for c in df.columns if "time" in c.lower()]

    # remove id columns
    numeric_cols = [c for c in numeric_cols if c not in id_cols]
    cat_cols = [c for c in cat_cols if c not in id_cols]

    # remove date/time from categorical
    cat_cols = [c for c in cat_cols if c not in date_cols + time_cols]

    # ---------------- IMPORT CELL ----------------
    imports = """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
"""

    # ---------------- CLEANING CELL ----------------
    cleaning = f"""
# drop duplicates
df = df.drop_duplicates()

# drop id columns
df = df.drop(columns={id_cols}, errors="ignore")
"""

    # ---------------- DATE PROCESSING ----------------
    date_processing = ""
    for col in date_cols:
        date_processing += f"""
df["{col}"] = pd.to_datetime(df["{col}"], errors="coerce")
df["{col}_year"] = df["{col}"].dt.year
df["{col}_month"] = df["{col}"].dt.month
df["{col}_day"] = df["{col}"].dt.day
"""

    # ---------------- TIME PROCESSING ----------------
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

    # ---------------- TARGET ----------------
    if target_column:
        target_split = f"""
# target column
target = "{target_column}"

X = df.drop(columns=[target])
y = df[target]
"""
    else:
        target_split = """
X = df.copy()
"""

    # ---------------- PIPELINE ----------------
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

    # ---------------- SPLIT ----------------
    if target_column:
        split = """
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42
)

print("Preprocessing complete")
"""
    else:
        split = """
print("Preprocessing complete (no target column)")
"""

    return [
        ("1️⃣ Imports", imports),
        ("2️⃣ Data Cleaning", cleaning),
        ("3️⃣ Feature Engineering", feature_engineering),
        ("4️⃣ Target Split", target_split),
        ("5️⃣ Preprocessing Pipeline", pipeline),
        ("6️⃣ Train Test Split", split),
    ]