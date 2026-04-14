import numpy as np

def calculate_health_score(df, quality):

    scores = {}

    # -----------------------
    # Missing values score
    # -----------------------
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()

    completeness = 100 - ((missing_cells / total_cells) * 100)
    scores["completeness"] = round(completeness, 2)

    # -----------------------
    # Duplicate score
    # -----------------------
    duplicates = quality["duplicates"]
    duplicate_score = 100 - ((duplicates / len(df)) * 100)
    scores["duplicates"] = round(duplicate_score, 2)

    # -----------------------
    # correlation score
    # -----------------------
    corr_issues = len(quality["high_correlation"])
    corr_score = 100 - (corr_issues * 10)
    scores["correlation"] = max(corr_score, 0)

    # -----------------------
    # outlier score
    # -----------------------
    total_outliers = sum(quality["outliers"].values()) if quality["outliers"] else 0
    outlier_score = 100 - ((total_outliers / len(df)) * 100)
    scores["outliers"] = round(outlier_score, 2)

    # -----------------------
    # imbalance score
    # -----------------------
    imbalance_count = len(quality["imbalance"])
    imbalance_score = 100 - (imbalance_count * 15)
    scores["imbalance"] = max(imbalance_score, 0)

    # -----------------------
    # Final score
    # -----------------------
    final_score = np.mean(list(scores.values()))

    return round(final_score, 2), scores