def generate_suggestions(df, quality, target=None):

    suggestions = []

    # -----------------------------
    # Missing values
    # -----------------------------
    for col, val in quality["missing"].items():

        if col == target:
            suggestions.append(
                f"Target column '{col}' has missing values. Consider removing rows with missing target"
            )
            continue

        if val > 40:
            suggestions.append(
                f"Column '{col}' has high missing values ({val:.1f}%). Consider dropping or advanced imputation"
            )

        elif val > 0:
            if df[col].dtype == "object":
                suggestions.append(
                    f"Fill missing values in '{col}' using mode"
                )
            else:
                suggestions.append(
                    f"Fill missing values in '{col}' using median"
                )

    # -----------------------------
    # duplicates
    # -----------------------------
    if quality["duplicates"] > 0:
        suggestions.append("Remove duplicate rows")

    # -----------------------------
    # constant columns
    # -----------------------------
    for col in quality["constant_columns"]:
        if col != target:
            suggestions.append(f"Drop constant column '{col}'")

    # -----------------------------
    # correlation
    # -----------------------------
    for col in quality["high_correlation"]:
        if col != target:
            suggestions.append(
                f"Remove highly correlated feature '{col}'"
            )

    # -----------------------------
    # imbalance
    # -----------------------------
    for col in quality["imbalance"]:
        if col == target:
            suggestions.append(
                f"Target '{col}' is imbalanced. Consider SMOTE or class weighting"
            )

    # -----------------------------
    # outliers
    # -----------------------------
    for col in quality["outliers"]:
        if col != target:
            suggestions.append(
                f"Handle outliers in '{col}' using IQR or clipping"
            )

    return suggestions