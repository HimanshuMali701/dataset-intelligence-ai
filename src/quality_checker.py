import pandas as pd
import numpy as np

def check_missing_values(df):
    missing = df.isnull().mean() * 100
    return missing[missing > 0].sort_values(ascending=False)


def check_duplicates(df):
    return df.duplicated().sum()


def check_constant_columns(df):
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    return constant_cols


def check_high_correlation(df, threshold=0.9):
    numeric_df = df.select_dtypes(include=np.number)

    corr_matrix = numeric_df.corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    high_corr = [
        column for column in upper.columns
        if any(upper[column] > threshold)
    ]

    return high_corr


def check_class_imbalance(df):
    imbalance = {}

    for col in df.select_dtypes(include="object").columns:
        counts = df[col].value_counts(normalize=True)

        if counts.iloc[0] > 0.8:
            imbalance[col] = counts.to_dict()

    return imbalance


def detect_outliers(df):
    outlier_counts = {}

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()

        if outliers > 0:
            outlier_counts[col] = int(outliers)

    return outlier_counts


def run_quality_checks(df):

    report = {}

    report["missing"] = check_missing_values(df)
    report["duplicates"] = check_duplicates(df)
    report["constant_columns"] = check_constant_columns(df)
    report["high_correlation"] = check_high_correlation(df)
    report["imbalance"] = check_class_imbalance(df)
    report["outliers"] = detect_outliers(df)

    return report