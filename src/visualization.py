import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def _clean_fig(fig):
    fig.tight_layout()
    return fig


def plot_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 3.6))
    missing.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    return _clean_fig(fig)


def plot_correlation(df):
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax, square=True, cbar=True)
    ax.set_title("Correlation Heatmap")
    return _clean_fig(fig)


def plot_target_distribution(df, target):
    if target is None or target not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(7, 3.6))
    df[target].value_counts().head(20).plot(kind="bar", ax=ax)
    ax.set_title(f"Target Distribution: {target}")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    return _clean_fig(fig)


def plot_numeric_distribution(df):
    numeric_cols = list(df.select_dtypes(include="number").columns[:4])
    figs = []

    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        figs.append(_clean_fig(fig))

    return figs