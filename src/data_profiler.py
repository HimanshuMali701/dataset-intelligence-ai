def profile_dataset(df):

    profile = {}

    profile["rows"] = df.shape[0]
    profile["columns"] = df.shape[1]

    profile["missing_values"] = df.isnull().sum().to_dict()

    profile["duplicate_rows"] = df.duplicated().sum()

    profile["dtypes"] = df.dtypes.astype(str).to_dict()

    profile["numeric_columns"] = list(df.select_dtypes(include="number").columns)

    profile["categorical_columns"] = list(df.select_dtypes(include="object").columns)

    return profile