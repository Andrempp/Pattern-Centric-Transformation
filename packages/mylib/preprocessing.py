import pandas as pd


"""Creates a DataFrame with statistics about each column of the input DataFrame"""
def describe_df(df: pd.DataFrame, to_ignore: list = [], selected: list = []) -> pd.DataFrame:
    if len(selected) > 0:
        df = df[selected]
    df = df.drop(columns=to_ignore, errors="ignore")
    
    cols = df.columns
    numeric_cols = df.select_dtypes(include="number").columns
    
    stats = pd.DataFrame(index =  cols)
    stats["Type"] = df.dtypes
    stats["Fraction of NaN"] = round(df.isna().sum() / df.shape[0], 3)
    stats["Mode"] = df.mode().iloc[0,:]
    for c in cols:
        stats.loc[c, "Unique values"] = int(len(df[c].unique()))
    stats.loc[numeric_cols,"Mean"] = df[numeric_cols].mean()
    stats.loc[numeric_cols, "Standard dev."] = df[numeric_cols].std()
    stats.loc[numeric_cols, "Min"] = df[numeric_cols].min()
    stats.loc[numeric_cols, "Max"] = df[numeric_cols].max()
    stats = stats.sort_index()
    
    return stats
    
    
"""Removes columns of DataFrame that are all NaN
If remove_if_one_unique_val True, also removes columns with only one unique value"""
def remove_columns_all_nan(df: pd.DataFrame, remove_if_one_unique_val = False) -> pd.DataFrame:
    if not remove_if_one_unique_val:
        df = df.dropna(axis=1, how="all")
        return df
    
    for c in df.columns:
        if len(df[c].value_counts()) <= 1:
            df = df.drop(columns=[c])
    return df