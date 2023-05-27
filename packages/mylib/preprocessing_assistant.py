from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
import pandas as pd


"""Definition
Important:
    - ID as index and not column
    - identify and replace values synonym of NaN 
"""
def preprocessing_assistant(df: pd.DataFrame, confirm_all = False, to_ignore = []) -> None:
    ignored_df = df.loc[:, to_ignore]
    df = df.drop(columns = to_ignore)
    #Description of columns
    display( describe_df(df) )
    
    #Remove rows with all NaN and warn about high NaN-percentage rows
    df = remove_rows_nan(df, confirm_all)
    
    #Remove columns with all NaN and warn about single-value columns
    df = remove_columns_nan(df, confirm_all)
    
    #Identify possible multiple IDs and choose which ones to maintain
    df = remove_id_like_columns(df, confirm_all)
    
    #Identify columns with extremely low variance
    df = remove_low_variance_columns(df, confirm_all)
    
    #Calculate correlations between columns to identify redudant columns
    df = remove_redudant_columns(df, confirm_all)
    
    df[ignored_df.columns] = ignored_df
    
    return df


"""Removes rows with all NaNs and warns and possibly removes rows with NaN-percentage > threshold"""
def remove_rows_nan(df: pd.DataFrame, confirm_all: bool = False, threshold : float = 0.9) -> pd.DataFrame:
    print("\n##### Removing rows with all NaNs")
    
    #remove rows with all NaNs
    df_no_nan = df.dropna(axis = "index", how = "all")
    diff = pd.concat([df,df_no_nan]).drop_duplicates(keep=False)
    if diff.empty:
        print("No all NaN rows")
    else:
        print(f"Indexes of all-NaN rows removed: {diff.index.tolist()}")
        
    #warn and possibly remove rows with high percentage of NaNs
    percentage_of_nan = df_no_nan.isna().sum(axis="columns")/len(temp.columns)
    indexes = percentage_of_nan[percentage_of_nan>0.9].index
    if not indexes.empty:
        print(f"Rows with percentage of NaN greater than {threshold}")
        display(df_no_nan.loc[indexes,:])
        res = str(input("Want to remove these rows? y/n")).lower() if not confirm_all else "y"
        if res == "y":
            df_no_nan = df_no_nan.drop(index=indexes)
            print("Removed")
        else:
            print("Not removed")
    return df_no_nan
    

"""Removes columns with all NaNs and warns and possibly removes columns with a single value"""
def remove_columns_nan(df: pd.DataFrame, confirm_all: bool = False) -> pd.DataFrame:
    print("\n##### Removing columns with all NaNs")
    
    #remove columns with all NaNs
    df_no_nan = df.dropna(axis="columns", how="all")
    diff = set(list(set(df.columns) - set(df_no_nan.columns)) + list(set(df.columns) - set(df_no_nan.columns)))
    if len(diff)==0:
        print("No all NaN columns")
    else:
        print(f"All-NaN columns removed: {list(diff)}")
        
    #warn and possibly remove columns with a single value
    print()
    val_counts = {}
    cols_to_remove = []
    for c in df_no_nan.columns:
        value_count = df_no_nan[c].value_counts()
        if len(value_count) <= 1:
            res = str(input(f"Column '{c}' only has value: {value_count.keys().tolist()}\t Remove? y/n\t")) if not confirm_all else "y"
            if res == "y":
                cols_to_remove.append(c)
                print(f"Removed {c}")
            else: 
                print("Not removed")
    df_no_nan = df_no_nan.drop(columns=cols_to_remove)
    return df_no_nan


"""Identifies and warns about redudant columns, this is, columns with high correlation"""
def remove_redudant_columns(df: pd.DataFrame, confirm_all: bool = False) -> pd.DataFrame:
    #TODO: taking a long time to compute compared to Pearson
    def chisquare_wrapper(x: pd.Series, y: pd.Series, min_freq = 5) -> float: 
        contigency_table = pd.crosstab(x, y)
        #if (~(contigency_table[contigency_table<min_freq].isna())).any().any():  #if any frequency count is < min_freq
        #    return 1
        results = chi2_contingency(contigency_table)
        return results[1] #p-value
    
    #methods to use in pandas.DataFrame.corr()
    correlation_per_data_type = {"numeric": "pearson", "categorical": chisquare_wrapper, "boolean": "pearson"}
    
    columns_by_type = divide_data_by_type(df)
    for col_type in columns_by_type:
        is_categorical = (col_type == "categorical")
        columns = columns_by_type[col_type]
        method = correlation_per_data_type[col_type]
        
        if is_categorical:
            a = encode_variables(df[columns])
            t_df = encode_variables(df[columns]).corr(method=method)
        else:
            t_df = df[columns].corr(method=method)
        
        correlated_groups, upper_triangle = get_redudant_vars_from_corr_matrix(t_df, is_categorical)
        print(f"# In {col_type} variables")
        for group in correlated_groups:
            if not set(group).issubset(set(df.columns)):   #if one of the columns was already deleted
                continue
            #order triangle
            #https://stackoverflow.com/questions/45909776/sort-rows-of-a-dataframe-in-descending-order-of-nan-counts
            temp_triangle = upper_triangle.loc[group, group].copy()
            temp_triangle = temp_triangle.iloc[temp_triangle.isnull().sum(axis=1).mul(1).argsort()]
            temp_triangle = temp_triangle.loc[:, temp_triangle.index.to_list()]
            print("Correlations:")
            display(temp_triangle)
            print("Head of DataFrame:")
            display(df.loc[:, group].head(5))
            res = str(input("Select columns to remove? y/n"))
            if res.lower() == "y":
                print("Input numbers of columns to remove separeted by ','")
                options = '\n'.join([str(i) + "-" + c for i,c in enumerate(temp_triangle.columns)]) + '\n'
                res = str(input(options)).replace(" ", "").split(',')
                index_of_cols_to_remove = list(map(int,res))
                cols_to_remove = temp_triangle.columns[index_of_cols_to_remove]
                print("Removing columns: ", cols_to_remove)
                df = df.drop(columns=cols_to_remove)
                
    return df


        
        
def get_redudant_vars_from_corr_matrix(df: pd.DataFrame, pvalue = False, threshold_corr: float=0.9, threshold_pvalue: float=0.001):
    upper_triangle = df.where(np.triu(np.ones(df.shape), k=1).astype('bool'))
    display(upper_triangle)

    correlated_groups = []
    for column in upper_triangle.columns:
        if not pvalue:
            sel = upper_triangle[abs(upper_triangle[column]) > threshold_corr].index.tolist()
        else:
            sel = upper_triangle[abs(upper_triangle[column]) < threshold_pvalue].index.tolist()
        if len(sel) > 0:
            subset = check_if_exists_subset(sel, correlated_groups)
            try:
                correlated_groups.remove(subset)
            except ValueError:
                pass
            correlated_groups.append([column] + sel)
    
    return correlated_groups, upper_triangle

        


"""Receives a list L1 and a list of lists L2, checks if in the list of lists L2 there are subsets of L1 
Returns subset if existent, otherwise None"""
def check_if_exists_subset(list1: list, list_of_lists: list) -> bool:
    for li in list_of_lists:
        if set(li).issubset(set(list1)):
            return li
    return None

    
"""Encodes numerically all the given DataFrame"""
def encode_variables(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        df[column] = LabelEncoder().fit_transform(df[column])
    return df
    

"""Divides DataFrame columns by dtype, returning a dict with keys 'numeric', 'categorical' and 'boolean'"""
def divide_data_by_type(df: pd.DataFrame) -> dict:
    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(include="object_").columns.tolist()
    boolean_columns = df.select_dtypes(include="bool_").columns.tolist()
    
    columns_not_included = set(df.columns) - set(numerical_columns) - set(categorical_columns) - set(boolean_columns)
    if len(columns_not_included) > 0:
        raise RuntimeError(f"Columns {columns_not_included} not included in division.")
        
    return {"numeric": numerical_columns, "categorical": categorical_columns, "boolean": boolean_columns}
    
    
def remove_id_like_columns(df, confirm_all = False, threshold = 0.99):
    print(f"\n##### Looking for ID-like columns, columns with {threshold} of the same value")
    nrows = df.shape[0]
    for col in df.columns:
        n_unique = len(df[col].unique())
        if n_unique/nrows > threshold:
            print(f"Column {col} has {n_unique} unique values out of {nrows} rows. Example '{df.loc[:,col].iloc[0]}'")
            res = str(input("Delete this column? y/n")) if not confirm_all else "y"
            if res.lower() == "y":
                df = df.drop(columns=col)
                print("Deleted")
    return df
    
    
def remove_low_variance_columns(df: pd.DataFrame, confirm_all = False, threshold = 0.95) -> pd.DataFrame:
    print(f"\n##### Looking for columns with low variance, when a single value corresponds to {threshold} of the total values")
    nrows = df.shape[0]
    for col in df.columns:
        highest_count_of_value = df[col].value_counts().iloc[0]
        if highest_count_of_value/nrows > threshold:
            print(f"Column '{col}' has distribution of values:\n{df[col].value_counts()}")
            res = str(input("Delete column? y/n"))if not confirm_all else "y"
            if res.lower() == "y":
                df = df.drop(columns=col)
                print("Deleted")
    return df


