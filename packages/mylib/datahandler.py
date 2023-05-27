import pandas as pd
import numpy as np

def get_id_column(df: pd.DataFrame):
    nrows = df.shape[0]
    for column in df.columns:
        if len(df[column].value_counts()) == nrows:
            return column
    
    raise RuntimeError("No id column found") 
    
    
def reduce_numeric_64_to_32(df: pd.DataFrame, verbose = 0):
    max_int32 = np.iinfo(np.int32).max
    max_float32 = np.finfo(np.float32).max
    counter_int = 0
    counter_float = 0
    for col in df.columns:
        coltype = df[col].dtype
        if coltype == np.int64:
            if (df[col].abs() < max_int32).all():
                df[col] = df[col].astype("int32")
                counter_int +=1
        elif coltype == np.float64:
            if (df[col].abs() < max_float32).all():
                df[col] = df[col].astype("float32")
                counter_float+=1
                
    if verbose >= 1: print(f"Reduced {counter_int} ints and {counter_float} floats")
    return df