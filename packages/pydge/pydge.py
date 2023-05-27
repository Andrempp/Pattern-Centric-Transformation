import json
import pandas as pd
from typing import List, Union
import os
import subprocess
import time

from packages.mymemoize.mymemoize import memoize

MODULE_DIR = os.path.dirname(__file__)
R_SCRIPT_DIR = os.path.join(MODULE_DIR, 'r_dir/')

# Utils ###############################################################################
# Utility functions ###################################################################
#######################################################################################
# ola nina quero tratar de ti ya 
def execute(cmd):
    """Creates subprocess to run received command and yields the output as it is produced"""

    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


# for live use on dataframes
@memoize(table_name='pydge_deg_filtering', kwarg_ignore=('verbose',))
def deg_filtering(counts_df: pd.DataFrame, target: str, pvalue = None, filter_only = False, n_genes = None, verbose=1) -> List[str]:
    """Receives a dataframe with counts, executes DEG analysis, and returns the selected genes as list of str.
    
    Does not return the target variable.
    """
    print("running deg_filtering")
    if pvalue is None and n_genes is None and not filter_only:
        raise ValueError("If 'filter_only' is False then 'pvalue' or 'n_genes' must be set.")
    
    if pvalue is not None and filter_only:
        raise Warning(f"Parameter 'pvalue = {pvalue}' will be ignored due to 'filter_only = True'.")
    
    if n_genes is not None and filter_only:
        raise Warning(f"Parameter 'n_genes = {n_genes}' will be ignored due to 'filter_only = True'.")
    
    if pvalue is not None and n_genes is not None:
        raise Warning(f"Both 'pvalue' and 'n_genes' were set. 'pvalue' will be used and n_genes ignored")
    
    input_file = os.path.join(MODULE_DIR, f"input_deg_filtering_{time.time_ns()}.csv")
    output_file = os.path.join(MODULE_DIR, f"output_deg_filtering_{time.time_ns()}.csv")
    counts_df.to_csv(input_file)

    try:
        for output in execute(["Rscript", f'{R_SCRIPT_DIR}get_deg.R', input_file, output_file, target]):
            if verbose >= 1: print(output, end="")
        degs = pd.read_csv(output_file, index_col=0)
    finally:
        os.remove(input_file)
        os.remove(output_file)

    if filter_only: #return all genes, filtering already none with R script
        pvalue = 1
        filtered_degs = degs[degs['PValue']<=pvalue]
        genes = filtered_degs.index.tolist()
    elif pvalue is not None:
        filtered_degs = degs[degs['PValue']<=pvalue]
        genes = filtered_degs.index.tolist()
    else:
        filtered_degs = degs.sort_values(by=["PValue"]).iloc[:n_genes, :]
        genes = filtered_degs.index.tolist()
    return genes


# used by tcgahandler
def generate_dge_file(counts_df: pd.DataFrame, target: str, dge_file: str, verbose=1):
    
    print(f"Generating DGE file {dge_file}.")

    counts_file = os.path.join(MODULE_DIR, f"temp_get_dge_{time.time_ns()}.csv")
    print(counts_file)
    counts_df.to_csv(counts_file)

    try:
        for output in execute(["Rscript", f'{R_SCRIPT_DIR}generate_dge.R', counts_file, dge_file, target]):
            if verbose >= 1: print(output, end="")
    finally:
        os.remove(counts_file)


def get_dge(dge_file: str, verbose=1) -> pd.DataFrame:
    temp_file = os.path.join(MODULE_DIR, f"temp_get_dge_{time.time_ns()}.csv")

    if not os.path.isfile(dge_file):
        raise FileNotFoundError(f"DGE file received '{dge_file}' doesn't exist. Generate it using {generate_dge_file.__name__}.")
    
    for output in execute(["Rscript", f'{R_SCRIPT_DIR}get_all_dge.R', dge_file, temp_file]):
        if verbose >= 1: print(output, end="")

    all_dge_df = pd.read_csv(temp_file, index_col=0)
    os.remove(temp_file)
    return all_dge_df


# used by tcgahandler
def get_dge_data(input_df: pd.DataFrame, target: str, dge_file: str, pvalue_cutoff=0.05, filter_only=False, verbose=1) -> pd.DataFrame:
    """Reads the DataFrame from {input_file} and selects only the columns corresponding to DGEs and the {target}"""

    if filter_only: pvalue_cutoff = 1

    all_dge_df = get_dge(dge_file)

    filtered_dge_df = all_dge_df[all_dge_df['PValue']<=pvalue_cutoff]
    genes = filtered_dge_df.index.tolist()
    input_df = input_df.loc[:, genes + [target]]

    return input_df


def get_dge_data_n(input_df: pd.DataFrame, target: str, dge_file: str, number_dim: int, verbose=1) -> pd.DataFrame:
    all_dge_df = get_dge(dge_file)
    dge_df = all_dge_df.sort_values(by="PValue", ascending=True)[:number_dim]
    genes =  dge_df.index.tolist()
    input_df = input_df.loc[:, genes + [target]]
    return input_df





def get_stats(dge_file: str, target: str, cutoffs: list) -> pd.DataFrame:
    """Gets statistics for different p-value cutoffs in DGE selection

    Return a DataFrame with each row having the p-value cutoff, the total number of DEGs selected, the number of
    positive DEGs, and the number of negative DEGs
    Runs function 'get_dge' for each cutoff value
    """

    dge = get_dge(dge_file)
    
    dge_stats = pd.DataFrame(columns=["p-value cutoff", "Total DEGs", "Positive DEGs", "Negative DEGs"])
    for c in cutoffs:
        filtered_dge = dge[dge['PValue']<=c]
        total = filtered_dge.shape[0]
        pos = filtered_dge[filtered_dge['logFC']>=0].shape[0]
        neg = filtered_dge[filtered_dge['logFC']<0].shape[0]
        row = pd.DataFrame({"p-value cutoff": [c], "Total DEGs": [total],
                            "Positive DEGs": [pos], "Negative DEGs": [neg]})
        dge_stats = pd.concat([dge_stats, row], ignore_index=True)
    return dge_stats


###############################################################################################

# def get_all_dge(dge_file: str, verbose=0):
#     temp_file = os.path.join(MODULE_DIR, "temp_dge_all.csv")

#     for output in execute(["Rscript", f'{R_SCRIPT_DIR}get_all_dge.R', dge_file, temp_file]):
#         if verbose >= 1: print(output, end="")

#     all_dge_df = pd.read_csv(temp_file, index_col=0)
#     os.remove(temp_file)
#     return all_dge_df

# def get_dge_by_n(dge_file: str, n: int) -> pd.DataFrame:
#     all_dge = get_all_dge(dge_file=dge_file)
#     dge_by_n = all_dge.iloc[:n, :]
#     print(f'p-value for n={n} is {dge_by_n.iloc[-1,:]["PValue"]}')
#     return dge_by_n


# def get_dge(counts_df: pd.DataFrame, dge_file: str, target: str, pvalue_cutoff=0.05, verbose=1) -> pd.DataFrame:
#     """Get DataFrame with the genes and corresponding differential expression in respect to {target}

#     Each gene has a value: -1 for under-expression, 1 for over-expression, and 0 for no differential expression
#     If file with DGE data already exists, uses it, otherwise, generates it
#     Uses {counts_file} as input, and saves the DGE data in the {output_folder}
#     """
#     counts_file = os.path.join(MODULE_DIR, "temp_get_dge.csv")
#     temp_file = os.path.join(MODULE_DIR, "temp.csv")

#     try:
#         counts_df.to_csv(counts_file)
#         if not os.path.isfile(dge_file):  # check if file already exists or needs to be generated
#             print(f"DGE file {dge_file} doesn't exist, generating.")
#             for output in execute(["Rscript", f'{R_SCRIPT_DIR}generate_dge.R', counts_file, dge_file, target]):
#                 if verbose >= 1: print(output, end="")

#         for output in execute(["Rscript", f'{R_SCRIPT_DIR}get_corrected_dge.R', dge_file, temp_file, str(pvalue_cutoff)]):
#             if verbose >= 1: print(output, end="")

#         dge_df = pd.read_csv(temp_file, index_col=0)
#     finally:
#         os.remove(counts_file)
#         os.remove(temp_file)

#     return dge_df


# def get_filtered_genes(counts_df: pd.DataFrame, filtered_file: str, target: str, verbose = 0):
    
#     counts_file = os.path.join(MODULE_DIR, "temp_get_filtered_genes.csv")
#     try:
#         if not os.path.isfile(filtered_file):
#             print(f"Calculating filtered genes, saving at {filtered_file}")
#             counts_df.to_csv(counts_file)
#             for output in execute(["Rscript", f'{R_SCRIPT_DIR}filter_by_expr.R', counts_file, filtered_file, target]):
#                 if verbose >= 1: print(output, end="")
        
#         with open(filtered_file, 'r') as f:
#             filtered_dict = json.load(f)
#             filtered_genes = [g for g,k in filtered_dict.items() if k]

#     finally:
#         os.remove(counts_file)
#     return filtered_genes



# def get_dge_stats(counts_df: pd.DataFrame, dge_file: str, target: str, cutoffs: list) -> pd.DataFrame:
#     """Gets statistics for different p-value cutoffs in DGE selection

#     Return a DataFrame with each row having the p-value cutoff, the total number of DEGs selected, the number of
#     positive DEGs, and the number of negative DEGs
#     Runs function 'get_dge' for each cutoff value
#     """

#     dge_stats = pd.DataFrame(columns=["p-value cutoff", "Total DEGs", "Positive DEGs", "Negative DEGs"])

#     for c in cutoffs:
#         res = get_dge(counts_df, dge_file, target, pvalue_cutoff=c)
#         counts = res.iloc[:, 0].value_counts()
#         row = pd.DataFrame({"p-value cutoff": [c], "Total DEGs": [counts[1] + counts[-1]],
#                             "Positive DEGs": [counts[1]], "Negative DEGs": [counts[-1]]})
#         dge_stats = pd.concat([dge_stats, row], ignore_index=True)
#     return dge_stats


# def get_dge_dataset(input_df: pd.DataFrame, counts_df: pd.DataFrame, aux_file: str, target: str, pvalue_cutoff=0.05, filter_only=False) -> pd.DataFrame:
#     """Reads the DataFrame from {input_file} and selects only the columns corresponding to DGEs and the {target}"""

#     if filter_only:
#         genes = get_filtered_genes(counts_df, aux_file, target)
#     else:
#         res = get_dge(counts_df, aux_file, target, pvalue_cutoff=pvalue_cutoff)
#         genes = res[res.iloc[:, 0] != 0].index.tolist()
#         #df = pd.read_csv(input_file, index_col=0)
    
#     input_df = input_df.loc[:, genes + [target]]
#     return input_df

