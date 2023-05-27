import itertools
import os
from typing import List
import time, re
import pandas as pd
import numpy as np
import imblearn

import jpype.imports
import imblearn
from jpype.types import *

from .pattern import Pattern
from packages.mymemoize import memoize


MAX_HEAP_SIZE_MB = 14 * 1024
print(f"Maximum heap size: {MAX_HEAP_SIZE_MB} MB")

MODULE_DIR = os.path.dirname(__file__)
JAR_FILE = os.path.join(MODULE_DIR, 'jars/bicpams_5.jar')
OUTPUT_FILE = os.path.join(MODULE_DIR, f'output/result_{time.time_ns()}.txt')

if not jpype.isJVMStarted():
    jpype.startJVM(f"-Xmx{MAX_HEAP_SIZE_MB}M", classpath=[JAR_FILE])

# Xmx flag -> maximum memory allocation, -Xmx8g = 8 Gb, default value is 246 MB
# Xms flag -> initial memory allocation, -Xms2048m = 2048 MB, no default value

# CHANGES IN BICPAMS ########
# - implemented reset(file) in BicResult to reset writer to avoid appending 
# output instead of fresh write; and enables setting the output file


from java.lang import String, Runtime, System
from java.io import PrintStream, File
from java.util import ArrayList, Arrays
from utils import BicReader, BicResult
from java.io import File
from domain import Dataset, Bicluster, Biclusters
from generator.BicMatrixGenerator import PatternType
from bicpam.bicminer.BiclusterMiner import Orientation
from bicpam.mapping import Itemizer
from bicpam.mapping.Itemizer import DiscretizationCriteria, FillingCriteria, NoiseRelaxation, NormalizationCriteria
from bicpam.closing import Biclusterizer, BiclusterMerger, BiclusterFilter
from bicpam.pminer.fim import ClosedFIM
from bicpam.pminer.spm import SequentialPM
from bicpam.pminer.spm.SequentialPM import SequentialImplementation
from bicpam.bicminer.coherent import AdditiveBiclusterMiner, MultiplicativeBiclusterMiner, SymmetricBiclusterMiner
from bicpam.bicminer.constant import ConstantBiclusterMiner, ConstantOverallBiclusterMiner
from bicpam.bicminer.order import OrderPreservingBiclusterMiner
from utils.others import CopyUtils
from performance.significance import BSignificance
import bicpam.bicminer

# Dictionaries that map transformations from string to Java objects
orientation_dict = {
    "rows": Orientation.PatternOnRows,
    "columns": Orientation.PatternOnColumns
}
bicminer_dict = {
    "additive": AdditiveBiclusterMiner,
    "constant": ConstantBiclusterMiner,
    "symmetric": SymmetricBiclusterMiner,
    "constant_overall": ConstantOverallBiclusterMiner,
    "multiplicative": MultiplicativeBiclusterMiner,
    "order_perserving": OrderPreservingBiclusterMiner
}
normalization_dict = {
    "column": NormalizationCriteria.Column
}
discretization_dict = {
    "normal_distribution": DiscretizationCriteria.NormalDist
}
noise_dict = {
    "optional": NoiseRelaxation.OptionalItem
}
filling_dict = {
    "remove": FillingCriteria.RemoveValue
}

translator_dict = {
    'orientation': orientation_dict,
    'bicminer': bicminer_dict,
    'normalization': normalization_dict,
    'discretization': discretization_dict,
    'noise_relaxation': noise_dict,
    'filling_criteria': filling_dict,
}


def get_java_object(object_type: str, key: str):
    """Return Java object corresponding to object_type and key arguments"""

    try:
        dictionary = translator_dict[object_type]
    except KeyError:
        raise ValueError(f"Parameter {{object_type}} '{object_type}' is invalid, "
                         f"must be one of the following: {list(translator_dict.keys())}")

    try:
        obj = dictionary[key.lower()]
    except KeyError:
        raise ValueError(f"Parameter {{key}} '{key}' is invalid, "
                         f"must be one of the following: {list(dictionary.keys())}")

    return obj


# Functions to execute biclustering ####################################################################################

def read_dataset(data: str | pd.DataFrame, class_index: int | str = -1, intscores=None, nr_labels=None):
    """Gets java object domain.Dataset from either a given path or a pd.DataFrame"""

    if isinstance(data, str):  # if receives column name, load columns to find index
        return read_dataset_path(path=data, class_index=class_index)
    elif isinstance(data, pd.DataFrame):
        return read_dataset_dataframe(df=data, class_index=class_index, intscores=intscores,
                                      nr_labels=nr_labels)
    else:
        raise ValueError(f'Argument {{data}} must be either str or pd.DataFrame, not "{type(data)}"')


def read_dataset_path(path: str, class_index: int | str = -1) -> Dataset:
    """Read java object domain.Dataset from given path, sets class according to class_index (default -1)"""

    if isinstance(class_index, str):  # if receives column name, load columns to find index
        class_index = get_index_by_col_name(path, class_index)

    path = String(path)
    if path.contains(".arff"):
        data = Dataset(BicReader.getInstances(path), class_index)
    else:
        data = Dataset(BicReader.getConds(path, 1, ","), BicReader.getGenes(path, ","),
                       BicReader.getTable(path, 1, ","), class_index)

    return data


def read_dataset_dataframe(df: pd.DataFrame, class_index: int | str = -1, intscores=None, 
                           nr_labels=None) -> Dataset:
    """Transform pd.DataFrame to java object domain.Dataset, sets class according to class_index (default -1)"""

    MISSING = 999999.0
    columns = df.columns.tolist()

    if isinstance(class_index, str):  # if receives column name, find index
        if class_index in columns:
            class_index = columns.index(class_index)
        else:
            raise ValueError(f'Class column "{class_index} not in data columns."')

    rows = df.index.tolist()
    value_table = np.around(df.fillna(MISSING).values, decimals=2)
    #print(t)


    # convert to Java
    columns = ArrayList(columns)
    rows = ArrayList(rows)
    value_table = jpype.JArray(jpype.JDouble, 2)(value_table.tolist())

    data = Dataset(columns, rows, value_table, class_index)
    if intscores is not None:
        for i in range(len(intscores)):
            for j in range(len(intscores[i])):
                intscores[i][j] = JInt(intscores[i][j])
            intscores[i] = ArrayList(intscores[i])
            #print(intscores[i])
        #intscores = jpype.JArray(intscores)
        data.itemize( ArrayList(intscores), nr_labels)
        #data.intscores = ArrayList(intscores)
    return data


def itemizer(data: Dataset, nr_labels: int, symmetries: bool, normalization: str, discretization: str,
             noise_relaxation: str, filling_criteria: str, verbose=0) -> Dataset:
    """Discretizes numeric dataset according to arguments"""

    if verbose > 0: print("Runnin bicpy.itemizer")

    nr_labels = int(nr_labels) 
    normalization = get_java_object('normalization', normalization)
    discretization = get_java_object('discretization', discretization)
    noise_relaxation = get_java_object('noise_relaxation', noise_relaxation)
    filling_criteria = get_java_object('filling_criteria', filling_criteria)

    if verbose > 0: print("\tRunnin Itemizer.run")
    data = Itemizer.run(data, nr_labels, symmetries,
                        normalization,
                        discretization,
                        noise_relaxation,  # multi-item assignments
                        filling_criteria)
    if verbose > 0: print("End of Itemizer.run")
    return data


def get_pminer(data: Dataset, pattern_type: str, orientation: str, min_biclusters: int, min_columns: int,
               min_lift: float, to_posthandle: bool) -> bicpam.bicminer:
    """Creates pattern miner according to arguments"""

    if to_posthandle:
        min_overlap_merging = 0.8
        min_similarity = 0.5
        posthandler = Biclusterizer(BiclusterMerger(min_overlap_merging),
                                    BiclusterFilter(min_similarity))
    else:
        posthandler = Biclusterizer()

    miner = get_java_object('bicminer', pattern_type)
    orientation = get_java_object('orientation', orientation)

    if pattern_type == "order_perserving":
        pminer = SequentialPM()
        pminer.algorithm = SequentialImplementation.PrefixSpan
    else:
        pminer = ClosedFIM()
    pminer.inputMinNrBics(min_biclusters)
    pminer.inputMinColumns(min_columns)
    pminer.setClass(data.classValues, min_lift)
    # pminer.setTargetClass(targetClass,classSuperiority);
    bicminer = miner(data, pminer, posthandler, orientation)
    return bicminer


def run_bicpam(data: Dataset, bicminer: bicpam.bicminer, nr_iterations: int, orientation: str,
               remove_percentage: float, verbose=0) -> None:
    """Runs the biclustering algorithm, applying {bicminer} to the {data} for {nr_iterations} iterations

    The patterns returned by the biclustering are saved to the file './output/result.txt'
    """
    start = time.time()
    BicResult.reset(OUTPUT_FILE)  # to avoid appending new result to the same file as last result
    orientation = get_java_object('orientation', orientation)
    duration = time.time()
    bics = Biclusters()
    originalIndexes = CopyUtils.copyIntList(data.indexes)
    originalScores = CopyUtils.copyIntList(data.intscores)

    if verbose > 0: print("### Running biclustering")
    for i in range(0, nr_iterations):
        if verbose > 0:
            print(f"\n## Iteration {i + 1} out of {nr_iterations}")
            print(f"Current heap size: {Runtime.getRuntime().totalMemory() / (1024 ** 2)} MB")
            print("# Mining biclusters")
        iBics = bicminer.mineBiclusters()
        if verbose > 0: print("# Removing")
        data.remove(iBics.getElementCounts(), remove_percentage, 1)
        bicminer.setData(data)
        bics.addAll(iBics)
    data.indexes = originalIndexes
    data.intscores = originalScores
    bics.computePatterns(data, orientation)
    BSignificance.run(data, bics)
    bics.orderPValue()

    duration = time.time() - duration
    if verbose > 0: print(f"Time: {duration} s")
    BicResult.println("FOUND BICS:" + str(bics.toString(data.rows, data.columns)))
    for bic in bics.getBiclusters():
        BicResult.println(bic.toString(data) + "\n\n")
    print(f"run_bicpam duration: {time.time()-start}")


def get_patterns() -> list:
    """Reads the patterns returned by the biclustering from a file and turns them into a list of Pattern objects"""

    patterns_list = []

    with open(OUTPUT_FILE, "r") as file:
        output = file.read()

    patterns = output.split('\n\n')[0].split('\n')[1:]
    for p in patterns:
        args = Pattern.parser(p)
        pattern = Pattern(args["columns"], args["rows"], args["values"], args["pvalue"], args["lift"])
        patterns_list.append(pattern)

    os.remove(OUTPUT_FILE)

    return patterns_list


def balance_data(data: pd.DataFrame, balancing='smote', class_index: int | str = -1) -> pd.DataFrame:
    """"Balances the data according to its class using the method defined by {balancing}"""

    balancers = {
        'smote': imblearn.over_sampling.SMOTE(random_state=42)
    }
    try:
        balancer = balancers[balancing]
    except KeyError:
        raise ValueError(f'Balancing {balancing} not supported. Must be one of the following: {list(balancers.keys())}')

    columns = data.columns.tolist()
    if isinstance(class_index, str):  # if receives column name, find index
        if class_index in columns:
            class_index = columns.index(class_index)
        else:
            raise ValueError(f'Class column "{class_index} not in data columns."')

    y = data.iloc[:, class_index].values
    X = data.drop(columns=data.columns[class_index]).values
    X_res, y_res = balancer.fit_resample(X, y)

    cols = data.columns.tolist()
    class_col = cols.pop(class_index)
    new_data = pd.DataFrame(X_res, columns=cols)
    new_data[class_col] = y_res

    new_index = data.index.tolist()
    left = new_data.shape[0] - len(new_index)
    left = ['artificial_' + str(i) for i in range(0, left)]
    new_index += left
    new_data = new_data.set_index(pd.Index(new_index))
    return new_data



@memoize(table_name='bicpy_run', kwarg_ignore=('verbose',))
def run(params: dict, data: pd.DataFrame, discretize=True, intscores=None, verbose=0, memoization=True, 
        class_index: str | int = -1) -> List[Pattern]:
    """Run the biclustering algorithm and return a list of Patterns"""
    
    print('Running bicpy.run')
    if verbose > 0:
        print(f"Maximum heap memory of JVM: {Runtime.getRuntime().maxMemory() / (1024 ** 2)} MB")
    original_out = System.out

    try:
        # changes the SystemOut to avoid prints from the Java code
        if verbose == 0: System.setOut(PrintStream(File("/dev/null")))  # NUL for windows, /dev/null for unix
        params['min_lift'] = round(params['min_lift'], 2)
        if params['balancing']:
            data = balance_data(data)

        
        if discretize:
            data = read_dataset(data, class_index)
            discrete_data = itemizer(data, params["nr_labels"], params["symmetries"], params["normalization"],
                                    params["discretization"], params["noise_relaxation"], params["filling_criteria"], 
                                    verbose=verbose)
        else:
            if intscores is None:
                raise ValueError("When discretize is False, intscores cannot be None.")
            discrete_data = read_dataset(data, class_index, intscores=intscores, 
                                         nr_labels=params["nr_labels"])
            # print("AAAAAAAAAAAAAAAAAAAAAA")
            # print(type(discrete_data))
            # print(len(discrete_data.intscores))

        if verbose>0: print('get_pminer')
        bicminer = get_pminer(discrete_data, params["pattern_type"], params["orientation"], params["min_biclusters"],
                              params["min_columns"], params["min_lift"], params['to_posthandle'])
        if verbose>0: print("run bicpam")
        run_bicpam(discrete_data, bicminer, params["nr_iterations"], params["orientation"], params["remove_percentage"],
                   verbose)
        if verbose>0: print('get patterns')
        patterns = get_patterns()
        #print(len(patterns))
        # result = {"patterns": patterns, "parameterization": params}
    finally:
        if verbose>0: print('restore out')
        System.setOut(original_out)  # restore the SystemOut after finishing the Java code
    if verbose>0: print('end of bicpy run')
    return patterns


# Auxiliary Function #########################################################

def parse_string_dataset(text: str) -> pd.DataFrame:
    """ Transform string representation in a DataFrame

    Receives a string as returned by the method .toString() of object domain.Dataset and returns the corresponding
    pd.DataFrame
    """
    text = str(text).replace(' ', '').strip()

    columns = re.findall("Courses:\[(.*?)\]", text)[0].split(",")

    rows = []
    indexes = []
    for line in text.split('\n')[2:]:  # skip first 2 lines corresponding to listing of rows and columns
        split_line = line.strip().split("=>")
        index = split_line[0]
        values = split_line[1].split('|')
        values = list(filter(None, values))  # remove empty strings from list
        values = [v.replace(',', '.') for v in values]  # if number is '3,14' change to '3.14'
        values = list(map(float, values))  # transform from string to float
        rows.append(values)
        indexes.append(index)

    df = pd.DataFrame(rows, columns=columns, index=indexes)
    return df


@memoize(table_name='bicpy_discretize_data', kwarg_ignore=('verbose',))
def discretize_data(data: pd.DataFrame, parameterization: dict, verbose=0, class_index: str | int = -1) -> pd.DataFrame:
    """Discretizes data available at {data_path} using bicpy.itemizer

    In the process removes the target variable. It is not important since this auxiliary dataset is used to calculate
    new variables without altering the target
    """
    start = time.time()
    print("Running bicpy.discretize data")
    if verbose > 0: print("Running bicpy.discretize data")

    original_out = System.out
    try:
        if verbose == 0: System.setOut(PrintStream(File("/dev/null")))  # NUL for windows, /dev/null for unix
        data = read_dataset(data, class_index)
        discrete_data = itemizer(data, parameterization["nr_labels"], parameterization["symmetries"],
                                 parameterization["normalization"], parameterization["discretization"],
                                 parameterization["noise_relaxation"], parameterization["filling_criteria"],verbose=verbose)
        intscores = discrete_data.intscores
        discrete_data_text = discrete_data.toString(False)
        df = parse_string_dataset(discrete_data_text)
    finally:
        System.setOut(original_out)

    # convert intscores to list of lists
    intscores_n = []
    for i in range(len(intscores)):
        ints = list(intscores[i])
        ints = [int(j) for j in ints]
        intscores_n.append(ints)

    print(f"discretize_data duration: {time.time()-start}")
    return df, intscores_n


def create_parameterizations(p: dict) -> List[dict]:
    """Receives a dict with lists as values and returns list of dictionaries with possible combinations of these vals"""

    keys = []
    vals = []
    for key, item in p.items():
        if isinstance(item, list):
            keys.append(key)
            vals.append(item)
    combinations = list(itertools.product(*vals))

    parameterizations = []
    for comb in combinations:
        d = p.copy()
        for key, value in zip(keys, comb):
            d[key] = value
        parameterizations.append(d)
    return parameterizations


def add_default_parameterization(param: dict, default_param: dict) -> dict:
    """Adds to default parameterization the keys-values in param"""

    for default_key in default_param:
        if default_key not in param:
            param[default_key] = default_param[default_key]

    return param


def get_index_by_col_name(data_path: str, column_name: str) -> int:
    """Get index of a column in a dataframe by its name"""

    columns = pd.read_csv(data_path, index_col=0, nrows=0).columns.tolist()
    if column_name in columns:
        class_index = columns.index(column_name)
    else:
        raise ValueError(f'Class column "{column_name} not in data columns."')
    return class_index
