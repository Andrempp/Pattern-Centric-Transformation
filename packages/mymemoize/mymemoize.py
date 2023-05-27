import hashlib
import functools
import pandas as pd
import json
import pickle
import codecs
from .cache import Cache
primitive = (int, float, str, bool, tuple, set)

def hash_dataframe(df: pd.DataFrame) -> str:
    # order the dataframe to ensure that the same dataframe with different orders returns the same value
    df = df.reindex(sorted(df.columns), axis=1)  # order by column name
    df = df.sort_index()  # order by index
    rows_hash = pd.util.hash_pandas_object(df)  # hash each row
    # hash_value = hash(str(rows_hash.values))  # hash the list of hash values for each row
    hash_value = hashlib.sha256(rows_hash.values).hexdigest()
    return hash_value


def generate_key(*args, **kwargs) -> str:
    def adapt_for_key(argument):
        # does not support nested dicts, lists, tuples, and sets
        # only supports dict keys of type string
        # not tested for lists of complex objects
        if isinstance(argument, primitive):
            return argument
        elif isinstance(argument, pd.DataFrame):
            return hash_dataframe(argument)
        elif isinstance(argument, dict):
            return json.dumps(argument, sort_keys=True, ensure_ascii=True)
        elif isinstance(argument, list) and isinstance(argument[0], primitive):
            return tuple(sorted(argument))
        elif argument is None:
            return "None"
        else:  # any other object is pickled and converted to string
            return codecs.encode(pickle.dumps(argument), "base64").decode()

    key = list()
    for arg in args:
        key.append(adapt_for_key(arg))
    for arg_key, arg_val in sorted(kwargs.items()):
        key.append(arg_key)
        key.append(adapt_for_key(arg_val))
    return str(tuple(key)).replace("'", "").replace('"', '')  # remove ' and " from string to avoid problems in SQL


def get_kwargs_for_key(keyword_args: dict, to_ignore: tuple):
    kwargs_key = keyword_args.copy()
    for key in to_ignore:
        kwargs_key.pop(key, None)
    return kwargs_key

def memoize(table_name: str, kwarg_ignore: tuple = (), ignore_self = False):
    """Memoize decorator, does not receive default argument values"""

    def decorator_memoize(func):
        cache = Cache(table_name=table_name)

        @functools.wraps(func)
        def wrapper_memoize(*args, **kwargs):
            args_key = args[1:] if ignore_self else args
            kwargs_key = get_kwargs_for_key(kwargs, kwarg_ignore)
            cache_key = generate_key(*args_key, **kwargs_key)
            # print("Cache key:", cache_key)
            # print(cache)
            if cache.get(key=cache_key) is None:
                # print('not in cache')
                result = func(*args, **kwargs)
                cache.add(key=cache_key, result=result)
            return cache.get(cache_key)

        return wrapper_memoize

    return decorator_memoize
