import pickle
from sqlite3 import InterfaceError, ProgrammingError
import pandas as pd
import tabulate
import sqlite3, os


class Cache:
    # TODO: dont load object in results for display
    MODULE_DIR = os.path.dirname(__file__)
    cache_path = os.path.join(MODULE_DIR, 'mymemoize.db')
    max_representation_string_size = 100
    max_width_tabulate = 70

    key_col = 'key'
    result_col = 'result'
    insert_string = "INSERT into mycache values (?, ?, ?)"

    def __init__(self, table_name: str):
        self.table = table_name
        # check if DB exists, open connection
        if not os.path.isfile(self.cache_path):
            print(f'Cache at {self.cache_path} does not exist, creating it.')
        self.con = sqlite3.connect(self.cache_path, detect_types=sqlite3.PARSE_DECLTYPES)
        # self.con = sqlite3.connect(':memory:')  # for dev

        # check if table exists, if not, create it
        res = self.con.execute(f"SELECT name FROM sqlite_master WHERE name='{self.table}'")
        if res.fetchone() is None:
            print(f'Table "{self.table}" does not exist, creating it. ')
            self.con.execute(f"CREATE TABLE {self.table}({self.key_col} PRIMARY KEY, {self.result_col} result_type)")
        sqlite3.register_converter("result_type", pickle.loads)

    def get(self, key):
        string = f"SELECT {self.result_col} FROM {self.table} WHERE {self.key_col}='{key}'"
        res = self.con.execute(string)
        res = res.fetchone()
        if res is None:
            return res
        else:
            return res[0]

    def add(self, key, result):
        query = f"INSERT INTO {self.table} VALUES (?, ?)"
        try:
            with self.con:
                self.con.execute(query, (key, result))
                # print('did it')
        except (InterfaceError, ProgrammingError):
            # print('interface error')
            sqlite3.register_adapter(result.__class__, pickle.dumps)
            with self.con:
                self.con.execute(query, (key, result))
                # print('did it after interface error')


    def close_connection(self):
        self.con.close()


    def reset(self):
        confirmation = str(input(f"Are you sure you want to delete table {self.table} ?"))
        if confirmation.lower() == "y":
            query = f"DROP table {self.table}"
            self.con.execute(query)
            print("Deleted")
        else:
            print("Canceled deleting")

            


    def truncate_string_representation(self, representation: list) -> list:
        new_repr = []
        for key, result in representation:
            key = key[:self.max_representation_string_size]
            result = str(result)[:self.max_representation_string_size]
            new_repr.append((key, result))
        return new_repr

    
    def get_cache_as_df(self) -> pd.DataFrame:
        query = f"SELECT * FROM {self.table}"
        res = self.con.execute(query)
        res = res.fetchall()
        df = pd.DataFrame(res, columns=['key', 'result'])
        return df


    def __str__(self):
        query = f"SELECT * FROM {self.table}"
        res = self.con.execute(query)
        res = res.fetchall()
        representation = self.truncate_string_representation(res)
        if representation == []:
            representation = [[], []]
        representation = tabulate.tabulate(representation, tablefmt='fancy_grid',
                                           headers=[self.key_col, self.result_col],
                                           maxcolwidths=self.max_width_tabulate)
        return representation

    def __repr__(self):
        return f"Cache(table_name={self.table})"
