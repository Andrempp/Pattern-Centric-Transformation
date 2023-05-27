import re

# Pattern class ########################################################################################################
class Pattern:
    """
        A class used to represent a Pattern returned by the biclustering algorithm

        Parameters
        ----------
        columns : list
            a list of strings with the column names of the pattern
        rows : list
            a list of strings with the row names/indexes of the pattern
        values : list
            a list of numbers with the values corresponding to each column of the pattern
        pvalue : float
            the p-value associated with the pattern
        lift : list
            the lifts associated with the pattern for each class
    """

    def __init__(self, columns: list, rows: list, values: list, pvalue: float, lift: list):
        self.columns = columns
        self.rows = rows
        self.values = values
        self.pvalue = pvalue
        self.lift = lift

    @staticmethod
    def parser(text: str) -> dict:
        """Parses a text returned by the biclustering and returns the Pattern's attributes in a dictionary"""

        properties = {}
        properties["columns"] = re.findall("Y=\[(.*?)\]", text)[0].split(",")
        properties["rows"] = re.findall("X=\[(.*?)\]", text)[0].split(",")
        temp = re.findall("I=\[(.*?)\]", text)[0].split(",")
        properties["values"] = list(map(int, temp))

        pvalue = re.findall("pvalue=([0-9]+\.[A-z0-9-]+)", text)
        properties['pvalue'] = None if len(pvalue) == 0 else float(pvalue[0])

        lift = re.findall("Lifts=\[(.*?)\]", text)
        properties['lift'] = None if len(lift) == 0 else list(map(float, lift[0].split(', ')))  # get both lifts

        return properties

    def __str__(self):
        string = ""
        string += f"Columns: {self.columns}\n"
        string += f"Rows: {self.rows}\n"
        string += f"Values: {self.values}\n"
        string += f"p-value: {self.pvalue}\n"
        string += f"Lift: {self.lift}\n"
        return string

    def __repr__(self):
        return f"Pattern(columns={self.columns}, rows={self.rows}, values={self.values}, pvalue={self.pvalue}, " \
               f"lift={self.lift})"