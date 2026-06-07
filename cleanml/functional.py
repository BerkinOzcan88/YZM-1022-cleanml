from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from pandas.api.types import is_string_dtype


DataFrameFunction = Callable[[pd.DataFrame], pd.DataFrame]


def compose(*functions: DataFrameFunction) -> DataFrameFunction:
    """Combine multiple DataFrame functions into one function.

    Args:
        *functions: Functions that each take and return a DataFrame.

    Returns:
        A function that applies the given functions in order.
    """
    
    def composed(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        
        for function in functions:
            result = function(result)
            
        return result
    
    return composed


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the DataFrame with duplicate rows removed.

    Args:
        data: Input DataFrame.

    Returns:
        A DataFrame with duplicate rows removed.
    """
    return data.copy().drop_duplicates()


def strip_whitespace(data: pd.DataFrame) -> pd.DataFrame:
    """Strip leading and trailing whitespace from string columns.

    Args:
        data: Input DataFrame.

    Returns:
        A copy of the DataFrame with whitespace stripped from string columns.
    """
    result = data.copy()
    
    for column in result.columns:
        if result[column].dtype == "object" or is_string_dtype(result[column]):
            result[column] = result[column].str.strip()
    
    return result


def lowercase_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Convert all column names to lowercase strings.

    Args:
        data: Input DataFrame.

    Returns:
        A copy of the DataFrame with lowercase column names.
    """
    
    result = data.copy()
    result.columns = [str(column).lower() for column in result.columns]
    
    return result


def replace_spaces_in_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Replace spaces in column names with underscores.

    Args:
        data: Input DataFrame.

    Returns:
        A copy of the DataFrame with spaces replaced in column names.
    """
    
    result = data.copy()
    result.columns = [str(column).replace(" ", "_") for column in result.columns]
    
    return result


def drop_columns(*columns: str) -> DataFrameFunction:
    """Creatse a function that drops selected columns if they exist.

    Args:
        *columns: Column names to drop.

    Returns:
        A DataFrame function that drops the selected columns.
    """
    
    def dropper(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        existing_columns = [column for column in columns if column in result.columns]
        return result.drop(columns=existing_columns)
    
    return dropper
