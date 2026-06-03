from __future__ import annotations

from collections.abc import Callable

import pandas as pd


DataFrameFunction = Callable[[pd.DataFrame], pd.DataFrame]


def compose(*functions: DataFrameFunction) -> DataFrameFunction:
    
    def composed(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        
        for function in functions:
            result = function(result)
            
        return result
    
    return composed


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    return data.copy().drop_duplicates()


def strip_whitespace(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    
    for column in result.columns:
        if result[column].dtype == "object":
            result[column] = result[column].str.strip()
    
    return result


def lowercase_column_names(data: pd.DataFrame) -> pd.DataFrame:
    
    result = data.copy()
    result.columns = [str(column).lower() for column in result.columns]
    
    return result


def replace_spaces_in_column_names(data: pd.DataFrame) -> pd.DataFrame:
    
    result = data.copy()
    result.columns = [str(column).replace(" ", "_") for column in result.columns]
    
    return result


def drop_columns(*columns: str) -> DataFrameFunction:
    
    def dropper(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        existing_columns = [column for column in columns if column in result.columns]
        return result.drop(columns=existing_columns)
    
    return dropper