from __future__ import annotations

from collections.abc import Callable

import pandas as pd


DataFrameFunction = Callable[[pd.DataFrame], pd.DataFrame]


def compose(*functions: DataFrameFunction) -> DataFrameFunction:
    """
    Compose multiple DataFrame transformation functions into one function.

    The functions are applied from left to right.

    Example:
        clean = compose(remove_duplicates, strip_whitespace, lowercase_column_names)
        result = clean(df)
    """

    def composed(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()

        for function in functions:
            result = function(result)

        return result

    return composed


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the DataFrame with duplicate rows removed.
    """
    return data.copy().drop_duplicates()


def strip_whitespace(data: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading and trailing whitespace from all string values.
    """
    result = data.copy()

    for column in result.columns:
        if result[column].dtype == "object":
            result[column] = result[column].str.strip()

    return result


def lowercase_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the DataFrame with lowercase column names.
    """
    result = data.copy()
    result.columns = [str(column).lower() for column in result.columns]
    return result


def replace_spaces_in_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the DataFrame with spaces in column names replaced by underscores.
    """
    result = data.copy()
    result.columns = [str(column).replace(" ", "_") for column in result.columns]
    return result


def drop_columns(*columns: str) -> DataFrameFunction:
    """
    Return a function that drops selected columns.

    This is a higher-order function because it returns another function.
    """

    def dropper(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        existing_columns = [column for column in columns if column in result.columns]
        return result.drop(columns=existing_columns)

    return dropper