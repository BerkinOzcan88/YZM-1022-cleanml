# LabelEncoder
# OneHotEncoder

from __future__ import annotations

import pandas as pd

from cleanml.base import BaseTransformer

class OneHotEncoder(BaseTransformer):
    """Convert categorical columns into one-hot encoded indicator columns.

    Args:
        columns: Categorical columns to encode.
        drop_original: Whether to remove original categorical columns.
    """

    def __init__(self, columns: list[str], drop_original: bool=True) -> None:
        super().__init__()
        self._columns = columns
        self._drop_original = drop_original
        self._categories = {}
        
    def fit(self, data: pd.DataFrame, target : pd.Series | None = None) -> OneHotEncoder:
        """Learn sorted category values for each selected column.

        Args:
            data: DataFrame used to learn categories.
            target: Ignored optional target values.

        Returns:
            The fitted encoder.

        Raises:
            TypeError: If data is not a pandas DataFrame.
            ValueError: If selected columns are missing.
        """
        
        self._validate_dataframe(data)
        
        columns_to_use = self._get_columns(data)
        self._validate_columns(data, columns_to_use)
        
        for column in columns_to_use:  
            categories = data[column].dropna().unique()
            self._categories[column] = sorted(categories)
        
        self._mark_as_fitted()
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the data with one-hot encoded columns.

        Unknown categories receive zeroes for all learned category columns.

        Args:
            data: DataFrame to transform.

        Returns:
            A transformed copy of the input DataFrame.

        Raises:
            RuntimeError: If the encoder has not been fitted.
            TypeError: If data is not a pandas DataFrame.
            ValueError: If selected columns are missing.
        """
        
        self._check_is_fitted()
        self._validate_dataframe(data)
        self._validate_columns(data, self._columns)
        
        transformed_data = data.copy()
        
        for column in self._columns:
            for category in self._categories[column]:
                new_column_name = f"{column}_{category}"
                transformed_data[new_column_name] = (transformed_data[column] == category).astype(int)
                
            if self._drop_original:
                transformed_data = transformed_data.drop(columns=[column])
        
        return transformed_data

class LabelEncoder(BaseTransformer):
    """Replace categorical values with integer labels.

    Args:
        columns: Categorical columns to encode.
    """
    
    def __init__(self, columns: list[str]) -> None:
        super().__init__()
        self._columns = columns
        self._mappings: dict = {}
        
    def fit(self, data: pd.DataFrame, target: pd.Series | None = None) -> LabelEncoder:
        """Learn category-to-integer mappings for selected columns.

        Args:
            data: DataFrame used to learn mappings.
            target: Ignored optional target values.

        Returns:
            The fitted encoder.

        Raises:
            TypeError: If data is not a pandas DataFrame.
            ValueError: If selected columns are missing.
        """
        
        self._validate_dataframe(data)
        self._validate_columns(data, self._columns)
        
        for column in self._columns:
            categories = data[column].dropna().unique()
            self._mappings[column] = {category: i for i, category in enumerate(sorted(categories))}
            
        self._mark_as_fitted()
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the data with categories replaced by labels.

        Unknown categories become missing values.

        Args:
            data: DataFrame to transform.

        Returns:
            A transformed copy of the input DataFrame.

        Raises:
            RuntimeError: If the encoder has not been fitted.
            TypeError: If data is not a pandas DataFrame.
            ValueError: If selected columns are missing.
        """
        
        self._check_is_fitted()
        self._validate_dataframe(data)
        self._validate_columns(data, self._columns)
        
        transformed_data = data.copy()
        
        for column in self._columns:
            transformed_data[column] = transformed_data[column].map(self._mappings[column])
        
        return transformed_data
