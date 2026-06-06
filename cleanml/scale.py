# StandardScaler
# MinMaxScaler
from __future__ import annotations

import pandas as pd 

from cleanml.base import BaseTransformer

class StandardScaler(BaseTransformer):
    """Scale numeric columns to zero mean and unit standard deviation.

    Args:
        columns: Columns to scale. If None, all columns are used.
    """
    
    def __init__(self, columns: list | None = None):
        super().__init__()
        self._columns = columns
        self._means = {}
        self._stds = {}
        self._fitted_columns = []
    
    def fit(self, data: pd.DataFrame, target : pd.Series | None = None) -> BaseTransformer:
        """Learn column means and standard deviations.

        Args:
            data: DataFrame used to calculate scaling statistics.
            target: Ignored optional target values.

        Returns:
            The fitted scaler.

        Raises:
            TypeError: If data is not a pandas DataFrame.
            ValueError: If selected columns are missing.
        """
        
        self._validate_dataframe(data)
        
        columns_to_use = self._get_columns(data)
        self._validate_columns(data, columns_to_use)
        self._validate_numeric_columns(data, columns_to_use)
        self._fitted_columns = columns_to_use
        
        for column in columns_to_use:
            self._means[column] = data[column].mean()
            self._stds[column] = data[column].std()
        
        self._mark_as_fitted()
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the data with selected columns standardized.

        Args:
            data: DataFrame to transform.

        Returns:
            A transformed copy of the input DataFrame.

        Raises:
            RuntimeError: If the scaler has not been fitted.
            TypeError: If data is not a pandas DataFrame.
            ValueError: If selected columns are missing.
        """
        
        self._check_is_fitted()
        self._validate_dataframe(data)
        
        self._validate_columns(data, self._fitted_columns)
        self._validate_numeric_columns(data, self._fitted_columns)
        
        transformed_data = data.copy()
        
        for column in self._fitted_columns:
            std = self._stds[column]
            
            if std == 0:
                transformed_data[column] = 0
            else:
                transformed_data[column] = (transformed_data[column] - self._means[column]) / std
        return transformed_data

class MinMaxScaler(BaseTransformer):
    """Scale numeric columns into the range 0 to 1.

    Args:
        columns: Columns to scale. If None, all columns are used.
    """

    def __init__(self, columns: list | None = None):
        super().__init__()
        self._columns = columns
        self._maxs = {}
        self._mins = {}
        self._fitted_columns = []
        
    def fit(self, data: pd.DataFrame, target : pd.Series | None = None) -> BaseTransformer:
        """Learn column minimum and maximum values.

        Args:
            data: DataFrame used to calculate scaling statistics.
            target: Ignored optional target values.

        Returns:
            The fitted scaler.

        Raises:
            TypeError: If data is not a pandas DataFrame.
            ValueError: If selected columns are missing.
        """
        
        self._validate_dataframe(data)
        
        columns_to_use = self._get_columns(data)
        self._validate_columns(data, columns_to_use)
        self._validate_numeric_columns(data, columns_to_use)
        self._fitted_columns = columns_to_use
        
        for column in columns_to_use:
            self._maxs[column] = data[column].max()
            self._mins[column] = data[column].min()
        
        self._mark_as_fitted()
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the data with selected columns min-max scaled.

        Args:
            data: DataFrame to transform.

        Returns:
            A transformed copy of the input DataFrame.

        Raises:
            RuntimeError: If the scaler has not been fitted.
            TypeError: If data is not a pandas DataFrame.
            ValueError: If selected columns are missing.
        """
        
        self._check_is_fitted()
        self._validate_dataframe(data)
        
        self._validate_columns(data, self._fitted_columns)
        self._validate_numeric_columns(data, self._fitted_columns)
        
        transformed_data = data.copy()
        
        for column in self._fitted_columns:
            max_value = self._maxs[column]
            min_value = self._mins[column]
            
            if (max_value - min_value) == 0:
                transformed_data[column] = 0
            else:
                transformed_data[column] = (transformed_data[column] - min_value) / (max_value - min_value)
        return transformed_data
