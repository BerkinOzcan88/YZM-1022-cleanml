from __future__ import annotations

from cleanml.base import BaseTransformer 


from abc import ABC, abstractmethod

import pandas as pd 

class ImputationStrategy(ABC):
    """Base class for missing-value imputation strategies."""

    @abstractmethod
    def calculate(self,column: pd.Series) -> float:
        """Calculate the replacement value for a column.

        Args:
            column: Column used to calculate the fill value.

        Returns:
            A scalar value used to replace missing values.
        """
        pass

class MeanStrategy(ImputationStrategy):
    """Use the column mean as the fill value."""

    def calculate(self,column: pd.Series) -> float:
        """Return the mean value of a column."""
        return column.mean()

class MedianStrategy(ImputationStrategy):
    """Use the column median as the fill value."""

    def calculate(self,column: pd.Series) -> float:
        """Return the median value of a column."""
        return column.median()

class ModeStrategy(ImputationStrategy):
    """Use the first column mode as the fill value."""

    def calculate(self,column: pd.Series) -> float:
        """Return the first mode value of a column.

        Returns:
            The first mode value, or ``pd.NA`` if no mode exists.
        """
        modes = column.mode()
        if modes.empty:
            return pd.NA
        return modes.iloc[0]

class ConstantStrategy(ImputationStrategy):
    """Use a fixed constant as the fill value.

    Args:
        value: Value used to replace missing values.
    """

    def __init__(self, value):
        self.value = value
    
    def calculate(self, column: pd.Series) -> float:
        """Return the configured constant value."""
        return self.value

class MissingValueImputer(BaseTransformer):
    """Fill missing values in selected DataFrame columns.

    Args:
        strategy: Strategy used to calculate replacement values.
        columns: Columns to impute. If None, all columns are used.
    """

    def __init__(self, strategy: ImputationStrategy, columns: list=None):
        super().__init__()
        self._strategy = strategy
        self._columns = columns
        self._fill_values: dict = {}
    
    def fit(self, data: pd.DataFrame, target : pd.Series | None = None):
        """Learn fill values from the input DataFrame.

        Args:
            data: DataFrame used to calculate fill values.
            target: Ignored optional target values.

        Returns:
            The fitted imputer.

        Raises:
            TypeError: If data is not a pandas DataFrame.
            ValueError: If selected columns are missing.
        """
        
        self._validate_dataframe(data)
        
        columns_to_use = self._get_columns(data)
        self._validate_columns(data, columns_to_use)
        
        for column in columns_to_use:  
            fill_value = self._strategy.calculate(data[column])
            self._fill_values[column] = fill_value
        
        self._mark_as_fitted()
        return self
        
    def transform(self, data: pd.DataFrame):
        """Return a copy of the data with missing values filled.

        Args:
            data: DataFrame to transform.

        Returns:
            A transformed copy of the input DataFrame.

        Raises:
            RuntimeError: If the imputer has not been fitted.
            TypeError: If data is not a pandas DataFrame.
            ValueError: If selected columns are missing.
        """
        
        self._check_is_fitted()
        self._validate_dataframe(data)
        
        columns_to_use = self._get_columns(data)
        self._validate_columns(data, columns_to_use)
        
        transformed_data = data.copy()
        
        for column in columns_to_use:
            transformed_data[column] = transformed_data[column].fillna(
                self._fill_values[column]
            )
        
        return transformed_data
