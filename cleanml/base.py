from __future__ import annotations
from abc import ABC, abstractmethod

import pandas as pd

class BaseTransformer(ABC):
    """Abstract base class for all DataFrame transformers.

    Subclasses implement ``fit`` and ``transform`` and can use the helper
    validation methods provided here.
    """
    def __init__(self):
        self._is_fitted = False
    
    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame ,
        target: pd.Series | None = None
    )-> BaseTransformer:
        """Learn transformation parameters from a DataFrame.

        Args:
            data: DataFrame used to fit the transformer.
            target: Optional target values for supervised transformers.

        Returns:
            The fitted transformer instance.
        """
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a transformed copy of a DataFrame.

        Args:
            data: DataFrame to transform.

        Returns:
            A transformed copy of the input DataFrame.
        """
        pass
    
    def fit_transform(
        self,
        data: pd.DataFrame,
        target: pd.Series | None = None
    )-> pd.DataFrame:
        """Fit the transformer and return transformed data.

        Args:
            data: DataFrame used for fitting and transforming.
            target: Optional target values for supervised transformers.

        Returns:
            A transformed copy of the input DataFrame.
        """
        self.fit(data, target)
        return self.transform(data)

    def _mark_as_fitted(self) -> None:
        """Mark the transformer as fitted."""

        self._is_fitted = True
        
    def _check_is_fitted(self) -> None:
        """Raise an error if the transformer has not been fitted.

        Raises:
            RuntimeError: If the transformer is not fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before transforming")

    def _validate_dataframe(self, data: pd.DataFrame) -> None:
        """Check that input data is a pandas DataFrame.

        Raises:
            TypeError: If data is not a pandas DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame.")
    
    def _validate_columns(self, data: pd.DataFrame, columns: list[str]) -> None:
        """Check that all selected columns exist in the DataFrame.

        Raises:
            ValueError: If any selected columns are missing.
        """
        
        invalid_columns = [col for col in columns if col not in data.columns]
        
        if invalid_columns:
            raise ValueError(f"Columns not found: {invalid_columns}")
        
    def _get_columns(self, data: pd.DataFrame) -> list[str]:
        if self._columns is None:
            return list(data.columns)
            
        return self._columns
