from __future__ import annotations
from abc import ABC, abstractmethod

import pandas as pd

class BaseTransformer(ABC):
    """
    Abstract base class for all the transformers.
    
    Every transformer should inherit from this class and implment fit and transform methods.
    """
    def __init__(self):
        self._is_fitted = False
    
    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame ,
        target: pd.Series | None
    )-> BaseTransformer:
        """
        Learns the wanted information from the dataset.
        
        This method return itself.
        """
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the transformation to the dataset.
        
        This method returns a transformed copy of the dataset.
        """
        pass
    
    def fit_transform(
        self,
        data: pd.DataFrame,
        target: pd.Series | None
    )-> pd.DataFrame:
        """
        Fits the transformer and transforms the data.
        
        This method returns a transformed copy of the dataset.
        """
        self.fit(data, target)
        return self.transform(data)

    def _mark_as_fitted(self) -> None:
        """
        Marks the transformer as fitted.
        
        Child classes call this method at the and of their fit() method.
        """

        self._is_fitted = True
        
    def _check_if_fitted(self) -> None:
        """
        Raises an error if not fitted.
        
        Child classes use this method to check before the transform.
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before transforming")

    def _validate_dataframe(self, data: pd.DataFrame) -> None:
        """
        Checks that the input is a pandas DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
    
    def _validate_columss(self, data: pd.DataFrame, columns: list[str]) -> None:
        """
        Checks if all selected columns are valid.
        
        Raises a value error if there is columns that are not valid.
        """
        
        invalid_columns = [col for col in columns if col not in data.columns]
        
        if invalid_columns:
            raise ValueError(f"Columns not found: {invalid_columns}")