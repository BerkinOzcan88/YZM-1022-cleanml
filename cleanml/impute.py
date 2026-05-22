# ImputationStrategy X
# MeanStrategy X
# ModeStrategy X
# ConstantStrategy X
# MissingValueImputer

from cleanml.base import BaseTransformer 

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd 

class ImputationStrategy(ABC):
    @abstractmethod
    def calculate(self,column: pd.Series) -> float:
        pass

class MeanStrategy(ImputationStrategy):
    def calculate(self,column: pd.Series) -> float:
        return column.mean()

class MedianStrategy(ImputationStrategy):
    def calculate(self,column: pd.Series) -> float:
        return column.median()

class ModeStrategy(ImputationStrategy):
    def calculate(self,column: pd.Series) -> float:
        return column.mode()

class ModeStrategy(ImputationStrategy):
    def __init__(self, value):
        self.value = value
    
    def calculate(self, column: pd.Series) -> float:
        return self.value

class MissingValueImputer(BaseTransformer):
    def __init__(self, strategy: ImputationStrategy, columns=None):
        super().__init__()
        self._strategy = strategy
        self._columns = columns
        self._fill_values: dict = {}
    
    def fit(self, data: pd.DataFrame, target : pd.Series | None):
        
        self._validate_dataframe(data)
        
        columns_to_use = self._get_columns(data)
        self._validate_columns(data, columns_to_use)
        
        for column in columns_to_use:  
            fill_value = self.strategy.calculate(data[column])
            self.fill_values[column] = fill_value
        
        self._mark_as_fitted()
        return self
        
    def transform(self, data: pd.DataFrame):
        
        self._check_is_fitted()
        self._validate_dataframe(data)
        
        columns_to_use = self._get_columns(data)
        self._validate_columns(data, columns_to_use)
        
        transformed_data = data.copy()
        
        for column in columns_to_use:
            transformed_data[column] = transformed_data[column].fillna(
                self.fill_values[column]
            )
        
        return transformed_data
    
    def _get_columns(self, data: pd.DataFrame) -> list[str]:
        if self._columns is None:
            return list(data.columns)
        
        return self._columns