# StandardScaler
# MinMaxScaler
from __future__ import annotations

import numpy as np
import pandas as pd 

from cleanml.base import BaseTransformer

class StandardScaler(BaseTransformer):
    
    def __init__(self, columns: list | None = None):
        super().__init__()
        self._columns = columns
        self._means = {}
        self._stds = {}
    
    def fit(self, data: pd.DataFrame, target : pd.Series | None = None) -> BaseTransformer:
        
        self._validate_dataframe(data)
        
        columns_to_use = self._get_columns(data)
        self._validate_columns(data, columns_to_use)
        
        for column in columns_to_use:
            self._means[column] = data[column].mean()
            self._stds[column] = data[column].std()
        
        self._mark_as_fitted()
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        
        self._check_if_fitted()
        self._validate_dataframe(data)
        
        self._validate_columns(data, self._columns)
        
        transformed_data = data.copy()
        
        for column in self._columns:
            std = self._stds[column]
            mean = self._means[column]
            
            if std == 0:
                transformed_data[column] = 0
            else:
                transformed_data[column] = (transformed_data[column] - self._means[column]) / std
        return transformed_data

class MinMaxScaler(BaseTransformer):
    def __init__(self, columns: list | None = None):
        super().__init__()
        self._columns = columns
        self._maxs = {}
        self._mins = {}
        
    def fit(self, data: pd.DataFrame, target : pd.Series | None = None) -> BaseTransformer:
        
        self._validate_dataframe(data)
        
        columns_to_use = self._get_columns(data)
        self._validate_columns(data, columns_to_use)
        
        for column in columns_to_use:
            self._maxs[column] = data[column].max()
            self._mins[column] = data[column].min()
        
        self._mark_as_fitted()
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        
        self._check_if_fitted()
        self._validate_dataframe(data)
        
        self._validate_columns(data, self._columns)
        
        transformed_data = data.copy()
        
        for column in self._columns:
            max = self._maxs[column]
            min = self._mins[column]
            
            if (max - min) == 0:
                transformed_data[column] = 0
            else:
                transformed_data[column] = (transformed_data[column] -min) / (max - min)
        return transformed_data
