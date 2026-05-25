# LabelEncoder
# OneHotEncoder

from __future__ import annotations

import pandas as pd

from cleanml.base import BaseTransformer

class OneHotEncoder(BaseTransformer):
    def __init__(self, columns: list, drop_original: bool=True):
        super().__init__()
        self._columns = columns
        self._drop_original = drop_original
        self._categories = {}
        
    def fit(self, data: pd.DataFrame, target : pd.Series | None = None) -> OneHotEncoder:
        
        self._validate_dataframe(data)
        
        columns_to_use = self._get_columns(data)
        self._validate_columns(data, columns_to_use)
        
        for column in columns_to_use:  
            categories = data[column].dropna().unique()
            self._categories[column] = sorted(categories)
        
        self._mark_as_fitted()
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        
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
    
    def __init__(self, columns: list):
        super().__init__()
        self._columns = columns
        self._mappings: dict = {}
        
    def fit(self, data: pd.DataFrame, target: pd.Series | None = None) -> LabelEncoder:
        
        self._validate_dataframe(data)
        self._validate_columns(data, self._columns)
        
        for column in self._columns:
            categories = data[column].dropna().unique()
            self._mappings[column] = {category: i for i, category in enumerate(sorted(categories))}
            
        self._mark_as_fitted()
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        
        self._check_is_fitted()
        self._validate_dataframe(data)
        self._validate_columns(data, self._columns)
        
        transformed_data = data.copy()
        
        for column in self._columns:
            transformed_data[column] = transformed_data[column].map(self._mappings[column])
        
        return transformed_data
