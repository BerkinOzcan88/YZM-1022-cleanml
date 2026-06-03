from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pandas as pd

from cleanml.base import BaseTransformer


class ParallelColumnTransformer(BaseTransformer):
    
    def __init__(
        self, transformers: list[BaseTransformer], max_workers: int | None = None):
        super().__init__()
        self._transformers = transformers
        self._max_workers = max_workers
        
        self._validate_transformers()
    
    def fit(
        self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> ParallelColumnTransformer:
        
        self._validate_dataframe(data)
        
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [
                executor.submit(transformer.fit, data.copy(), target)
                for transformer in self._transformers
            ]
            
            for future in futures:
                future.result()
        
        self._mark_as_fitted()
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        
        self._check_is_fitted()
        self._validate_dataframe(data)
        
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [
                executor.submit(transformer.transform, data.copy())
                for transformer in self._transformers
            ]
            
            transformed_results = [future.result() for future in futures]
        
        return self._merge_results(data, transformed_results)
    
    def fit_transform(
        self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        
        self._validate_dataframe(data)
        
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [
                executor.submit(transformer.fit_transform, data.copy(), target)
                for transformer in self._transformers
            ]
            
            transformed_results = [future.result() for future in futures]
            
        self._mark_as_fitted()
        return self._merge_results(data, transformed_results)
    
    def _merge_results(
        self, original_data: pd.DataFrame, transformed_results: list[pd.DataFrame]) -> pd.DataFrame:
        
        merged_data = original_data.copy()
        original_columns = set(original_data.columns)
        
        for transformed_data in transformed_results:
            transformed_columns = set(transformed_data.columns)
            
            dropped_columns = original_columns - transformed_columns
            added_columns = transformed_columns - original_columns
            common_columns = original_columns & transformed_columns
            
            for column in common_columns:
                if not transformed_data[column].equals(original_data[column]):
                    merged_data[column] = transformed_data[column]
                    
            for column in added_columns:
                merged_data[column] = transformed_data[column]
            
            for column in dropped_columns:
                if column in merged_data.columns:
                    merged_data = merged_data.drop(columns=[column])
        
        return merged_data
    
    def _validate_transformers(self) -> None:
        
        if not isinstance(self._transformers, list):
            raise TypeError("transformers must be provided as a list.")
        
        if len(self._transformers) == 0:
            raise ValueError("ParallelColumnTransformer must contain at least one transformer.")
        
        for transformer in self._transformers:
            if not hasattr(transformer, "fit") or not callable(transformer.fit):
                raise TypeError(
                    f"{transformer} is not valid because it does not have a fit() method."
                )
                
            if not hasattr(transformer, "transform") or not callable(transformer.transform):
                raise TypeError(
                    f"{transformer} is not valid because it does not have a transform() method."
                )