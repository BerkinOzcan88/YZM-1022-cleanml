from __future__ import annotations

import pandas as pd
from cleanml.base import BaseTransformer


class Pipeline(BaseTransformer):
    
    def __init__(self, steps : list[BaseTransformer]):
        super().__init__()
        self.steps = steps
        
        self._validate_steps()
    
    def fit(
        self,
        data : pd.DataFrame,
        target : pd.Series | None
    )-> Pipeline:
        """
        Fits each step in order.
        
        Each step is fitted on top od the last step.
        """
        self._validate_dataframe(data)
        
        current_data = data.copy()
        
        for step in self.steps:
            step.fit(current_data, target)
            current_data = step.transform(current_data)
            
        self._mark_as_fitted()
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        step._check_if_fitted()
        self._validate_dataframe(data)
        
        current_data = data.copy()
        
        for step in self.steps:
            current_data = step.transform(current_data)
            
        return current_data
    
    def fit_transform(self,
        data : pd.DataFrame,
        target : pd.Series | None
    )-> pd.DataFrame:
        self._validate_dataframe(data)
        
        current_data = data.copy()
        
        for step in self.steps:
            step.fit_transform(current_data, target)
        self._mark_as_fitted()
        
        return current_data
    
    def _validate_steps(self) -> None:
        if not isinstance(self.steps, list):
            raise TypeError("Pipeline steps must be a list.")
        
        if len(self.steps) == 0:
            raise ValueError("Pipeline must contain at least one step.")
        
        if not all(isinstance(step, BaseTransformer) for step in self.steps):
            raise TypeError("Every step must be a transformer.")
    
    def get_step_names(self) -> list[str]:
        """
        Return the class names of all steps in the pipeline.
        """
        return [step.__class__.__name__ for step in self.steps]
    
    def __len__(self) -> int:
        """
        Return the number of steps in the pipeline.
        """
        return len(self.steps)
    
    def __repr__(self) -> str:
        """
        Return a representation of the pipeline.
        """
        step_names = " -> ".join(self.get_step_names())
        return f"Pipeline({step_names})"