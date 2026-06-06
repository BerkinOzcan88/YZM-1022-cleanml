from __future__ import annotations

import pandas as pd
from cleanml.base import BaseTransformer


class Pipeline(BaseTransformer):
    """Run multiple transformers sequentially on a DataFrame.

    Args:
        steps: Ordered list of transformers to fit and apply.
    """
    
    def __init__(self, steps : list[BaseTransformer]):
        super().__init__()
        self.steps = steps
        self._observers : list = []
        
        self._validate_steps()
    
    def fit(
        self,
        data: pd.DataFrame,
        target: pd.Series | None = None
    ) -> "Pipeline":
        """Fit each pipeline step in order.

        Each step is fitted on the output produced by the previous step.

        Args:
            data: DataFrame used to fit the pipeline.
            target: Optional target values passed to each step.

        Returns:
            The fitted pipeline.

        Raises:
            TypeError: If data is not a pandas DataFrame.
            Exception: Re-raises any exception from a pipeline step.
        """
        self._validate_dataframe(data)
        
        current_data = data.copy()
        
        self._notify("pipeline_started", {"pipeline": self})
        
        try:
            for step in self.steps:
                step_name = step.__class__.__name__
                
                self._notify("step_started", {
                    "step": step,
                    "step_name": step_name
                })
                
                step.fit(current_data, target)
                current_data = step.transform(current_data)
                
                self._notify("step_finished", {
                    "step": step,
                    "step_name": step_name
                })
                
            self._mark_as_fitted()
            
            self._notify("pipeline_finished", {"pipeline": self})
            
            return self
        
        except Exception as error:
            self._notify("error_occurred", {
                "pipeline": self,
                "error": error
            })
            raise
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply each fitted pipeline step in order.

        Args:
            data: DataFrame to transform.

        Returns:
            A transformed copy of the input DataFrame.

        Raises:
            RuntimeError: If the pipeline has not been fitted.
            TypeError: If data is not a pandas DataFrame.
            Exception: Re-raises any exception from a pipeline step.
        """
        self._check_is_fitted()
        self._validate_dataframe(data)
        
        current_data = data.copy()
        
        self._notify("pipeline_started", {"pipeline": self})
        
        try:
            for step in self.steps:
                step_name = step.__class__.__name__
                
                self._notify("step_started", {
                    "step": step,
                    "step_name": step_name
                })
                
                current_data = step.transform(current_data)
                
                self._notify("step_finished", {
                    "step": step,
                    "step_name": step_name
                })
                
            self._notify("pipeline_finished", {"pipeline": self})
            
            return current_data
        
        except Exception as error:
            self._notify("error_occurred", {
                "pipeline": self,
                "error": error
            })
            raise
    
    def fit_transform(self,
        data : pd.DataFrame,
        target : pd.Series | None = None
    )-> pd.DataFrame:
        """Fit each step and return the final transformed DataFrame.

        Args:
            data: DataFrame used for fitting and transforming.
            target: Optional target values passed to each step.

        Returns:
            A transformed copy of the input DataFrame.

        Raises:
            TypeError: If data is not a pandas DataFrame.
            Exception: Re-raises any exception from a pipeline step.
        """
        self._validate_dataframe(data)
        
        current_data = data.copy()
        
        self._notify("pipeline_started", {"pipeline": self})
        
        try:
            for step in self.steps:
                step_name = step.__class__.__name__
                
                self._notify("step_started", {
                "step": step,
                "step_name": step_name
                })
            
                current_data = step.fit_transform(current_data, target)
            
                self._notify("step_finished", {
                    "step": step,
                    "step_name": step_name
                })
            
            self._mark_as_fitted()
            
            self._notify("pipeline_finished", {"pipeline": self})
            
            return current_data
        
        except Exception as error:
            self._notify("error_occurred", {
            "pipeline": self,
            "error": error
        })
            raise
        
    def _validate_steps(self) -> None:
        if not isinstance(self.steps, list):
            raise TypeError("Pipeline steps must be a list.")
        
        if len(self.steps) == 0:
            raise ValueError("Pipeline must contain at least one step.")
        
        if not all(isinstance(step, BaseTransformer) for step in self.steps):
            raise TypeError("Every step must be a transformer.")
    
    def add_observer(self, observer: object) -> None:
        """Register an observer to receive pipeline events.

        Args:
            observer: Object with an ``on_event`` method.
        """
        self._observers.append(observer)
    
    def remove_observer(self, observer: object) -> None:
        """Remove a previously registered observer.

        Args:
            observer: Observer object to remove.
        """
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _notify(self, event_type: str, data: dict) -> None:
        for observer in self._observers:
            if hasattr(observer, "on_event") and callable(observer.on_event):
                observer.on_event(event_type, data)
    
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
