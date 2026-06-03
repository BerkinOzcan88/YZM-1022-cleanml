from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseLoader(ABC):
    def __init__(self, file_path: str) -> None:
        self._file_path = Path(file_path)
        
    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass
    
    def _check_file_exists(self) -> None:
        if not self._file_path.exists():
            raise FileNotFoundError(f"File not found: {self._file_path}")


class CSVLoader(BaseLoader):
    def load(self) -> pd.DataFrame:
        self._check_file_exists()
        return pd.read_csv(self._file_path)


class JSONLoader(BaseLoader):
    def load(self) -> pd.DataFrame:
        self._check_file_exists()
        return pd.read_json(self._file_path)


class DataLoaderFactory:
    @staticmethod
    def create(file_type: str, file_path: str) -> BaseLoader:
        file_type = file_type.lower().strip()
        
        if file_type == "csv":
            return CSVLoader(file_path)
        
        if file_type == "json":
            return JSONLoader(file_path)
        
        raise ValueError(
            f"Unsupported file type: {file_type}. Supported types are: csv, json."
        )
        
    @staticmethod
    def from_file_path(file_path: str) -> BaseLoader:
        suffix = Path(file_path).suffix.lower()
        
        if suffix == ".csv":
            return CSVLoader(file_path)
        
        if suffix == ".json":
            return JSONLoader(file_path)
        
        raise ValueError(
            f"Unsupported file extension: {suffix}. Supported extensions are: .csv, .json."
        )