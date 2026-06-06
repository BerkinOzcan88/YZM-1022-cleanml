from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseLoader(ABC):
    """Base class for loading tabular data from a file.

    Args:
        file_path: Path to the file that should be loaded.
    """

    def __init__(self, file_path: str) -> None:
        self._file_path = Path(file_path)
        
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load data from the configured file path.

        Returns:
            Loaded data as a pandas DataFrame.
        """
        pass
    
    def _check_file_exists(self) -> None:
        if not self._file_path.exists():
            raise FileNotFoundError(f"File not found: {self._file_path}")


class CSVLoader(BaseLoader):
    """Load a CSV file into a pandas DataFrame."""

    def load(self) -> pd.DataFrame:
        """Read the configured CSV file.

        Returns:
            Loaded CSV data as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the configured file path does not exist.
        """
        self._check_file_exists()
        return pd.read_csv(self._file_path)


class JSONLoader(BaseLoader):
    """Load a JSON file into a pandas DataFrame."""

    def load(self) -> pd.DataFrame:
        """Read the configured JSON file.

        Returns:
            Loaded JSON data as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the configured file path does not exist.
        """
        self._check_file_exists()
        return pd.read_json(self._file_path)


class DataLoaderFactory:
    """Create data loaders from file types or file paths."""

    @staticmethod
    def create(file_type: str, file_path: str) -> BaseLoader:
        """Create a loader for an explicit file type.

        Args:
            file_type: File type name, such as ``csv`` or ``json``.
            file_path: Path to the file that should be loaded.

        Returns:
            A loader instance for the requested file type.

        Raises:
            ValueError: If the file type is not supported.
        """
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
        """Create a loader based on a file extension.

        Args:
            file_path: Path whose extension decides the loader type.

        Returns:
            A loader instance for the file extension.

        Raises:
            ValueError: If the extension is not supported.
        """
        suffix = Path(file_path).suffix.lower()
        
        if suffix == ".csv":
            return CSVLoader(file_path)
        
        if suffix == ".json":
            return JSONLoader(file_path)
        
        raise ValueError(
            f"Unsupported file extension: {suffix}. Supported extensions are: .csv, .json."
        )
