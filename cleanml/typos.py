from __future__ import annotations

import pandas as pd

from cleanml.base import BaseTransformer

def edit_distance(first: str, second: str) -> int:
    """Calculate the Levenshtein edit distance between two strings.

    Args:
        first: First string to compare.
        second: Second string to compare.

    Returns:
        Minimum number of insertions, deletions, and substitutions needed to
        change one string into the other.
    """
    rows = len(first) + 1
    columns = len(second) + 1
    
    dp = [[0 for _ in range(columns)] for _ in range(rows)]
    
    for i in range(rows):
        dp[i][0] = i
        
    for j in range(columns):
        dp[0][j] = j
    
    for i in range(1, rows):
        for j in range(1, columns):
            if first[i - 1] == second[j - 1]:
                cost = 0
            else:
                cost = 1
    
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    
    return dp[-1][-1]

class CategoryTypoFixer(BaseTransformer):
    """Replace near-matching category values with valid category names.

    Args:
        column: Name of the categorical column to fix.
        valid_categories: Accepted category names.
        max_distance: Maximum edit distance allowed for replacement. If None,
            every value is replaced by its closest valid category.
    """
    
    def __init__(self, column: str, valid_categories: list[str], max_distance: int | None = 2):
        super().__init__()
        self._column = column
        self._valid_categories = valid_categories
        self._max_distance = max_distance
        self._replacements: dict[object, object] = {}
    
    def fit(self, data: pd.DataFrame, target : pd.Series | None = None) -> CategoryTypoFixer:
        """Learn replacements for observed category values.

        Args:
            data: DataFrame containing the category column.
            target: Ignored optional target values.

        Returns:
            The fitted typo fixer.

        Raises:
            TypeError: If data is not a pandas DataFrame.
            ValueError: If the selected column is missing or no valid
                categories are provided.
        """
        
        self._validate_dataframe(data)
        self._validate_columns(data, [self._column])
        
        if len(self._valid_categories) == 0:
            raise ValueError("valid_categories must contain at least one category.")
        
        
        unique_values = data[self._column].dropna().unique()
        
        for value in unique_values:
            closest_category = self._find_closest_category(str(value))
            distance = edit_distance(str(value), str(closest_category))
        
            if self._max_distance is None or distance <= self._max_distance:
                self._replacements[value] = closest_category
            else:
                self._replacements[value] = value
    
        self._mark_as_fitted()
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the data with known category typos fixed.

        Args:
            data: DataFrame to transform.

        Returns:
            A transformed copy of the input DataFrame.

        Raises:
            RuntimeError: If the typo fixer has not been fitted.
            TypeError: If data is not a pandas DataFrame.
            ValueError: If the selected column is missing.
        """
        
        self._check_is_fitted()
        self._validate_dataframe(data)
        self._validate_columns(data, [self._column])
        
        transformed_data = data.copy()
        
        transformed_data[self._column] = transformed_data[self._column].map(self._fix_value)

        return transformed_data
    
    def _find_closest_category(self, value: str) -> str:
        
        closest_category = self._valid_categories[0]
        smallest_distance = edit_distance(value, closest_category)
        
        for category in self._valid_categories[1:]:
            distance = edit_distance(value, category)
        
            if distance < smallest_distance:
                smallest_distance = distance
                closest_category = category
        
        return closest_category

    def _fix_value(self, value: object) -> object:
        
        if pd.isna(value):
            return value
        
        return self._replacements.get(value, value)
