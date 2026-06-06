import pandas as pd

from cleanml.base import BaseTransformer


class ColumnReportingTransformer(BaseTransformer):
    def fit(self, data: pd.DataFrame, target: pd.Series | None = None):
        self.selected_columns = self._get_columns(data)
        self._mark_as_fitted()
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        return data.copy()


def test_base_transformer_get_columns_defaults_to_all_columns():
    data = pd.DataFrame({"age": [10], "city": ["Ankara"]})
    transformer = ColumnReportingTransformer()

    transformer.fit(data)

    assert transformer.selected_columns == ["age", "city"]
