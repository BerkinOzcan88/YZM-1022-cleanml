import pandas as pd
import pytest

from cleanml import (
    ConstantStrategy,
    MeanStrategy,
    MedianStrategy,
    MissingValueImputer,
    ModeStrategy,
)


def test_mean_imputer_fills_missing_numeric_values_without_mutating_input():
    data = pd.DataFrame({"age": [10.0, None, 20.0], "name": ["Ada", "Bob", "Cy"]})
    original = data.copy(deep=True)

    result = MissingValueImputer(MeanStrategy(), columns=["age"]).fit_transform(data)

    assert result is not data
    assert result["age"].tolist() == [10.0, 15.0, 20.0]
    pd.testing.assert_frame_equal(data, original)


def test_median_imputer_fills_missing_numeric_values():
    data = pd.DataFrame({"score": [1.0, None, 100.0]})

    result = MissingValueImputer(MedianStrategy(), columns=["score"]).fit_transform(data)

    assert result["score"].tolist() == [1.0, 50.5, 100.0]


def test_mode_imputer_uses_first_mode_scalar():
    data = pd.DataFrame({"city": ["Ankara", None, "Istanbul", "Istanbul"]})

    result = MissingValueImputer(ModeStrategy(), columns=["city"]).fit_transform(data)

    assert result["city"].tolist() == ["Ankara", "Istanbul", "Istanbul", "Istanbul"]


def test_constant_imputer_fills_selected_columns_only():
    data = pd.DataFrame({"city": ["Ankara", None], "age": [20, None]})

    result = MissingValueImputer(ConstantStrategy("Unknown"), columns=["city"]).fit_transform(data)

    assert result["city"].tolist() == ["Ankara", "Unknown"]
    assert pd.isna(result.loc[1, "age"])


def test_imputer_columns_none_uses_all_columns():
    data = pd.DataFrame({"city": ["Ankara", None], "country": [None, "Turkey"]})

    result = MissingValueImputer(ConstantStrategy("Unknown")).fit_transform(data)

    assert result.to_dict("list") == {
        "city": ["Ankara", "Unknown"],
        "country": ["Unknown", "Turkey"],
    }


def test_imputer_rejects_non_dataframe_input():
    imputer = MissingValueImputer(MeanStrategy(), columns=["age"])

    with pytest.raises(TypeError, match="Data must be a pandas DataFrame"):
        imputer.fit([1, 2, 3])


def test_imputer_rejects_missing_columns():
    imputer = MissingValueImputer(MeanStrategy(), columns=["missing"])

    with pytest.raises(ValueError, match="Columns not found"):
        imputer.fit(pd.DataFrame({"age": [10, 20]}))


def test_imputer_transform_before_fit_raises_runtime_error():
    imputer = MissingValueImputer(MeanStrategy(), columns=["age"])

    with pytest.raises(RuntimeError, match="must be fitted"):
        imputer.transform(pd.DataFrame({"age": [10, None]}))
