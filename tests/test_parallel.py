import pandas as pd
import pytest

from cleanml import (
    ConstantStrategy,
    MeanStrategy,
    MissingValueImputer,
    OneHotEncoder,
    ParallelColumnTransformer,
)


def test_parallel_column_transformer_merges_independent_column_changes():
    data = pd.DataFrame({"age": [10.0, None], "city": ["Ankara", None]})
    original = data.copy(deep=True)
    transformer = ParallelColumnTransformer([
        MissingValueImputer(MeanStrategy(), columns=["age"]),
        MissingValueImputer(ConstantStrategy("Unknown"), columns=["city"]),
    ])

    result = transformer.fit_transform(data)

    assert result.to_dict("list") == {
        "age": [10.0, 10.0],
        "city": ["Ankara", "Unknown"],
    }
    pd.testing.assert_frame_equal(data, original)


def test_parallel_column_transformer_merges_added_and_dropped_columns():
    data = pd.DataFrame({"city": ["Ankara", "Istanbul"], "age": [10, 20]})
    transformer = ParallelColumnTransformer([
        OneHotEncoder(columns=["city"]),
    ])

    result = transformer.fit_transform(data)

    assert result.to_dict("list") == {
        "age": [10, 20],
        "city_Ankara": [1, 0],
        "city_Istanbul": [0, 1],
    }


def test_parallel_column_transformer_rejects_empty_transformer_list():
    with pytest.raises(ValueError, match="at least one transformer"):
        ParallelColumnTransformer([])


def test_parallel_column_transformer_rejects_invalid_transformer():
    with pytest.raises(TypeError, match="fit"):
        ParallelColumnTransformer([object()])


def test_parallel_column_transformer_transform_before_fit_raises_runtime_error():
    transformer = ParallelColumnTransformer([
        MissingValueImputer(MeanStrategy(), columns=["age"]),
    ])

    with pytest.raises(RuntimeError, match="must be fitted"):
        transformer.transform(pd.DataFrame({"age": [10.0]}))
