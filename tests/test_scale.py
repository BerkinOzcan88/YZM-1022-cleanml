import numpy as np
import pandas as pd
import pytest

from cleanml import MinMaxScaler, StandardScaler


def test_standard_scaler_scales_explicit_columns_without_mutating_input():
    data = pd.DataFrame({"age": [1.0, 2.0, 3.0], "name": ["a", "b", "c"]})
    original = data.copy(deep=True)

    result = StandardScaler(columns=["age"]).fit_transform(data)

    assert result is not data
    assert np.allclose(result["age"], [-1.0, 0.0, 1.0])
    assert result["name"].tolist() == ["a", "b", "c"]
    pd.testing.assert_frame_equal(data, original)


def test_standard_scaler_columns_none_scales_all_columns_seen_during_fit():
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})

    result = StandardScaler().fit_transform(data)

    assert np.allclose(result["x"], [-1.0, 0.0, 1.0])
    assert np.allclose(result["y"], [-1.0, 0.0, 1.0])


def test_standard_scaler_constant_column_becomes_zero():
    data = pd.DataFrame({"x": [5.0, 5.0, 5.0]})

    result = StandardScaler(columns=["x"]).fit_transform(data)

    assert result["x"].tolist() == [0, 0, 0]


def test_minmax_scaler_scales_explicit_columns_without_mutating_input():
    data = pd.DataFrame({"age": [10.0, 20.0, 30.0], "name": ["a", "b", "c"]})
    original = data.copy(deep=True)

    result = MinMaxScaler(columns=["age"]).fit_transform(data)

    assert result is not data
    assert np.allclose(result["age"], [0.0, 0.5, 1.0])
    assert result["name"].tolist() == ["a", "b", "c"]
    pd.testing.assert_frame_equal(data, original)


def test_minmax_scaler_columns_none_scales_all_columns_seen_during_fit():
    data = pd.DataFrame({"x": [10.0, 20.0, 30.0], "y": [100.0, 150.0, 200.0]})

    result = MinMaxScaler().fit_transform(data)

    assert np.allclose(result["x"], [0.0, 0.5, 1.0])
    assert np.allclose(result["y"], [0.0, 0.5, 1.0])


def test_minmax_scaler_constant_column_becomes_zero():
    data = pd.DataFrame({"x": [5.0, 5.0, 5.0]})

    result = MinMaxScaler(columns=["x"]).fit_transform(data)

    assert result["x"].tolist() == [0, 0, 0]


@pytest.mark.parametrize("scaler_class", [StandardScaler, MinMaxScaler])
def test_scaler_rejects_non_numeric_explicit_columns(scaler_class):
    data = pd.DataFrame({"name": ["Ada", "Bob"]})

    with pytest.raises(TypeError, match="Columns must be numeric"):
        scaler_class(columns=["name"]).fit(data)


@pytest.mark.parametrize("scaler_class", [StandardScaler, MinMaxScaler])
def test_scaler_columns_none_rejects_mixed_dataframe(scaler_class):
    data = pd.DataFrame({"age": [10.0, 20.0], "name": ["Ada", "Bob"]})

    with pytest.raises(TypeError, match="Columns must be numeric"):
        scaler_class().fit(data)


@pytest.mark.parametrize("scaler_class", [StandardScaler, MinMaxScaler])
def test_scaler_rejects_non_numeric_transform_data(scaler_class):
    scaler = scaler_class(columns=["age"]).fit(pd.DataFrame({"age": [10.0, 20.0]}))

    with pytest.raises(TypeError, match="Columns must be numeric"):
        scaler.transform(pd.DataFrame({"age": ["ten", "twenty"]}))


@pytest.mark.parametrize("scaler_class", [StandardScaler, MinMaxScaler])
def test_scaler_rejects_missing_columns(scaler_class):
    scaler = scaler_class(columns=["missing"])

    with pytest.raises(ValueError, match="Columns not found"):
        scaler.fit(pd.DataFrame({"age": [10, 20]}))


@pytest.mark.parametrize("scaler_class", [StandardScaler, MinMaxScaler])
def test_scaler_transform_before_fit_raises_runtime_error(scaler_class):
    scaler = scaler_class(columns=["age"])

    with pytest.raises(RuntimeError, match="must be fitted"):
        scaler.transform(pd.DataFrame({"age": [10, 20]}))
