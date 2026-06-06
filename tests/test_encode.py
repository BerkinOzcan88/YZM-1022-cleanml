import pandas as pd
import pytest

from cleanml import LabelEncoder, OneHotEncoder


def test_label_encoder_maps_categories_and_does_not_mutate_input():
    data = pd.DataFrame({"city": ["Ankara", "Istanbul", "Ankara"], "age": [10, 20, 30]})
    original = data.copy(deep=True)

    result = LabelEncoder(columns=["city"]).fit_transform(data)

    assert result is not data
    assert result["city"].tolist() == [0, 1, 0]
    assert result["age"].tolist() == [10, 20, 30]
    pd.testing.assert_frame_equal(data, original)


def test_label_encoder_unknown_category_becomes_nan():
    encoder = LabelEncoder(columns=["city"]).fit(pd.DataFrame({"city": ["Ankara", "Istanbul"]}))

    result = encoder.transform(pd.DataFrame({"city": ["Ankara", "Izmir"]}))

    assert result.loc[0, "city"] == 0
    assert pd.isna(result.loc[1, "city"])


def test_one_hot_encoder_creates_columns_and_drops_original_by_default():
    data = pd.DataFrame({"city": ["Ankara", "Istanbul", "Ankara"]})

    result = OneHotEncoder(columns=["city"]).fit_transform(data)

    assert "city" not in result.columns
    assert result["city_Ankara"].tolist() == [1, 0, 1]
    assert result["city_Istanbul"].tolist() == [0, 1, 0]


def test_one_hot_encoder_can_keep_original_column():
    data = pd.DataFrame({"city": ["Ankara", "Istanbul"]})

    result = OneHotEncoder(columns=["city"], drop_original=False).fit_transform(data)

    assert result["city"].tolist() == ["Ankara", "Istanbul"]
    assert result["city_Ankara"].tolist() == [1, 0]
    assert result["city_Istanbul"].tolist() == [0, 1]


def test_one_hot_encoder_unknown_category_gets_zeroes_for_known_categories():
    encoder = OneHotEncoder(columns=["city"]).fit(pd.DataFrame({"city": ["Ankara", "Istanbul"]}))

    result = encoder.transform(pd.DataFrame({"city": ["Izmir"]}))

    assert result.to_dict("list") == {
        "city_Ankara": [0],
        "city_Istanbul": [0],
    }


@pytest.mark.parametrize("encoder_class", [LabelEncoder, OneHotEncoder])
def test_encoder_rejects_missing_columns(encoder_class):
    encoder = encoder_class(columns=["missing"])

    with pytest.raises(ValueError, match="Columns not found"):
        encoder.fit(pd.DataFrame({"city": ["Ankara"]}))


@pytest.mark.parametrize("encoder_class", [LabelEncoder, OneHotEncoder])
def test_encoder_transform_before_fit_raises_runtime_error(encoder_class):
    encoder = encoder_class(columns=["city"])

    with pytest.raises(RuntimeError, match="must be fitted"):
        encoder.transform(pd.DataFrame({"city": ["Ankara"]}))
