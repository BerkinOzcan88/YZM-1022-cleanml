import pandas as pd
import pytest

from cleanml import CategoryTypoFixer, edit_distance


def test_edit_distance_counts_insertions_deletions_and_substitutions():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "abc") == 0


def test_category_typo_fixer_replaces_close_categories_without_mutating_input():
    data = pd.DataFrame({"city": ["Istanbull", "Ankra", "Izmri", None]})
    original = data.copy(deep=True)

    result = CategoryTypoFixer(
        column="city",
        valid_categories=["Istanbul", "Ankara", "Izmir"],
        max_distance=2,
    ).fit_transform(data)

    assert result["city"].tolist()[:3] == ["Istanbul", "Ankara", "Izmir"]
    assert pd.isna(result.loc[3, "city"])
    pd.testing.assert_frame_equal(data, original)


def test_category_typo_fixer_keeps_values_outside_max_distance():
    data = pd.DataFrame({"city": ["London"]})

    result = CategoryTypoFixer(
        column="city",
        valid_categories=["Istanbul", "Ankara", "Izmir"],
        max_distance=2,
    ).fit_transform(data)

    assert result["city"].tolist() == ["London"]


def test_category_typo_fixer_can_force_replacement_when_max_distance_is_none():
    data = pd.DataFrame({"city": ["Istanbulxxxx"]})

    result = CategoryTypoFixer(
        column="city",
        valid_categories=["Istanbul", "Ankara", "Izmir"],
        max_distance=None,
    ).fit_transform(data)

    assert result["city"].tolist() == ["Istanbul"]


def test_category_typo_fixer_rejects_empty_valid_categories():
    fixer = CategoryTypoFixer(column="city", valid_categories=[])

    with pytest.raises(ValueError, match="valid_categories"):
        fixer.fit(pd.DataFrame({"city": ["Ankara"]}))


def test_category_typo_fixer_rejects_missing_column():
    fixer = CategoryTypoFixer(column="missing", valid_categories=["Ankara"])

    with pytest.raises(ValueError, match="Columns not found"):
        fixer.fit(pd.DataFrame({"city": ["Ankara"]}))
