import pandas as pd

from cleanml import (
    compose,
    drop_columns,
    lowercase_column_names,
    remove_duplicates,
    replace_spaces_in_column_names,
    strip_whitespace,
)


def test_compose_runs_dataframe_functions_in_order_without_mutating_input():
    data = pd.DataFrame({"First Name": [" Ada ", " Bob "]})
    original = data.copy(deep=True)
    cleaner = compose(
        strip_whitespace,
        lowercase_column_names,
        replace_spaces_in_column_names,
    )

    result = cleaner(data)

    assert result.to_dict("list") == {"first_name": ["Ada", "Bob"]}
    pd.testing.assert_frame_equal(data, original)


def test_remove_duplicates_returns_unique_rows():
    data = pd.DataFrame({"x": [1, 1, 2], "y": ["a", "a", "b"]})

    result = remove_duplicates(data)

    assert result.to_dict("list") == {"x": [1, 2], "y": ["a", "b"]}


def test_strip_whitespace_only_changes_object_columns():
    data = pd.DataFrame({"name": [" Ada ", "Bob "], "age": [10, 20]})

    result = strip_whitespace(data)

    assert result.to_dict("list") == {"name": ["Ada", "Bob"], "age": [10, 20]}


def test_lowercase_column_names_converts_all_names_to_strings():
    data = pd.DataFrame({1: [10], "Name": ["Ada"]})

    result = lowercase_column_names(data)

    assert list(result.columns) == ["1", "name"]


def test_replace_spaces_in_column_names_uses_underscores():
    data = pd.DataFrame({"First Name": ["Ada"], "Last Name": ["Lovelace"]})

    result = replace_spaces_in_column_names(data)

    assert list(result.columns) == ["First_Name", "Last_Name"]


def test_drop_columns_ignores_missing_columns():
    data = pd.DataFrame({"name": ["Ada"], "age": [10]})

    result = drop_columns("age", "missing")(data)

    assert result.to_dict("list") == {"name": ["Ada"]}
