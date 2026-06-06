import pandas as pd
import pytest

from cleanml import CSVLoader, DataLoaderFactory, JSONLoader


def test_csv_loader_reads_csv_file(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"age": [10, 20], "city": ["Ankara", "Istanbul"]}).to_csv(path, index=False)

    result = CSVLoader(str(path)).load()

    assert result.to_dict("list") == {
        "age": [10, 20],
        "city": ["Ankara", "Istanbul"],
    }


def test_json_loader_reads_json_file(tmp_path):
    path = tmp_path / "data.json"
    pd.DataFrame({"age": [10, 20], "city": ["Ankara", "Istanbul"]}).to_json(path)

    result = JSONLoader(str(path)).load()

    assert result.to_dict("list") == {
        "age": [10, 20],
        "city": ["Ankara", "Istanbul"],
    }


def test_loader_raises_file_not_found_for_missing_file(tmp_path):
    loader = CSVLoader(str(tmp_path / "missing.csv"))

    with pytest.raises(FileNotFoundError, match="File not found"):
        loader.load()


def test_data_loader_factory_creates_loader_from_file_type(tmp_path):
    csv_loader = DataLoaderFactory.create(" CSV ", str(tmp_path / "data.csv"))
    json_loader = DataLoaderFactory.create("json", str(tmp_path / "data.json"))

    assert isinstance(csv_loader, CSVLoader)
    assert isinstance(json_loader, JSONLoader)


def test_data_loader_factory_creates_loader_from_file_path(tmp_path):
    csv_loader = DataLoaderFactory.from_file_path(str(tmp_path / "data.csv"))
    json_loader = DataLoaderFactory.from_file_path(str(tmp_path / "data.json"))

    assert isinstance(csv_loader, CSVLoader)
    assert isinstance(json_loader, JSONLoader)


def test_data_loader_factory_rejects_unsupported_file_type(tmp_path):
    with pytest.raises(ValueError, match="Unsupported file type"):
        DataLoaderFactory.create("xlsx", str(tmp_path / "data.xlsx"))


def test_data_loader_factory_rejects_unsupported_extension(tmp_path):
    with pytest.raises(ValueError, match="Unsupported file extension"):
        DataLoaderFactory.from_file_path(str(tmp_path / "data.xlsx"))
