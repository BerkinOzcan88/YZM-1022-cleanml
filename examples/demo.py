from pathlib import Path

import pandas as pd

from cleanml import (
    Pipeline,
    DataLoaderFactory,
    MissingValueImputer,
    MeanStrategy,
    ModeStrategy,
    MinMaxScaler,
    OneHotEncoder,
    CategoryTypoFixer,
    ConsoleLogger,
    compose,
    strip_whitespace,
    lowercase_column_names,
    replace_spaces_in_column_names,
)


DATA_PATH = Path(__file__).parent / "messy_data.csv"


def create_demo_csv() -> None:
    df = pd.DataFrame({
        "Age": [19, 22, None, 25],
        "Salary": [25000, None, 32000, 40000],
        "City": [" Istanbull ", " Ankra ", " Izmri ", " Istanbul "],
        "Gender": [" Male ", " Female ", None, " Male "],
    })

    df.to_csv(DATA_PATH, index=False)


def main() -> None:
    create_demo_csv()

    loader = DataLoaderFactory.from_file_path(str(DATA_PATH))
    raw_df = loader.load()

    print("\nOriginal data:")
    print(raw_df)

    basic_cleaning = compose(
        strip_whitespace,
        lowercase_column_names,
        replace_spaces_in_column_names,
    )

    cleaned_names_df = basic_cleaning(raw_df)

    print("\nAfter functional cleaning:")
    print(cleaned_names_df)

    pipeline = Pipeline([
        MissingValueImputer(
            strategy=MeanStrategy(),
            columns=["age", "salary"],
        ),
        MissingValueImputer(
            strategy=ModeStrategy(),
            columns=["gender"],
        ),
        CategoryTypoFixer(
            column="city",
            valid_categories=["Istanbul", "Ankara", "Izmir"],
        ),
        MinMaxScaler(
            columns=["age", "salary"],
        ),
        OneHotEncoder(
            columns=["city", "gender"],
        ),
    ])

    pipeline.add_observer(ConsoleLogger())

    final_df = pipeline.fit_transform(cleaned_names_df)

    print("\nFinal ML-ready data:")
    print(final_df)


if __name__ == "__main__":
    main()
