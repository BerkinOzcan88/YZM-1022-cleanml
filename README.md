# cleanml

cleanml is an installable Python machine learning preprocessing library. It
helps users turn messy tabular data into model-ready data by loading files,
cleaning columns, imputing missing values, fixing category typos, scaling
numeric features, encoding categorical features, and composing these operations
into reusable pipelines.

This project is a reusable Python package, not a web app, dashboard, or command
line-only script.

## Installation

```bash
pip install .
```

## Quick Start

```python
import pandas as pd

from cleanml import (
    Pipeline,
    MissingValueImputer,
    MeanStrategy,
    ModeStrategy,
    MinMaxScaler,
    OneHotEncoder,
    CategoryTypoFixer,
    compose,
    strip_whitespace,
    lowercase_column_names,
    replace_spaces_in_column_names,
)


raw_data = pd.DataFrame({
    "Age": [19, 22, None, 25],
    "Salary": [25000, None, 32000, 40000],
    "City": [" Istanbull ", " Ankra ", " Izmri ", " Istanbul "],
    "Gender": [" Male ", " Female ", None, " Male "],
})

basic_cleaning = compose(
    strip_whitespace,
    lowercase_column_names,
    replace_spaces_in_column_names,
)

data = basic_cleaning(raw_data)

pipeline = Pipeline([
    MissingValueImputer(MeanStrategy(), columns=["age", "salary"]),
    MissingValueImputer(ModeStrategy(), columns=["gender"]),
    CategoryTypoFixer(
        column="city",
        valid_categories=["Istanbul", "Ankara", "Izmir"],
    ),
    MinMaxScaler(columns=["age", "salary"]),
    OneHotEncoder(columns=["city", "gender"]),
])

model_ready_data = pipeline.fit_transform(data)
print(model_ready_data)
```

## Main Features

- `DataLoaderFactory`, `CSVLoader`, and `JSONLoader` load tabular data from
  files.
- `MissingValueImputer` fills missing values with mean, median, mode, or a
  constant value.
- `StandardScaler` and `MinMaxScaler` scale numeric columns.
- `LabelEncoder` and `OneHotEncoder` encode categorical columns.
- `CategoryTypoFixer` corrects close category misspellings using edit distance.
- `Pipeline` runs multiple preprocessing steps in order.
- `ParallelColumnTransformer` runs independent column transformations with
  threads.
- Functional helpers such as `compose`, `strip_whitespace`, and
  `drop_columns` support pure, composable DataFrame cleaning.

## Architecture

The library is organized around a transformer architecture inspired by common
machine learning libraries. All preprocessing classes inherit from
`BaseTransformer`, implement `fit` and `transform`, and return transformed
copies instead of mutating the user's original DataFrame.

The public API is exported from `cleanml/__init__.py`, so users can import the
main tools directly from `cleanml`.

## Design Patterns Used

| Pattern | Where | How it is used |
| --- | --- | --- |
| Pipeline / Layered Architecture | `Pipeline` | Runs a sequence of transformers where each step receives the previous step's output. |
| Strategy Pattern | `ImputationStrategy`, `MeanStrategy`, `MedianStrategy`, `ModeStrategy`, `ConstantStrategy` | Allows `MissingValueImputer` to swap the missing-value algorithm without changing the imputer. |
| Factory Pattern | `DataLoaderFactory` | Creates the correct loader (`CSVLoader` or `JSONLoader`) from a file type or file extension. |
| Observer Pattern | `PipelineObserver`, `ConsoleLogger`, `EventHistory` | Lets external objects listen to pipeline events such as step start, step finish, and errors. |

## Course Learning Outcomes

| Requirement | Where it is fulfilled | Explanation |
| --- | --- | --- |
| 1. Object-Oriented Programming | `BaseTransformer`, `Pipeline`, scalers, encoders, imputers, loaders | The library uses classes, inheritance, abstract base classes, encapsulated state, and polymorphic `fit`/`transform` methods. |
| 2. Functional Programming | `cleanml.functional` | `compose` is a higher-order function that combines pure DataFrame functions such as `strip_whitespace` and `drop_columns`. |
| 3. Concurrency | `ParallelColumnTransformer` | Uses `ThreadPoolExecutor` to fit and transform independent column operations in parallel. |
| 4. Recursion / Dynamic Programming | `edit_distance` in `cleanml.typos` | Implements Levenshtein edit distance with a dynamic-programming table to support category typo correction. |
| 5. SOLID Principles | Separate modules for loading, imputing, scaling, encoding, typo fixing, observers, and pipelines | Each class has a focused responsibility, and core components depend on abstractions such as `BaseTransformer`, `ImputationStrategy`, and `PipelineObserver`. |
| 6. Architecture & Design Patterns | `Pipeline`, Strategy, Factory, Observer | The project uses a clear preprocessing pipeline architecture and multiple documented design patterns. |

## Demo

Run the example script:

```bash
python examples/demo.py
```

The demo creates a small messy dataset, loads it, applies functional cleaning,
runs a preprocessing pipeline, logs pipeline events, and prints the final
machine-learning-ready DataFrame. The script generates or updates
`examples/messy_data.csv` as part of the demo.

## Testing

Run the unit tests with:

```bash
python -m pytest -q
```

At the time of this README update, the test suite passes with:

```text
70 passed
```

The tests cover loaders, imputers, scalers, encoders, typo fixing, functional
helpers, observers, pipelines, and parallel transformations.
