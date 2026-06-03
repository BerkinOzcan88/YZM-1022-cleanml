from cleanml.pipeline import Pipeline

from cleanml.impute import (
    ImputationStrategy,
    MeanStrategy,
    MedianStrategy,
    ModeStrategy,
    ConstantStrategy,
    MissingValueImputer,
)

from cleanml.scale import (
    StandardScaler,
    MinMaxScaler,
)

from cleanml.encode import (
    LabelEncoder,
    OneHotEncoder,
)

from cleanml.typos import (
    edit_distance,
    CategoryTypoFixer,
)

from cleanml.parallel import ParallelColumnTransformer

from cleanml.observers import (
    PipelineObserver,
    ConsoleLogger,
    EventHistory,
)

from cleanml.loaders import (
    BaseLoader,
    CSVLoader,
    JSONLoader,
    DataLoaderFactory,
)

from cleanml.functional import (
    compose,
    remove_duplicates,
    strip_whitespace,
    lowercase_column_names,
    replace_spaces_in_column_names,
    drop_columns,
)


__version__ = "0.1.0"


__all__ = [
    "Pipeline",

    "ImputationStrategy",
    "MeanStrategy",
    "MedianStrategy",
    "ModeStrategy",
    "ConstantStrategy",
    "MissingValueImputer",

    "StandardScaler",
    "MinMaxScaler",

    "LabelEncoder",
    "OneHotEncoder",

    "edit_distance",
    "CategoryTypoFixer",

    "ParallelColumnTransformer",

    "PipelineObserver",
    "ConsoleLogger",
    "EventHistory",

    "BaseLoader",
    "CSVLoader",
    "JSONLoader",
    "DataLoaderFactory",

    "compose",
    "remove_duplicates",
    "strip_whitespace",
    "lowercase_column_names",
    "replace_spaces_in_column_names",
    "drop_columns",
]