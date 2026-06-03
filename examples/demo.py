import pandas as pd

from cleanml import Pipeline
from cleanml.impute import MissingValueImputer, MeanStrategy, ModeStrategy
from cleanml.scale import MinMaxScaler
from cleanml.encode import OneHotEncoder
from cleanml.typos import CategoryTypoFixer


df = pd.DataFrame({
    "age": [19, 22, None, 25],
    "salary": [25000, None, 32000, 40000],
    "city": ["Istanbul", "Ankara", "Izmir", "Istanbulll"],
    "gender": ["Male", "Female", None, "Male"]
})

pipeline = Pipeline([
    MissingValueImputer(strategy=MeanStrategy(), columns=["age", "salary"]),
    MissingValueImputer(strategy=ModeStrategy(), columns=["gender"]),
    CategoryTypoFixer(
        column="city",
        valid_categories=["Istanbul", "Ankara", "Izmir"]
    ),
    MinMaxScaler(columns=["age", "salary"]),
    OneHotEncoder(columns=["city", "gender"])
])
clean_df = pipeline.fit_transform(df)

print(clean_df)