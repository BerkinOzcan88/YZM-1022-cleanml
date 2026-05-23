import pandas as pd

from cleanml import Pipeline
from cleanml.impute import MissingValueImputer, MeanStrategy
from cleanml.scale import StandardScaler, MinMaxScaler


df = pd.DataFrame({
    "age": [19, 22, None, 25],
    "salary": [25000, None, 32000, 40000]
})

pipeline = Pipeline([
    MissingValueImputer(strategy=MeanStrategy(), columns=["age", "salary"]),
    MinMaxScaler(columns=["age", "salary"])
])

clean_df = pipeline.fit_transform(df)

print(clean_df)