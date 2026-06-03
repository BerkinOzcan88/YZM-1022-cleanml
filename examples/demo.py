from cleanml.loaders import DataLoaderFactory

loader = DataLoaderFactory.from_file_path("examples/messy_data.csv")
df = loader.load()

print(df)