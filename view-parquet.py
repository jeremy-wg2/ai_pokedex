import pandas as pd

# Path to your Parquet file
parquet_file_path = 'validation.parquet'

# Read the Parquet file into a DataFrame
df = pd.read_parquet(parquet_file_path)

# Display the first few rows of the DataFrame
print(df.head())

# Optionally, you can view the columns and data types
print(df.info())

# Display the first two rows of the DataFrame
print(df.head(2))