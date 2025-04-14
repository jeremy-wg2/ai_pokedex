import os
import pandas as pd
import shutil

# Define paths
source_folder = 'pokemon-dataset-full'
destination_folder = 'validation-dataset'
csv_file = 'validation_metadata.csv'

# Create destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Read CSV file
df = pd.read_csv(csv_file)

# Iterate over file names in the CSV and copy them
for file_name in df['file_name']:
    source_path = os.path.join(source_folder, file_name)
    destination_path = os.path.join(destination_folder, file_name)

    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
    else:
        print(f"File {file_name} not found in {source_folder}.")

print("All specified files have been copied.")