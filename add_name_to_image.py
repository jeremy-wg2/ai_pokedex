import os
import pandas as pd

#When scraping the pictures off Bulbpedia, the pictures often only had the pokedex number.
# This script uses the pokemon.csv file to determine the pokemons name and update the image name


def rename_images(image_directory, csv_path, rows_to_process=None):
    # Read the CSV file using pandas
    df = pd.read_csv(csv_path)

    # If rows_to_process is specified, slice the DataFrame
    if rows_to_process is not None:
        df = df.head(rows_to_process)

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Extract the number and name from the row
        number = str(row['Number']).zfill(4)  # Ensure the number is 4 digits
        number2 = str(row['Number']).zfill(3)  # Ensure the number is 4 digits
        name = row['Name']

        # Construct the old filename and the new filename
        #old_filename = f'Sprite{number}.png'
        old_filename = f'{number}-{name}.png'
        new_filename = f'{number}-{name}-.png'

        # Define the full paths to the old and new filenames
        old_filepath = os.path.join(image_directory, old_filename)
        new_filepath = os.path.join(image_directory, new_filename)

        # Rename the file if the old file exists
        if os.path.exists(old_filepath):
            os.rename(old_filepath, new_filepath)
            print(f'Renamed {old_filename} to {new_filename}')
        else:
            print(f'File {old_filename} not found')

    print('Renaming complete!')

# Example usage
image_directory = 'all-images'
csv_path = 'pokemon.csv'
rows_to_process = 1025  # Specify how many rows to process

rename_images(image_directory, csv_path, rows_to_process)
