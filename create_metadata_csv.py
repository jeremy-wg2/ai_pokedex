import os
import csv

#This script helped me create a CSV to be used as the meta data for the parqet file.

#Inputs --------------------

# Specify the directory containing the images
directory = "pokemon-dataset"

# Specify the names of the CSV files
output_csv = "pokemon_metadata.csv"
additional_data_csv = "pokemon-stats-updated.csv"  # Updated CSV filename

#End_Inputs --------------------

# Initialize a list to store the rows for the CSV
csv_data = []

# Read the updated additional data CSV into a dictionary
additional_data = {}

with open(additional_data_csv, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Use pokedex_num as the key, which is now 4 digits
        additional_data[row['pokedex_num']] = row

def determine_generation(pokedex_num):
    """Determine the generation based on the pokedex number."""
    num = int(pokedex_num)
    if 1 <= num <= 151:
        return "1"
    elif 152 <= num <= 251:
        return "2"
    elif 252 <= num <= 386:
        return "3"
    elif 387 <= num <= 493:
        return "4"
    elif 494 <= num <= 649:
        return "5"
    elif 650 <= num <= 721:
        return "6"
    elif 722 <= num <= 809:
        return "7"
    elif 810 <= num <= 905:
        return "8"
    elif 906 <= num <= 1025:
        return "9"
    else:
        return "Unknown"

# Loop through each file in the directory
for filename in os.listdir(directory):
    # Check if the file is a PNG (you can extend this to other formats if needed)
    if filename.endswith(".png"):
        # Split the filename into parts based on the '-' delimiter
        parts = filename.split('-')

        # Extract the pokedex number and the name
        pokedex_num = parts[0]
        name = parts[1]

        # Determine the generation
        generation = determine_generation(pokedex_num)

        # Retrieve additional data using 4-digit pokedex_num
        additional_info = additional_data.get(pokedex_num, {})

        # Determine if the file represents a shiny Pokémon
        shiny = "yes" if "-s.png" in filename else "no"

        # Prepare the row for CSV data
        csv_row = [
            filename, pokedex_num, name, generation,
            additional_info.get('Type 1', ''),
            additional_info.get('Type 2', ''),
            additional_info.get('HP', ''),
            additional_info.get('Attack', ''),
            additional_info.get('Defense', ''),
            additional_info.get('Sp.Attack', ''),
            additional_info.get('Sp.Defense', ''),
            additional_info.get('Speed', ''),
            shiny
        ]

        # Append the new row to the CSV data list
        csv_data.append(csv_row)

# Sort the CSV data based on the pokedex_num (as integer)
csv_data.sort(key=lambda x: int(x[1]))

# Write the sorted CSV data to a file
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow([
        "file_name", "pokedex_num", "name", "generation",
        "Type 1", "Type 2", "HP", "Attack", "Defense", "Sp.Attack", "Sp.Defense", "Speed",
        "shiny"
    ])
    # Write the sorted data rows
    writer.writerows(csv_data)

print(f"CSV file '{output_csv}' created successfully with additional data merged and shiny status as the last column.")