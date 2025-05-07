import os
import csv

#This script was used for debugging. It checks all the images in a folder and highlights images thare are in the csv_file
# and not in the folder.

def check_files_in_folder(csv_file, folder_path):
    # Read the CSV file
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)

        # Iterate over each row in the CSV
        for row in csv_reader:
            file_name = row['file_name']
            file_path = os.path.join(folder_path, file_name)

            # Check if the file exists in the folder
            if not os.path.isfile(file_path):
                print(f"File '{file_name}' does NOT exist in the folder.")

# Example usage
csv_file = 'train_metadata.csv'  # Replace with the path to your CSV file
folder_path = 'train-dataset-aug-3'     # Replace with the path to your folder

check_files_in_folder(csv_file, folder_path)