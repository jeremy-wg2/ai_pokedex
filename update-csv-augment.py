import csv

def add_rows_to_csv(input_file, output_file, X):
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)

        # Store all original and new rows
        rows = []

        for row in reader:
            # Add the original row
            rows.append(row)

            # Add X new rows with modified filenames
            for i in range(X + 1):
                new_row = row.copy()

                # Modify the filename to include the correct number format
                if i == 0:
                    new_row[0] = new_row[0].replace('.png', f'-{i}.png')
                else:
                    new_row[0] = new_row[0].replace('.png', f'-{i:02}.png')

                rows.append(new_row)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(header)
        # Write all rows
        writer.writerows(rows)


# Example usage
input_file = 'validation_metadata.csv'  # Replace with your input file path
output_file = 'validation_metadata_3.csv'  # Replace with your desired output file path
X = 4  # Set the number of additional rows to add for each original row (remember 0 is a row)
add_rows_to_csv(input_file, output_file, X)