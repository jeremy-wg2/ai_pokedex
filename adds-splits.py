import pandas as pd


#Summary: This script takes

# Load the CSV file into a DataFrame
df = pd.read_csv('pokemon_metadata.csv')

# Add the new 'split' column with default value as empty string
df['split'] = ''

# Define the split ratios
train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

# Function to assign splits within each pokedex group
def assign_splits(group):
    # Calculate number of rows for each split
    total_rows = len(group)
    num_train = int(total_rows * train_ratio)
    num_validation = int(total_rows * validation_ratio)
    num_test = total_rows - num_train - num_validation

    # Shuffle the group rows
    shuffled_group = group.sample(frac=1, random_state=42)

    # Assign splits
    shuffled_group.iloc[:num_train, shuffled_group.columns.get_loc('split')] = 'train'
    shuffled_group.iloc[num_train:num_train + num_validation, shuffled_group.columns.get_loc('split')] = 'validation'
    shuffled_group.iloc[num_train + num_validation:, shuffled_group.columns.get_loc('split')] = 'test'

    return shuffled_group

# Apply function to each group of unique pokedex_num
df = df.groupby('pokedex_num', group_keys=False).apply(assign_splits)

# Save the updated DataFrame to a new CSV file
df.to_csv('pokemon_metadata_full.csv', index=False)