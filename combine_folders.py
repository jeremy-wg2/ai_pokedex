import os
import shutil

#This script helped me combine all of the folders that I created when scraping Bulbpedia into one folder of all images.

def move_images_to_single_folder(source_dir, destination_dir):
    # Check if the source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"Created destination directory '{destination_dir}'.")

    # List all directories and files in the source directory
    for subdir, _, files in os.walk(source_dir):
        print(f"Scanning directory: {subdir}")
        for file in files:
            # Check if the file is an image by its extension
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                source_file_path = os.path.join(subdir, file)
                destination_file_path = os.path.join(destination_dir, file)

                # Move the image file to the destination directory
                try:
                    shutil.move(source_file_path, destination_file_path)
                    print(f"Moved: {source_file_path} to {destination_file_path}")
                except Exception as e:
                    print(f"Error moving file {source_file_path}: {e}")

if __name__ == "__main__":
    source_directory = "images-ready"
    destination_directory = "all-images"

    move_images_to_single_folder(source_directory, destination_directory)