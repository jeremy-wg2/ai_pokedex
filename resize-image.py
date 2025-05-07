import os
from PIL import Image

def resize_images(input_folder, output_folder, size=(256, 256)):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over each file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.png'):  # Process only PNG files
            file_path = os.path.join(input_folder, file_name)

            # Open the image
            with Image.open(file_path) as img:
                # Resize the image using LANCZOS filter for high quality
                img_resized = img.resize(size, Image.LANCZOS)

                # Save the resized image to the output folder
                output_path = os.path.join(output_folder, file_name)
                img_resized.save(output_path)

            print(f'Resized and saved: {file_name}')

# Specify the input and output folders
input_folder = 'pokemon-sprite-7-back-s'
output_folder = f'{input_folder}-256-256'

# Call the function to resize images
resize_images(input_folder, output_folder)