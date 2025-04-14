import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from datasets import load_dataset
import re
import os
import argparse

#This is the script you run to call the google_vit_pokemon_classification_poc model. Here is an example command
#Python3 pokedex_poc.py abra.png, which will take the abra.png and output the pokemon's name Abra.


def convert_to_jpg(input_file_path):
    """
    Convert an image file to JPEG format.

    :param input_file_path: str, path to the input image file (.png or .webp)
    :return: str, path to the converted JPEG image file
    """
    # Open the image file
    with Image.open(input_file_path) as img:
        # Convert the image to RGB (JPEG doesn't support transparency)
        rgb_image = img.convert("RGB")

        # Create the output file path with .jpg extension
        base_name = os.path.splitext(input_file_path)[0]  # Remove the original extension
        output_file_path = f"{base_name}.jpg"

        # Save the image in JPEG format
        rgb_image.save(output_file_path, "JPEG")

        return output_file_path

def main(image_path):
    # Load the model and feature extractor
    model = ViTForImageClassification.from_pretrained("models/google_vit_pokemon_classification_poc")
    feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    # Load the dataset
    dataset = load_dataset("keremberke/pokemon-classification", "full")

    # Prepare a dictionary to map class indices to image file paths
    class_to_path_map = {}

    # Populate the dictionary with image file paths associated with their labels
    for example in dataset['train']:
        image_file_path = example['image_file_path']
        label = example['labels']
        class_to_path_map[label] = image_file_path

    # Convert the image to JPEG if necessary
    if image_path.lower().endswith(('.png', '.webp')):
        image_path = convert_to_jpg(image_path)

    # Load the converted image
    image = Image.open(image_path)

    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Make predictions
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted class index
    predicted_class_idx = logits.argmax(-1).item()

    # Retrieve the image file path associated with the predicted class index
    predicted_image_file_path = class_to_path_map.get(predicted_class_idx, "")

    # Apply the regex to extract the Pokémon name from the image file path
    match = re.search(r"/([^/]+)/[^/]+\.jpg$", predicted_image_file_path)
    if match:
        pokemon_name = match.group(1)
    else:
        pokemon_name = "No match found"

    # Print the Pokémon name
    print(f"Predicted Pokémon name: {pokemon_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" The Pokedex predicts the Pokémon is.")
    parser.add_argument("image_path", type=str, help="Path to the image file (.png or .webp)")

    args = parser.parse_args()
    main(args.image_path)