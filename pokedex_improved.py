import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from datasets import load_dataset
import os
import csv

#This is the script you run to call the google_vit_pokemon_classification_augmented_dataset model. Here is an example command
#Python3 pokedex_improved.py pokemon_lion.png, which will take the pokemon_lion.png and output the pokemon Litleo.


def convert_to_jpg(input_file_path):
    """Convert an image file to JPEG format."""
    with Image.open(input_file_path) as img:
        rgb_image = img.convert("RGB")
        base_name = os.path.splitext(input_file_path)[0]
        output_file_path = f"{base_name}.jpg"
        rgb_image.save(output_file_path, "JPEG")
        return output_file_path

def load_pokemon_names(csv_file_path):
    """Load Pokémon names from a CSV file into a dictionary."""
    pokemon_map = {}
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            number = int(row['Number'])
            name = row['Name']
            pokemon_map[number] = name
    return pokemon_map

def main(image_path):
    # Hardcoded path to the CSV file
    csv_file_path = "pokemon.csv"

    # Load the Pokémon names from the CSV file
    pokemon_map = load_pokemon_names(csv_file_path)

    try:
        # Load the model and feature extractor
        model = ViTForImageClassification.from_pretrained("models/google_vit_pokemon_classification_augmented_dataset")
        feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

        # Load the dataset
        dataset = load_dataset("JJMack/pokemon-classification-gen1-9")

        # Convert the image to JPEG if necessary
        if image_path.lower().endswith(('.png', '.webp')):
            image_path = convert_to_jpg(image_path)

        # Load the converted image
        image = Image.open(image_path)

        # Preprocess the image
        inputs = feature_extractor(images=image, return_tensors="pt")

        # Make predictions
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the predicted class index
        predicted_class_idx = logits.argmax(-1).item()

        # Retrieve the Pokémon name associated with the predicted class index
        predicted_pokemon_name = pokemon_map.get(predicted_class_idx, "Unknown Pokémon")  # +1 because CSV uses 1-based indexing

        # Print the Pokémon name
        print(f"Predicted Pokémon: {predicted_pokemon_name}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict the Pokémon from an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file (.png or .webp)")

    args = parser.parse_args()
    main(args.image_path)