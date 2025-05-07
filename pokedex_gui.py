import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
import os
import csv
import time  # To introduce delay
import pygame  # For playing audio
import glob

def find_pokemon_image(pokemon_name, dataset_folder="pokemon-gui-white-bg"):
    """Find the image file for the predicted Pokémon."""
    search_pattern = os.path.join(dataset_folder, f"*{pokemon_name}*.png")
    matching_files = glob.glob(search_pattern)  # Find all files matching the pattern
    if matching_files:
        # Return the first matching file (or handle if multiple matches are found)
        return matching_files[0]
    else:
        return None

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


def predict_pokemon(image_path, csv_file_path="pokemon.csv"):
    """Predict the Pokémon from the image."""
    # Load the Pokémon names from the CSV file
    pokemon_map = load_pokemon_names(csv_file_path)

    try:
        # Load the model and feature extractor
        model = ViTForImageClassification.from_pretrained(
            "models/google_vit_pokemon_classification_augmented_dataset"
        )
        feature_extractor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )

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
        predicted_pokemon_name = pokemon_map.get(predicted_class_idx, "Unknown Pokémon")

        return predicted_pokemon_name

    except Exception as e:
        return f"An error occurred: {e}"


def play_audio(file_path):
    """Play an audio file using pygame."""
    pygame.mixer.init()  # Initialize the pygame mixer
    pygame.mixer.music.load(file_path)  # Load the audio file
    pygame.mixer.music.play()  # Play the audio


def browse_file():
    """Browse for an image file."""
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.webp")]
    )
    if file_path:
        # Clear the previous result
        result_label.config(text="")  # Reset the result label
        root.update()  # Update the GUI to reflect the cleared result

        # Display the image first
        display_image(file_path)
        root.update()  # Force the GUI to update and display the image immediately

        # Play the audio clip
        play_audio("whos_that_pokemon.mp3")

        # Wait for 7 seconds before displaying the result
        time.sleep(4)

        # Predict Pokémon and show the result
        result = predict_pokemon(file_path)
        result_label.config(text=f"It's {result}!")
        result_label.place(x=300, y=460, anchor="center")  # Centered below the button

def browse_file():
    """Browse for an image file."""
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.webp")]
    )
    if file_path:
        # Clear previous result and update GUI
        result_label.config(text="")
        display_image(file_path)  # Display the uploaded image
        play_audio("whos_that_pokemon.mp3")  # Play the audio

        # Wait 4000ms (4 seconds) before showing the result
        root.after(4000, display_result_after_delay, file_path)

def display_result_after_delay(file_path):
    # Predict the Pokémon
    result = predict_pokemon(file_path)

    # Update the result label
    result_label.config(text=f"It's {result}!")
    result_label.place(x=300, y=460, anchor="center")  # Centered below the button

    # Find and display the Pokémon's image
    pokemon_image_path = find_pokemon_image(result)
    if pokemon_image_path:
        display_image(pokemon_image_path)  # Update the image to the result Pokémon's image
    else:
        result_label.config(text=f"It's {result}! (Image not found)")

def display_image(image_path):
    """Display the selected image in the GUI."""
    image = Image.open(image_path)
    image.thumbnail((300, 300))  # Resize the image to fit in the GUI
    photo = ImageTk.PhotoImage(image)

    image_label.config(image=photo)
    image_label.image = photo


# Create the main application window
root = tk.Tk()
root.title("Pokémon Classifier")
root.geometry("600x600")  # Adjust window size to fit the background

# Load and set the background image
background_image = Image.open("background.png")  # Ensure background.png is in the same directory
background_photo = ImageTk.PhotoImage(background_image)

# Create a canvas to hold the background image
canvas = tk.Canvas(root, width=600, height=600)
canvas.pack(fill="both", expand=True)

# Add the background image to the canvas
canvas.create_image(0, 0, image=background_photo, anchor="nw")

# Add GUI elements on top of the canvas
title_label = tk.Label(root, text="Pokémon Classifier", font=("Arial", 16), bg="white")
title_label_window = canvas.create_window(300, 50, window=title_label)

# Customize the image label (Remove black outline by setting borderwidth and highlightthickness to 0)
image_label = tk.Label(root, borderwidth=0, highlightthickness=0)
image_label_window = canvas.create_window(300, 200, window=image_label)

# Customize the browse button (Red with dark blue text)
browse_button = tk.Button(
    root,
    text="Who's That Pokèmon?",
    command=browse_file,
    bg="red",  # Button background color
    fg="dark blue",  # Button text color
    font=("Arial", 12, "bold")  # Optional: Make text bold
)
browse_button_window = canvas.create_window(300, 400, window=browse_button)

# Customize the result label (initially hidden, centered below the button)
result_label = tk.Label(root, text="", font=("Arial", 16, "bold"), bg="white", fg="black")  # Bigger font size
result_label.place_forget()  # Hide the result label initially

# Start the GUI event loop
root.mainloop()