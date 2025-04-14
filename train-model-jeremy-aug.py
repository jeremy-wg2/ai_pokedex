import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
from io import BytesIO

# Load your local dataset files
dataset = load_dataset('parquet', data_files={'train': 'train-aug-3.parquet', 'validation': 'validation-aug-3.parquet'})

# Update the model configuration to match the number of classes
num_classes = 1025  # Set this to the number of classes in your dataset

# Load the model and feature extractor
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=num_classes
)
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Preprocess the dataset
def preprocess_images(examples):
    images = []
    for image_data in examples['image_data']:
        # If image_data is in bytes, convert using BytesIO
        if isinstance(image_data, bytes):
            pil_image = Image.open(BytesIO(image_data))
        else:
            # Assuming image_data is a NumPy array or compatible format
            pil_image = Image.fromarray(np.array(image_data))

        # Check for transparency and convert appropriately
        if pil_image.mode == 'P' or (pil_image.mode == 'RGBA' and 'transparency' in pil_image.info):
            pil_image = pil_image.convert("RGBA")

        # Convert the image to RGB mode
        pil_image = pil_image.convert("RGB")

        # Apply the feature extractor
        image_tensor = feature_extractor(pil_image, return_tensors='pt')['pixel_values'][0]
        images.append(image_tensor)

    examples['pixel_values'] = images
    return examples

# Apply the preprocessing function
dataset = dataset.map(preprocess_images, batched=True)

# Set format for PyTorch
dataset.set_format(type='torch', columns=['pixel_values', 'label'])

# Define a compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}

# Define training arguments
training_args = TrainingArguments(
    output_dir="./vit-pokemon",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,  # Pass the custom metrics function
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./vit-pokemon-model-jeremy-aug-3")