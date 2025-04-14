import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from PIL import Image
import numpy as np
from io import BytesIO

#This scripts evaluates a pre-trained model with a local dataset.

# Load the test dataset
dataset = load_dataset('parquet', data_files={'test': 'test-noback.parquet'})

# Load the trained model and feature extractor
model = ViTForImageClassification.from_pretrained("./vit-pokemon-model-jeremy-aug-3")
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

    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)

    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels, preds)

    # If this is a binary classification task, calculate ROC AUC
    roc_auc = None
    if len(set(labels)) == 2:
        roc_auc = roc_auc_score(labels, preds)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)

    # Return all metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    if roc_auc is not None:
        metrics["roc_auc"] = roc_auc

    return metrics

# Create a Trainer instance with no training arguments
trainer = Trainer(
    model=model,
    eval_dataset=dataset,
    compute_metrics=compute_metrics,  # Pass the custom metrics function
)

# Evaluate the model
eval_results = trainer.evaluate()

# Print the evaluation results
# Print the eval_results dictionary to inspect available keys
print("Evaluation Results:")
print(eval_results)

# Access and print metrics if they exist
if 'accuracy' in eval_results:
    print(f"Test Accuracy: {eval_results['accuracy']:.4f}")
else:
    print("Accuracy metric not found in eval_results.")

if 'precision' in eval_results:
    print(f"Test Precision: {eval_results['precision']:.4f}")
else:
    print("Precision metric not found in eval_results.")

if 'recall' in eval_results:
    print(f"Test Recall: {eval_results['recall']:.4f}")
else:
    print("Recall metric not found in eval_results.")

if 'f1' in eval_results:
    print(f"Test F1 Score: {eval_results['f1']:.4f}")
else:
    print("F1 score metric not found in eval_results.")

# Print ROC AUC if applicable
if 'roc_auc' in eval_results:
    print(f"Test ROC AUC: {eval_results['roc_auc']:.4f}")
else:
    print("ROC AUC metric not found in eval_results.")