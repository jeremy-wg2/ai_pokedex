import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from PIL import Image
import numpy as np

#This scripts evaluates a pre-trained model with a dataset on hugging face.

# Load the test dataset
dataset = load_dataset("JJMack/pokemon-classification-gen1-9", split='test')

# Load the trained model and feature extractor
model = ViTForImageClassification.from_pretrained("./vit-pokemon-model-jeremy")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Preprocess the dataset
def preprocess_images(examples):
    # Convert images to RGB format using PIL and apply the feature extractor
    images = [Image.fromarray(np.array(image)).convert("RGB") for image in examples['image_data']]
    pixel_values = feature_extractor(images, return_tensors="pt").pixel_values
    examples['pixel_values'] = pixel_values
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
    if len(set(labels)) == 2:
        roc_auc = roc_auc_score(labels, preds)
    else:
        roc_auc = None

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
print(f"Test Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Test Precision: {eval_results['eval_precision']:.4f}")
print(f"Test Recall: {eval_results['eval_recall']:.4f}")
print(f"Test F1 Score: {eval_results['eval_f1']:.4f}")

# Print ROC AUC if applicable
if 'eval_roc_auc' in eval_results:
    print(f"Test ROC AUC: {eval_results['eval_roc_auc']:.4f}")