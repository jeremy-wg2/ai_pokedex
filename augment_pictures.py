import os
import random
import torch
from torchvision import transforms
from PIL import Image


#This script takes 3 inputs,
# folder_path = 'validation-dataset'  # Path to your folder containing images
# output_folder = 'validation-dataset-aug-3'  # Path to save augmented images
# num_augmentations = 3  # Number of augmented images per original image
# It start with 1 transformation per picture, then increases based on num_transforms. So if want more transformations you can decrease this value

# Define individual transformations
transformations = [
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.RandomGrayscale(p=0.2),
]

# Define a function to select transformations
def get_transformations(num_transforms):
    # Select random transformations and add tensor conversion and normalization
    selected_transforms = random.sample(transformations, num_transforms)
    return transforms.Compose(selected_transforms + [
        transforms.ToTensor(),  # Convert to tensor
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),  # Add Gaussian noise
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def augment_images_in_folder(folder_path, output_folder, num_augmentations):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(image_path)

                # Check if the image is in palette mode with transparency
                if image.mode == 'P':
                    image = image.convert('RGBA')  # Convert to RGBA to preserve transparency
                    image = image.convert('RGB')   # Convert to RGB (discarding alpha channel)
                else:
                    image = image.convert('RGB')

            except Exception as e:
                print(f"Error opening {filename}: {e}")
                continue

            base_filename, _ = os.path.splitext(filename)
            original_filename = f"{base_filename}-0.png"
            image.save(os.path.join(output_folder, original_filename))

            for i in range(1, num_augmentations + 1):
                num_transforms = (i - 1) // 2 + 1 #update this value if you want more transformations per picture
                augmentation_pipeline = get_transformations(num_transforms)

                augmented_image_tensor = augmentation_pipeline(image)
                augmented_image = transforms.ToPILImage()(augmented_image_tensor)

                augmented_filename = f"{base_filename}-{i:02}.png"
                augmented_image.save(os.path.join(output_folder, augmented_filename))
                print(f"Saved {augmented_filename}")

# Parameters
folder_path = 'validation-dataset'  # Path to your folder containing images
output_folder = 'validation-dataset-aug-3'  # Path to save augmented images
num_augmentations = 3  # Number of augmented images per original image

augment_images_in_folder(folder_path, output_folder, num_augmentations)