import os
import pandas as pd
from PIL import Image
from io import BytesIO

#This script takes images and a csv as metadata and outputs a parquet file.

#Inputs --------------------

images_folder = 'test-dataset'
metadata_path = 'test_metadata.csv'
metadata_df = pd.read_csv(metadata_path)

#End_Inputs --------------------

data = []

for image_name in os.listdir(images_folder):
    image_path = os.path.join(images_folder, image_name)

    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_binary = buffered.getvalue()  # Get raw binary data

    image_data = {
        'file_name': image_name,
        'image_data': img_binary  # Store binary data
    }

    if 'file_name' in metadata_df.columns:
        image_metadata = metadata_df.loc[metadata_df['file_name'] == image_name]
        if not image_metadata.empty:
            image_data.update(image_metadata.to_dict(orient='records')[0])

    data.append(image_data)

df = pd.DataFrame(data)
parquet_file_path = 'test.parquet'
df.to_parquet(parquet_file_path)