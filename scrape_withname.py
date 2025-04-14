import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import os
import pandas as pd

def download_image(url, save_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image.save(save_path, 'PNG')
            print(f"Image downloaded and saved to {save_path}")
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

def scrape_and_download(base_url, csv_file, save_folder):
    os.makedirs(save_folder, exist_ok=True)

    # Load CSV file
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Failed to read CSV file: {e}")
        return

    for index, row in data.iterrows():
        number = row['Number']
        name = row['Name']
        # Format the number to ensure it has four digits
        number_formatted = f"{int(number):03d}"
        page_url = base_url.format(number_formatted, name)
        try:
            response = requests.get(page_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Find the meta tag with property 'og:image'
                meta_tag = soup.find('meta', property='og:image')
                if meta_tag:
                    image_url = meta_tag['content']
                    # Change the output filename to include both number and name
                    save_path = os.path.join(save_folder, f'{number_formatted}-{name}.png')
                    download_image(image_url, save_path)
                else:
                    print(f"Meta tag 'og:image' not found for page {page_url}")
            else:
                print(f"Failed to access page {page_url}. Status code: {response.status_code}")
        except Exception as e:
            print(f"An error occurred while processing page {page_url}: {e}")

# Example usage:
base_url = 'https://archives.bulbagarden.net/wiki/File:{}{}_PSMD.png'  # URL template
csv_file = 'pokemon.csv'  # CSV file path containing 'Number' and 'Name' columns
save_folder = 'pokemon-mystary_dungion'

scrape_and_download(base_url, csv_file, save_folder)