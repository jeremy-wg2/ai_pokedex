import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import os

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

def scrape_and_download(base_url, start, end, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    for i in range(start, end + 1):
        page_url = base_url.format(i)
        try:
            response = requests.get(page_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Find the meta tag with property 'og:image'
                meta_tag = soup.find('meta', property='og:image')
                if meta_tag:
                    image_url = meta_tag['content']
                    save_path = os.path.join(save_folder, f'Sprite{i:04d}.png')
                    download_image(image_url, save_path)
                else:
                    print(f"Meta tag 'og:image' not found for page {page_url}")
            else:
                print(f"Failed to access page {page_url}. Status code: {response.status_code}")
        except Exception as e:
            print(f"An error occurred while processing page {page_url}: {e}")

# Example usage:
base_url = 'https://archives.bulbagarden.net/wiki/File:{:03d}.png'  # URL template
start = 1  # Starting number
end = 1025  # Ending number
save_folder = 'pokemon-sprite'

scrape_and_download(base_url, start, end, save_folder)


