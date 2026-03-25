import os
import requests
import zipfile

URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "tiny-imagenet-200.zip")
EXTRACT_PATH = DATA_DIR


def download():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(ZIP_PATH):
        print("Dataset già scaricato.")
        return

    print("Downloading Tiny ImageNet...")
    response = requests.get(URL, stream=True)

    with open(ZIP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Download completato!")


def extract():
    print("Estrazione dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
    print("Estrazione completata!")


if __name__ == "__main__":
    download()
    extract()