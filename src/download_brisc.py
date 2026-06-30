import os
import zipfile
import urllib.request
import time

URL = "https://zenodo.org/api/records/17524350/files/brisc2025.zip/content"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "brisc2025.zip")

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
    percent = min(int(count * block_size * 100 / total_size), 100)
    print(f"\rDownloading... {percent}% | {progress_size / (1024 * 1024):.2f} MB / {total_size / (1024 * 1024):.2f} MB | Speed: {speed} KB/s", end="")

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    if not os.path.exists(ZIP_PATH):
        print(f"Downloading BRISC dataset from Zenodo...")
        try:
            urllib.request.urlretrieve(URL, ZIP_PATH, reporthook)
            print("\nDownload complete!")
        except Exception as e:
            print(f"\nError downloading dataset: {e}")
            return
    else:
        print(f"Zip file already exists at {ZIP_PATH}. Skipping download.")

    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Extraction complete!")
        os.remove(ZIP_PATH)
        print("Removed temporary zip file.")
    except Exception as e:
        print(f"Error during extraction: {e}")

if __name__ == "__main__":
    main()
