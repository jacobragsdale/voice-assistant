import os
import sys
import requests
import zipfile
import tqdm
from pathlib import Path

def download_model():
    """Download a Vosk speech recognition model."""
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Small English model (87MB)
    model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    model_path = models_dir / "vosk-model-small-en-us-0.15.zip"
    model_dir = models_dir / "vosk-model-small-en-us-0.15"
    
    # Check if model is already downloaded and extracted
    if model_dir.exists():
        print(f"Model already exists at {model_dir}")
        return str(model_dir)
    
    # Download the model
    print(f"Downloading model from {model_url}")
    print("This may take a few minutes...")
    
    response = requests.get(model_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(model_path, 'wb') as file:
        for data in tqdm.tqdm(
            response.iter_content(block_size),
            total=total_size // block_size,
            unit='KB',
            unit_scale=True
        ):
            file.write(data)
    
    # Extract the model
    print(f"Extracting model to {model_dir}")
    with zipfile.ZipFile(model_path, 'r') as zip_ref:
        zip_ref.extractall(models_dir)
    
    # Clean up zip file
    model_path.unlink()
    
    print(f"Model downloaded and extracted to {model_dir}")
    return str(model_dir)

if __name__ == "__main__":
    model_path = download_model()
    print(f"Model path: {model_path}")
    print("You can now use this model with the voice transcription app.") 