"""
extract_embeddings_resnet.py

This script extracts feature embeddings from images using a fine-tuned ResNet50 model.
It saves the embeddings along with image names and corresponding labels to a Parquet file.

We made use of X-vision-helper and integrated it with the custom file traversal

Reference: "https://github.com/moayadeldin/X-vision-helper/tree/master"
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pyarrow


MODEL_PATH = 'E:\\mld\\advanced\\resnet\\models\\resnet_finetuned_A_epoc_15.pth'  # Path to fine-tuned ResNet50 model
OUTPUT_EMBEDDINGS = 'E:\\mld\\advanced\\resnet\\output\\feature_embeddings.parquet'  # Output Parquet for embeddings
BATCH_SIZE = 32  
NUM_WORKERS = 4  


class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str or Path): Path to the CSV or Excel file with image filenames and labels.
            img_dir (str or Path): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.img_dir = Path(img_dir)
        self.csv_file = Path(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        img_path = self.img_dir / img_name
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, img_name, label

# Function to adjust the model to output embeddings
def adjust_model(model):
    modules = list(model.children())[:-1] 
    model = nn.Sequential(*modules)
    return model

# Function to extract embeddings
def extract_embeddings(model, data_loader, device):
    model.eval()
    embeddings = []
    image_names = []
    labels = []

    with torch.no_grad():
        for images, img_names, lbls in tqdm(data_loader, desc="Extracting Embeddings"):
            images = images.to(device)
            emb = model(images)
            emb = emb.view(emb.size(0), -1)
            embeddings.append(emb.cpu().numpy())
            image_names.extend(img_names)
            labels.extend(lbls.numpy().tolist())

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    return embeddings, image_names, labels


def main():
    # Paths Defined
    csv_file = 'E:\\mld\\advanced\\resnet\\B.csv'
    img_dir = 'E:\\mld\\advanced\\preprocessed_images'

    #configuration to use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # Initializing dataset
    dataset = ImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Loading the fine-tuned ResNet50 model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features

    # Dynamically determine number of classes based on data
    num_classes = len(dataset.data.iloc[:,1].unique())
    model.fc = nn.Linear(num_ftrs, num_classes)  # Replace with actual number of classes
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = adjust_model(model)
    model.to(device)

    # Extract embeddings
    embeddings, image_names, labels = extract_embeddings(model, data_loader, device)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Image Names count: {len(image_names)}")
    print(f"Labels shape: {labels.shape}")

    # Ensure output directory exists
    output_embeddings_path = Path(OUTPUT_EMBEDDINGS)
    output_embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    # Save embeddings, image names, and labels to a single Parquet file
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.insert(0, 'image_name', image_names)  
    embeddings_df['label'] = labels

    embeddings_df.to_parquet(OUTPUT_EMBEDDINGS, index=False)
    print(f"Embeddings with image names and labels saved to {OUTPUT_EMBEDDINGS}")

if __name__ == "__main__":
    main()
