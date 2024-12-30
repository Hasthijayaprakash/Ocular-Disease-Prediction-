"""
extract_embeddings_goooglenet.py

This script extracts feature embeddings from images using a fine-tuned googlenet model.
It saves the embeddings along with image names and corresponding labels to a Parquet file.


We made use of X-vision-helper and integrated it with the custom file traversal

Reference: "https://github.com/moayadeldin/X-vision-helper/tree/master"
"""



import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torchvision.models import googlenet, GoogLeNet_Weights


MODEL_PATH = 'E:\\mld\\advanced\\googlenet\\models\\googlenet_embed_ready_15epoc.pth'  # Path to fine-tuned GoogLeNet model
OUTPUT_EMBEDDINGS = 'E:\\mld\\advanced\\googlenet\\output\\feature_embeddings.parquet'  # Output Parquet for embeddings
BATCH_SIZE = 32  
NUM_WORKERS = 4  


# Defining the ImageDataset
class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.csv_file = Path(csv_file)
        self.transform = transform
        self.data = pd.read_csv(self.csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 1])
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
    """Modifying the model to output embeddings instead of classification scores."""
    model.fc = nn.Identity()
    model.aux_logits = False
    model.aux1 = nn.Identity()
    model.aux2 = nn.Identity()
    return model

# This function to extract embeddings
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

    csv_file = 'E:\\mld\\advanced\\googlenet\\B.csv'
    img_dir = 'E:\\mld\\advanced\\preprocessed_images'

        #configuration to use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # Initializing dataset and dataloader
    dataset = ImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Loading the fine-tuned GoogLeNet model
    num_classes = len(dataset.data.iloc[:, 1].unique())
    model = googlenet(weights=GoogLeNet_Weights.DEFAULT, aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes)
    model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = adjust_model(model)
    model.to(device)

    # Extracting the embeddings
    embeddings, image_names, labels = extract_embeddings(model, data_loader, device)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Image Names count: {len(image_names)}")
    print(f"Labels shape: {labels.shape}")

    # Output Folder
    output_embeddings_path = Path(OUTPUT_EMBEDDINGS)
    output_embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    # Saving embeddings, image names, and labels to a single Parquet file
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.insert(0, 'image_name', image_names)  
    embeddings_df['label'] = labels  

    embeddings_df.to_parquet(OUTPUT_EMBEDDINGS, index=False)
    print(f"Embeddings with image names and labels saved to {OUTPUT_EMBEDDINGS}")

if __name__ == "__main__":
    main()
