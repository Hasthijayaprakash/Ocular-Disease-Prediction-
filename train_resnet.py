"""
train_resnet.py
This script trains a ResNet50 model on the labeled image dataset and saves the trained model weights.
We made use of X-vision-helper and integrated it with the custom file traversal

Reference: "https://github.com/moayadeldin/X-vision-helper/tree/master"

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# the below parameters will be used 
MODEL_SAVE_PATH = 'E:\\mld\\advanced\\resnet\\models\\resnet_embed_ready_15epoc.pth' # path to the 'A' 
BATCH_SIZE = 32  
NUM_EPOCHS = 15  
LEARNING_RATE = 1e-4  


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
        label = self.data.iloc[idx, 1]
        img_path = self.img_dir / img_name
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)

        return image, label

def main():
    #configuration to use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initializing dataset, image directory and model
    csv_file = 'E:\\mld\\advanced\\resnet\\embed_ready.csv'
    img_dir = 'E:\\mld\\advanced\\preprocessed_images'
    dataset = ImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    num_classes = len(dataset.data.iloc[:,1].unique())
    model.fc = nn.Linear(num_ftrs, num_classes)  
    model = model.to(device)

    # Defining the  loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Main training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(data_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataset)
        epoch_acc = correct / total
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
