"""
train_googlenet.py
This script trains a Googlenet model on the labeled image dataset and saves the trained model weights.
We made use of X-vision-helper and integrated it with the custom file traversal

Reference: "https://github.com/moayadeldin/X-vision-helper/tree/master"

"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torchvision.models import googlenet, GoogLeNet_Weights


MODEL_SAVE_PATH = 'E:\\mld\\advanced\\googlenet\\models\\googlenet_embed_ready_15epoc.pth'  # Path to save the trained model
BATCH_SIZE = 32  
NUM_EPOCHS = 15  
LEARNING_RATE = 1e-4 


class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str or Path): Path to the CSV file with image filenames and labels.
            img_dir (str or Path): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
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

        return image, label

def main():
     #configuration to use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # Initialize dataset and dataloader
    csv_file = 'E:\\mld\\advanced\\googlenet\\A.csv'  
    img_dir = 'E:\\mld\\advanced\\preprocessed_images'
    dataset = ImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)  

    # Initialize the GoogLeNet model
    num_classes = len(dataset.data.iloc[:, 1].unique())
    model = googlenet(weights=GoogLeNet_Weights.DEFAULT, aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes)
    model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)
    model = model.to(device)

    # Defining loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(data_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs, aux1_outputs, aux2_outputs = model(images)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux1_outputs, labels)
            loss3 = criterion(aux2_outputs, labels)
            loss = loss1 + 0.3 * (loss2 + loss3)  

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataset)
        epoch_acc = correct / total
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
