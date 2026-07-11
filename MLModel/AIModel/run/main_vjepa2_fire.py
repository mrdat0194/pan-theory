import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import sys

# Add data_pipeline to path to import data_helper
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data_pipeline'))
from data_helper import data_pipeline_nn

class FireSmokeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Dummy logic: assumes 'fire' in filename means fire, 'smoke' means smoke
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if f.endswith('.jpg') or f.endswith('.png'):
                    self.images.append(f)
                    if 'fire' in f.lower():
                        self.labels.append(1) # Flame
                    elif 'smoke' in f.lower():
                        self.labels.append(2) # Smoke
                    else:
                        self.labels.append(0) # Safe

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# Pseudo V-JEPA 2 Backbone
class VJEPA2_Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super(VJEPA2_Classifier, self).__init__()
        # self.backbone = torch.hub.load('facebookresearch/vjepa2', 'vjepa_vit_small')
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes) # Flame vs Smoke vs Safe
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classification_head(features)
        return logits

if __name__ == "__main__":
    print("Initializing V-JEPA 2 for Fire/Smoke Detection...")
    
    # Paths to centralized data
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'FireSmoke', 'test_img')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    try:
        dataset = FireSmokeDataset(data_dir, transform=transform)
        if len(dataset) == 0:
            print("Warning: Dataset is empty.")
            
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        if train_size > 0:
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            print(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} testing samples.")
            
            model = VJEPA2_Classifier(num_classes=3)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            # Dummy Training Loop
            model.train()
            print("Starting classification fine-tuning...")
            for epoch in range(1, 6):
                running_loss = 0.0
                for images, labels in train_loader:
                    optimizer.zero_grad()
                    logits = model(images)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                print(f"Epoch {epoch}/5 - Training Loss: {running_loss/len(train_loader):.4f}")
            
            # Dummy Testing Loop
            model.eval()
            print("Starting evaluation...")
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            test_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    logits = model(images)
                    loss = criterion(logits, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            print(f"Test Loss: {test_loss/len(test_loader):.4f}")
            print(f"Test Accuracy: {100 * correct / total:.2f}%")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
