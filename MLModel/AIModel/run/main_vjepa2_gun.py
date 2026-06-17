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

class WeaponSDataset(Dataset):
    def __init__(self, data_dir, bbox_dir, transform=None):
        self.data_dir = data_dir
        self.bbox_dir = bbox_dir
        self.transform = transform
        self.images = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Load bounding box if exists
        bbox_path = os.path.join(self.bbox_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        bbox = []
        if os.path.exists(bbox_path):
            with open(bbox_path, 'r') as f:
                bbox = f.read().strip().split()
                # Dummy parse for YOLO/COCO format
                bbox = [float(x) for x in bbox[1:]] if len(bbox) > 1 else [0,0,0,0]
        else:
            bbox = [0, 0, 0, 0]

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(bbox)

# Pseudo V-JEPA 2 Backbone
class VJEPA2_Detector(nn.Module):
    def __init__(self, num_classes=1):
        super(VJEPA2_Detector, self).__init__()
        # In practice, this would load from facebookresearch/vjepa2
        # self.backbone = torch.hub.load('facebookresearch/vjepa2', 'vjepa_vit_small')
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.localization_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 4) # Bounding box x, y, w, h
        )

    def forward(self, x):
        features = self.backbone(x)
        bbox = self.localization_head(features)
        return bbox

if __name__ == "__main__":
    print("Initializing V-JEPA 2 for Gun Detection...")
    
    # Paths to centralized data
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'WeaponS', 'WeaponS')
    bbox_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'WeaponS', 'WeaponS_bbox')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # This assumes dataset is already there
    try:
        dataset = WeaponSDataset(data_dir, bbox_dir, transform=transform)
        
        # We can use data_pipeline logic if we had tabular data, but for images we use standard splits
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        print(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} testing samples.")
        
        model = VJEPA2_Detector()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        # Dummy Training Loop
        model.train()
        print("Starting fine-tuning...")
        # for epoch in range(5):
        #     for images, bboxes in train_loader:
        #         optimizer.zero_grad()
        #         preds = model(images)
        #         loss = criterion(preds, bboxes)
        #         loss.backward()
        #         optimizer.step()
        print("Training loop setup complete. (Uncomment loop to run)")

    except FileNotFoundError:
        print(f"Dataset not found at {data_dir}. Ensure data pipeline migration is complete.")
