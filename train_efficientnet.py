import os
import time
import copy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Import the existing preprocessing function
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing.preprocess import preprocess_mri

# Constants
DATA_DIR = Path("/Users/macbookpro/Downloads/FInal Year Project/Project/data")
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"
MODEL_SAVE_DIR = Path("/Users/macbookpro/Downloads/FInal Year Project/Project/brain_tumour_project/models")
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary', 'metastasis', 'pediatric_glioma']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

# Hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_CLASSES = len(CLASSES)


class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, class_to_idx):
        self.data_dir = Path(data_dir)
        self.class_to_idx = class_to_idx
        self.image_paths = []
        self.labels = []
        
        for cls_name in class_to_idx.keys():
            cls_dir = self.data_dir / cls_name
            if cls_dir.is_dir():
                for img_path in cls_dir.glob("*.[jp][pn]*"): # match jpg, png, jpeg
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Output of preprocess_mri is (224, 224, 3) float32 in [0, 1]
            processed_img = preprocess_mri(image_path=img_path)
            
            # PyTorch expects channels first: (C, H, W)
            processed_img = np.transpose(processed_img, (2, 0, 1))
            
            # Additional normalization typically expected by ImageNet pretrained models
            # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # Since EfficientNet_B4_Weights.DEFAULT expects this, we apply it:
            tensor_img = torch.tensor(processed_img, dtype=torch.float32)
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor_img = (tensor_img - mean) / std
            
            return tensor_img, torch.tensor(label, dtype=torch.long)
        
        except Exception as e:
            # If an image fails to load/preprocess, return a zero tensor to prevent crashing
            print(f"Warning: Failed to process {img_path}: {e}")
            return torch.zeros((3, 224, 224), dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def build_model(num_classes):
    print("Loading pre-trained EfficientNet-B4...")
    weights = EfficientNet_B4_Weights.DEFAULT
    model = efficientnet_b4(weights=weights)
    
    # Modify the classifier head for our number of classes
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model


def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            all_preds = []
            all_labels = []
            
            pbar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()}")
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = accuracy_score(all_labels, all_preds)
            
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print()
        
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val Acc: {best_acc:.4f}")
    
    model.load_state_dict(best_model_wts)
    return model


def main():
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        print(f"Error: Data directories not found. Check if {DATA_DIR} exists.")
        return
        
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    print("Initializing Datasets and Dataloaders...")
    train_dataset = BrainTumorDataset(TRAIN_DIR, CLASS_TO_IDX)
    val_dataset = BrainTumorDataset(TEST_DIR, CLASS_TO_IDX)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = build_model(NUM_CLASSES)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS, device=device)
    
    save_path = MODEL_SAVE_DIR / "efficientnet_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
