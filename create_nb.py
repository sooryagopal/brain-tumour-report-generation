import json

cells = []

def add_code_cell(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    })

# Cell 1
add_code_cell("!pip install torch torchvision opencv-python matplotlib seaborn scikit-learn tqdm pillow numpy pandas")

# Cell 2
add_code_cell("""import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from tqdm import tqdm
import PIL
import pathlib
import warnings
warnings.filterwarnings('ignore')""")

# Cell 3
add_code_cell("""from pathlib import Path
import torch

DATASET_ROOT = Path('./dataset')
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
NUM_CLASSES = 4
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
SAVE_DIR = Path('./outputs')
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")""")

# Cell 4
add_code_cell("""import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def explore_dataset(dataset_root):
    \"\"\"
    Explore the dataset and plot class distribution.
    \"\"\"
    data = []
    for split in ['Training', 'Testing']:
        for cls in CLASS_NAMES:
            path = dataset_root / split / cls
            if path.exists():
                count = len(list(path.glob('*.jpg')))
                data.append({'Split': split, 'Class': cls, 'Count': count})
    
    if not data:
        print("Dataset not found at", dataset_root)
        return
        
    df = pd.DataFrame(data)
    print("Dataset counts:")
    pivot_df = df.pivot(index='Class', columns='Split', values='Count')
    print(pivot_df)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Class', y='Count', hue='Split')
    plt.title('Dataset Distribution')
    plt.savefig(SAVE_DIR / 'class_distribution.png')
    plt.show()

explore_dataset(DATASET_ROOT)""")

# Cell 5
add_code_cell("""import cv2
import numpy as np

def preprocess_mri(image_path, target_size=(224,224)):
    \"\"\"
    Preprocess MRI image.
    \"\"\"
    # 1. Load image with cv2, convert to grayscale
    img = cv2.imread(str(image_path))
    if img is None:
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize to 224x224
    resized = cv2.resize(gray, target_size)
    
    # 3. GaussianBlur (kernel 5x5)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    
    # 4. Otsu threshold + erode 2 iterations + dilate 2 iterations
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)
    
    # 5. findContours → get largest contour → find extreme points → crop ROI
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cropped = resized[y:y+h, x:x+w]
    else:
        cropped = resized
        
    if cropped.size == 0:
        cropped = resized
        
    # 6. Resize cropped ROI back to 224x224
    cropped_resized = cv2.resize(cropped, target_size)
    
    # 7. Gamma correction gamma=1.2
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(cropped_resized, table)
    
    # 8. CLAHE clipLimit=2.0 tileGridSize=(8,8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gamma_corrected)
    
    # 9. Divide by 255 to normalise
    normalized = clahe_img.astype(np.float32) / 255.0
    
    # 10. Stack to 3 channels for EfficientNet
    stacked = np.stack((normalized,)*3, axis=-1)
    
    return stacked""")

# Cell 6
add_code_cell("""import matplotlib.pyplot as plt
import cv2
import numpy as np

def visualize_preprocessing(dataset_root, save_dir):
    \"\"\"
    Visualize preprocessing steps for one sample image from each class.
    \"\"\"
    fig, axes = plt.subplots(4, 7, figsize=(20, 12))
    steps = ['Original', 'Grayscale', 'Blur', 'Threshold', 'ROI Crop', 'Gamma', 'CLAHE']
    
    for i, title in enumerate(steps):
        axes[0, i].set_title(title)
        
    for row, cls in enumerate(CLASS_NAMES):
        path = dataset_root / 'Training' / cls
        if not path.exists():
            continue
        img_paths = list(path.glob('*.jpg'))
        if not img_paths:
            continue
        img_path = str(img_paths[0])
        
        # Step 1
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Step 2
        resized = cv2.resize(gray, (224, 224))
        
        # Step 3
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        
        # Step 4
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        eroded = cv2.erode(thresh, kernel, iterations=2)
        dilated = cv2.dilate(eroded, kernel, iterations=2)
        
        # Step 5
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cropped = resized[y:y+h, x:x+w]
        else:
            cropped = resized
        if cropped.size == 0:
            cropped = resized
            
        cropped_resized = cv2.resize(cropped, (224, 224))
        
        # Step 6
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(cropped_resized, table)
        
        # Step 7
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(gamma_corrected)
        
        images = [img_rgb, gray, blurred, dilated, cropped, gamma_corrected, clahe_img]
        
        for col, image in enumerate(images):
            axes[row, col].imshow(image, cmap='gray' if col > 0 else None)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].text(-40, image.shape[0]//2, cls, rotation=90, va='center', fontsize=12)
                
    plt.tight_layout()
    plt.savefig(save_dir / 'preprocessing_steps.png')
    plt.show()

visualize_preprocessing(DATASET_ROOT, SAVE_DIR)""")

# Cell 7
add_code_cell("""from torch.utils.data import Dataset
from PIL import Image

class BrainTumourDataset(Dataset):
    \"\"\"
    Custom Dataset class for Brain Tumour MRI dataset.
    \"\"\"
    def __init__(self, root_dir, split, class_names, transform, img_size):
        self.root_dir = root_dir
        self.split = split
        self.class_names = class_names
        self.transform = transform
        self.img_size = img_size
        self.samples = []
        
        for idx, cls in enumerate(self.class_names):
            cls_dir = self.root_dir / self.split / cls
            if cls_dir.exists():
                for img_path in cls_dir.glob('*.jpg'):
                    self.samples.append((img_path, idx))
                    
        print(f"Initialized BrainTumourDataset {self.split} with {len(self.samples)} samples.")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        preprocessed_array = preprocess_mri(img_path, target_size=(self.img_size, self.img_size))
        
        # Convert preprocessed float32 array (0-1) to uint8 (0-255) for PIL
        preprocessed_uint8 = (preprocessed_array * 255).astype(np.uint8)
        img_pil = Image.fromarray(preprocessed_uint8)
        
        if self.transform:
            img_tensor = self.transform(img_pil)
            
        return img_tensor, label""")

# Cell 8
add_code_cell("""import torchvision.transforms as transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = BrainTumourDataset(DATASET_ROOT, 'Training', CLASS_NAMES, train_transform, IMG_SIZE)
test_dataset = BrainTumourDataset(DATASET_ROOT, 'Testing', CLASS_NAMES, val_transform, IMG_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)""")

# Cell 9
add_code_cell("""import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

def build_model(num_classes, freeze_backbone=True):
    \"\"\"
    Build EfficientNet-B4 model with custom classifier.
    \"\"\"
    weights = EfficientNet_B4_Weights.IMAGENET1K_V1
    model = efficientnet_b4(weights=weights)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    # Replace classifier
    in_features = model.classifier[1].in_features # typically 1792 for b4
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(256, num_classes)
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    
    return model

model = build_model(NUM_CLASSES).to(DEVICE)""")

# Cell 10
add_code_cell("""import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)""")

# Cell 11
add_code_cell("""def train_one_epoch(model, loader, criterion, optimizer, device):
    \"\"\"
    Train the model for one epoch.
    \"\"\"
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.2f}%"})
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    \"\"\"
    Evaluate the model.
    \"\"\"
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)""")

# Cell 12
add_code_cell("""history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

best_val_acc = 0.0
epochs_no_improve = 0
early_stop_patience = 7

print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'LR':<10}")
print("-" * 75)

for epoch in range(1, EPOCHS + 1):
    if epoch == 5:
        print("Unfreezing backbone...")
        for param in model.parameters():
            param.requires_grad = True
        for g in optimizer.param_groups:
            g['lr'] = 1e-5
            
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, _, _, _ = evaluate(model, test_loader, criterion, DEVICE)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"{epoch:<6} | {train_loss:<10.4f} | {train_acc:<10.4f} | {val_loss:<10.4f} | {val_acc:<10.4f} | {current_lr:<10.2e}")
    
    scheduler.step(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_DIR / 'efficientnet_b4_best.pth')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        
    if epochs_no_improve >= early_stop_patience:
        print(f"Early stopping triggered after {epoch} epochs.")
        break""")

# Cell 13
add_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history['train_acc'], label='Train Acc')
ax1.plot(history['val_acc'], label='Val Acc')
ax1.set_title('Accuracy Curve')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history['train_loss'], label='Train Loss')
ax2.plot(history['val_loss'], label='Val Loss')
ax2.set_title('Loss Curve')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.savefig(SAVE_DIR / 'training_curves.png')
plt.show()""")

# Cell 14
add_code_cell("""from sklearn.metrics import classification_report, roc_auc_score

# Load best model
model.load_state_dict(torch.load(SAVE_DIR / 'efficientnet_b4_best.pth'))

test_loss, test_acc, all_preds, all_labels, all_probs = evaluate(model, test_loader, criterion, DEVICE)

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
print(f"AUC-ROC (macro OvR): {auc_roc:.4f}")""")

# Cell 15
add_code_cell("""from sklearn.metrics import confusion_matrix

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(SAVE_DIR / 'confusion_matrix.png')
plt.show()""")

# Cell 16
add_code_cell("""import random

def show_sample_predictions(model, dataset, device, num_samples=8):
    \"\"\"
    Show sample predictions.
    \"\"\"
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, label = dataset[idx]
            img_batch = img_tensor.unsqueeze(0).to(device)
            
            output = model(img_batch)
            prob = torch.nn.functional.softmax(output, dim=1)
            pred = output.argmax(1).item()
            conf = prob[0, pred].item()
            
            # Unnormalize image for display
            img_display = img_tensor.cpu().numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_display = std * img_display + mean
            img_display = np.clip(img_display, 0, 1)
            
            ax = axes[i]
            ax.imshow(img_display)
            ax.axis('off')
            
            true_name = CLASS_NAMES[label]
            pred_name = CLASS_NAMES[pred]
            
            title_color = 'green' if label == pred else 'red'
            title = f"True: {true_name}\\nPred: {pred_name} ({conf:.2f})"
            ax.set_title(title, color=title_color)
            
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'sample_predictions.png')
    plt.show()

show_sample_predictions(model, test_dataset, DEVICE)""")

# Cell 17
add_code_cell("""from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='macro')
rec = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

results = {
    'Model': ['EfficientNet-B4'],
    'Accuracy': [acc],
    'Precision': [prec],
    'Recall': [rec],
    'F1': [f1],
    'AUC': [auc_roc]
}

results_df = pd.DataFrame(results)
print("\\nResults Summary:")
print(results_df.to_string(index=False))

results_df.to_csv(SAVE_DIR / 'results_summary.csv', index=False)

print("\\nSaved files in outputs/:")
for f in SAVE_DIR.glob('*'):
    print(f.name)""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('brain_tumour_training.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

