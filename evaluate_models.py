import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b4, densenet121, resnet50
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# Import the existing preprocessing function
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing.preprocess import preprocess_mri

# Constants
DATA_DIR = Path("/Users/macbookpro/Downloads/FInal Year Project/Project/data")
TEST_DIR = DATA_DIR / "Testing"
MODEL_DIR = Path("/Users/macbookpro/Downloads/FInal Year Project/Project/brain_tumour_project/models")
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary', 'metastasis', 'pediatric_glioma']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}


class BrainTumorEvalDataset(Dataset):
    def __init__(self, data_dir, class_to_idx):
        self.data_dir = Path(data_dir)
        self.class_to_idx = class_to_idx
        self.image_paths = []
        self.labels = []
        
        for cls_name in class_to_idx.keys():
            cls_dir = self.data_dir / cls_name
            if cls_dir.is_dir():
                for img_path in cls_dir.glob("*.[jp][pn]*"):
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            processed_img = preprocess_mri(image_path=img_path)
            processed_img = np.transpose(processed_img, (2, 0, 1))
            tensor_img = torch.tensor(processed_img, dtype=torch.float32)
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor_img = (tensor_img - mean) / std
            
            return tensor_img, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            return torch.zeros((3, 224, 224), dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def load_model(arch, weights_file, num_classes, device):
    weights_path = MODEL_DIR / weights_file
    if not weights_path.exists():
        print(f"⚠️ Warning: Weights file {weights_file} not found. Skipping {arch}.")
        return None
        
    print(f"Loading {arch} weights...")
    if arch == 'efficientnet':
        model = efficientnet_b4(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == 'densenet':
        model = densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif arch == 'resnet':
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not TEST_DIR.exists():
        print(f"Error: Test directory not found at {TEST_DIR}")
        return
        
    test_dataset = BrainTumorEvalDataset(TEST_DIR, CLASS_TO_IDX)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
    print(f"Loaded {len(test_dataset)} testing samples.")
    
    models = {
        'EfficientNet-B4': load_model('efficientnet', 'efficientnet_best.pth', len(CLASSES), device),
        'DenseNet-121': load_model('densenet', 'densenet_best.pth', len(CLASSES), device),
        'ResNet-50': load_model('resnet', 'resnet_best.pth', len(CLASSES), device)
    }
    
    # Filter out models that failed to load
    loaded_models = {name: model for name, model in models.items() if model is not None}
    
    if not loaded_models:
        print("❌ Error: No models loaded successfully. Please train the models first.")
        return
        
    print("\nEvaluating models on testing dataset...")
    
    # Track metrics
    predictions = {name: [] for name in loaded_models.keys()}
    predictions['Ensemble'] = []
    true_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Inference"):
            inputs = inputs.to(device)
            true_labels.extend(labels.numpy())
            
            batch_probs = {name: [] for name in loaded_models.keys()}
            
            # Get logits & convert to probabilities for each model
            for name, model in loaded_models.items():
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                batch_probs[name] = probs
                
                # Single model prediction
                preds = np.argmax(probs, axis=1)
                predictions[name].extend(preds)
                
            # Ensemble predictions (Average Probabilities)
            avg_probs = np.mean([batch_probs[name] for name in loaded_models.keys()], axis=0)
            ensemble_preds = np.argmax(avg_probs, axis=1)
            predictions['Ensemble'].extend(ensemble_preds)
            
    # Calculate and print accuracy
    print("\n" + "="*50)
    print("                 ACCURACY REPORT")
    print("="*50)
    
    for name in predictions.keys():
        acc = accuracy_score(true_labels, predictions[name])
        print(f"🎯 {name:<17} Accuracy: {acc * 100:.2f}%")
        
    print("="*50)
    
    # Detailed Ensemble Classification Report
    print("\nEnsemble Detailed Metrics:")
    print(classification_report(true_labels, predictions['Ensemble'], target_names=CLASSES))


if __name__ == "__main__":
    main()
