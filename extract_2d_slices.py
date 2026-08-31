import os
import glob
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
import random

def normalize_image(image):
    # Normalize image to 0-255
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val > 0:
        image = (image - min_val) / (max_val - min_val)
    image = (image * 255).astype(np.uint8)
    return image

def process_brats_folder(src_folder, dest_train, dest_test, class_name, num_slices=10, split_ratio=0.8):
    patients = [os.path.join(src_folder, d) for d in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, d))]
    
    os.makedirs(os.path.join(dest_train, class_name), exist_ok=True)
    os.makedirs(os.path.join(dest_test, class_name), exist_ok=True)
    
    for patient_dir in tqdm(patients, desc=f"Processing {class_name}"):
        patient_id = os.path.basename(patient_dir)
        
        # Skip if it is not a patient directory (e.g. Training/Validation folders inside Pediotric glioma)
        if patient_id in ("Training", "Validation"):
            continue
            
        # Find T2W and SEG files
        t2w_files = glob.glob(os.path.join(patient_dir, "*-t2w.nii*"))
        seg_files = glob.glob(os.path.join(patient_dir, "*-seg.nii*"))
        
        if not t2w_files or not seg_files:
            continue
            
        t2w_path = t2w_files[0]
        seg_path = seg_files[0]
        
        try:
            # Load NIfTI files
            t2w_img = nib.load(t2w_path).get_fdata()
            seg_img = nib.load(seg_path).get_fdata()
        except Exception as e:
            print(f"Error loading {patient_id}: {e}")
            continue
            
        # Find slices with largest tumor area
        tumor_areas = []
        for i in range(seg_img.shape[2]):
            area = np.sum(seg_img[:, :, i] > 0)
            tumor_areas.append((i, area))
            
        # Sort by area descending and take top N slices
        tumor_areas.sort(key=lambda x: x[1], reverse=True)
        top_slices = [x[0] for x in tumor_areas[:num_slices] if x[1] > 0]
        
        if not top_slices:
            # If no tumor found, just take middle slices
            mid = t2w_img.shape[2] // 2
            top_slices = list(range(mid - num_slices//2, mid + num_slices//2 + 1))
            
        # Decide if this patient goes to train or test
        is_train = random.random() < split_ratio
        dest_dir = dest_train if is_train else dest_test
        class_dest_dir = os.path.join(dest_dir, class_name)
        
        for slice_idx in top_slices:
            if slice_idx >= t2w_img.shape[2]:
                continue
                
            slice_data = t2w_img[:, :, slice_idx]
            slice_data = normalize_image(slice_data)
            
            # Rotate 90 degrees to make it upright (typical for axial slices in NIfTI)
            slice_data = np.rot90(slice_data)
            
            # Resize to 224x224
            slice_data = cv2.resize(slice_data, (224, 224))
            
            # Save as JPG
            out_filename = f"{patient_id}_slice{slice_idx}.jpg"
            out_path = os.path.join(class_dest_dir, out_filename)
            cv2.imwrite(out_path, slice_data)

if __name__ == "__main__":
    random.seed(42)
    
    src_dir = "/Users/macbookpro/Downloads/DATA"
    dest_train = "/Users/macbookpro/Downloads/FInal Year Project/Project/data/Training"
    dest_test = "/Users/macbookpro/Downloads/FInal Year Project/Project/data/Testing"
    
    # Process metastasis
    meta_src = os.path.join(src_dir, "Brain metasis")
    if os.path.exists(meta_src):
        process_brats_folder(meta_src, dest_train, dest_test, "metastasis", num_slices=10)
        
    # Process pediatric glioma (handling nested subfolders)
    ped_src = os.path.join(src_dir, "Pediotric glioma")
    if os.path.exists(ped_src):
        for subfold in ["Training", "Validation"]:
            subfold_path = os.path.join(ped_src, subfold)
            if os.path.exists(subfold_path):
                process_brats_folder(subfold_path, dest_train, dest_test, "pediatric_glioma", num_slices=10)
    
    print("Done!")
