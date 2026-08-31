# NeuroScan AI: Multi-Class Brain Tumour Classification & LLM Report Generation System

Welcome to the comprehensive system documentation for **NeuroScan AI**, a state-of-the-art medical imaging and clinical reporting assistant. This software architecture was designed as a Final Year Project to classify brain tumours from MRI scans, grade their clinical severity, and generate structured radiology reports with a zero-hallucination guarantee.

---

## 📖 Project Overview

NeuroScan AI is a clinical decision support system that bridges the gap between raw medical imaging and clinical documentation. The project addresses three main challenges:
1. **Accurate Diagnosis:** Multi-class classification across 6 distinct tumour and non-tumour states.
2. **Clinical Urgency Grading:** Mapping quantitative model confidence to actionable severity categories.
3. **Automated Documentation:** Translating classification outputs into high-fidelity, structured clinical reports.

### 🌟 Technical Highlights (Individuality Features)
*   **3-Model CNN Ensemble:** Combines **EfficientNet-B4** (compound scaling), **DenseNet-121** (dense feature reuse), and **ResNet-50** (residual skip connections) to achieve higher accuracy and variance reduction.
*   **Massive Dataset Shrinkage:** Custom NIfTI preprocessing pipeline that compressed a 200GB BraTS 3D MRI dataset into a compact 2D axial slice library by extracting only the most tumour-informative slices using segmentation masks.
*   **Hard-Constrained LLM Report Generation:** Integrates local LLM (Ollama) generation which uses the classification outputs as structural prompts, entirely eliminating hallucinated labels.

---

## 🛠️ Architecture & Pipeline

```text
  [Raw MRI Upload] --> [CLAHE & ROI Preprocessing]
                                |
                    [3-Model CNN Ensemble]
            (EfficientNet-B4 + DenseNet-121 + ResNet-50)
                                |
             [Averaging Ensemble Predictions Block]
                                |
                  [Predicted Class & Confidence]
                   /                         \
    [Severity Grading Rules]          [Hard-Grounded Prompts]
                   \                         /
              [Structured LLM Report Generator (Ollama)]
                                |
                   [React Frontend Dashboard]
```

### 1. Data Preprocessing Pipeline (`preprocessing/preprocess.py`)
To isolate the region of interest (ROI) and normalize scanning discrepancies:
1. **Grayscale Conversion & Normalization:** Converts 3-channel input to single-channel floating-point representation.
2. **Bilateral Filtering:** Smooths noise while preserving the sharp edges of anatomical structures.
3. **Otsu's Thresholding & Contour Detection:** Finds the extreme boundaries of the brain shell and crops out black empty space (padding).
4. **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Enhances contrast in localized regions to bring out low-contrast tumour boundaries.
5. **Standardization:** Resizes to 224 x 224 x 3 and applies ImageNet mean/std normalization.

### 2. The 3-Model Ensemble (`backend/classifier.py`)
The ensemble model takes the final preprocessed image and runs it through three models:
*   **EfficientNet-B4:** Captures global semantic features using optimized depth, width, and resolution scaling.
*   **DenseNet-121:** Explores detailed pattern features by connecting every layer directly to all subsequent layers.
*   **ResNet-50:** Uses identity shortcuts to prevent vanishing gradient problems, capturing high-frequency spatial features.

Averaged Probability Formula:
```text
P_final(class) = [ P_eff(class) + P_dense(class) + P_res(class) ] / 3
```

If any model's weights (`.pth`) are missing, the system gracefully falls back to using only the available trained model(s) or activates **Demo Mode** with simulated outputs.

### 3. Severity Grading (`backend/severity.py`)
Applies deterministic, clinical rule-based mapping to determine urgency. It grades prediction confidences into four severity levels:
*   **Severe:** Rapid specialist referral.
*   **Moderate:** Further diagnostic scan workup.
*   **Mild:** Observational monitoring.
*   **Normal:** Routine follow-up.

### 4. Grounded Report Generation (`backend/report_generator.py`)
To prevent the LLM from hallucinating diagnostic findings:
*   The LLM's prompt is strictly limited to the classified tumour class and severity.
*   The system injects clinical details (common locations, WHO grades, description) directly from the vetted database (`backend/severity.py`).
*   Produces a structured markdown report containing:
    1. **Clinical Details** (Patient demographic baseline).
    2. **Findings** (Detailed radiological observations).
    3. **Impression** (Final clinical summary).
    4. **Recommendations** (Standard next-step protocols).

---

## 📂 Project Directory Structure

```text
brain_tumour_project/
├── backend/
│   ├── api.py                 # FastAPI endpoints (health, predict, evaluate)
│   ├── classifier.py          # 3-Model Ensemble loader and inference
│   ├── severity.py            # Clinical rules, metadata, and severity grading
│   ├── report_generator.py    # Grounded prompts & radiology reports
│   └── evaluator.py           # NLP evaluation metrics (BLEU, ROUGE, BERTScore)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ResultCard.jsx      # Classification & severity display
│   │   │   └── ProbabilityChart.jsx# Visual bar charts of probabilities
│   │   ├── pages/
│   │   │   ├── Home.jsx            # Main dashboard and statistics
│   │   │   └── About.jsx           # Technical pipeline & project info
│   │   └── App.jsx                 # Routing and layout
├── models/
│   ├── class_names.json       # JSON map of target classes
│   ├── efficientnet_best.pth  # Trained EfficientNet weights
│   ├── densenet_best.pth      # Trained DenseNet weights
│   └── resnet_best.pth        # Trained ResNet weights
├── preprocessing/
│   └── preprocess.py          # CLAHE, ROI cropping, normalization
├── extract_2d_slices.py       # NIfTI 3D to 2D JPEG extraction script
├── train_efficientnet.py      # EfficientNet-B4 training pipeline
├── train_densenet.py          # DenseNet-121 training pipeline
├── train_resnet.py            # ResNet-50 training pipeline
├── requirements.txt           # Python packages
└── start.sh                   # Orchestration bash script
```

---

## 📊 Dataset Specifications
The system classifies images into **6 distinct categories**:
1.  `glioma`: Adult low/high-grade glial tumours.
2.  `meningioma`: Extra-axial dural-based tumours.
3.  `pituitary`: Sellar region adenomas.
4.  `metastasis`: Secondary metastatic lesions (shrunk from BraTS NIfTI).
5.  `pediatric_glioma`: Pediatric gliomas (shrunk from BraTS NIfTI).
6.  `notumor`: Normal MRI brain scans with no abnormalities.

---

## ⚡ Execution Instructions

### Running the Whole System (Recommended)
You can start both the FastAPI backend and the React frontend using the provided startup script:
```bash
cd "/Users/macbookpro/Downloads/FInal Year Project/Project/brain_tumour_project"
bash start.sh
```

### Individual Service Control

#### Starting the FastAPI Backend:
```bash
python3 -m uvicorn backend.api:app --reload --port 8000
```
*   API running at: `http://localhost:8000`
*   Interactive documentation: `http://localhost:8000/docs`

#### Starting the React Frontend:
```bash
cd frontend
npm install
npm run dev
```
*   Web App running at: `http://localhost:5173`

---

## 📈 Evaluation & Research Validation
The pipeline evaluates LLM generated reports against reference reports using three metrics:
1.  **BLEU-1:** Measures exact word-matching precision.
2.  **ROUGE-L:** Evaluates the longest common subsequence recall (for sentence structural cohesion).
3.  **BERTScore:** Computes contextual token embedding similarity to ensure clinical semantic alignment.
