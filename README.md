# 🧠 NeuroScan AI — Brain Tumour MRI Report Generator


Multi-class brain tumour MRI classification with confidence-grounded severity grading and automated structured radiology report generation.

---

## 📋 System Overview

| Layer       | Technology               |
|-------------|--------------------------|
| Frontend    | React + Vite (Glassmorphism Dark UI) |
| Backend     | FastAPI (Python)          |
| CNN Model   | EfficientNet-B4           |
| LLM         | Ollama (llama3.2-vision) + template fallback |
| Evaluation  | BLEU-1, ROUGE-L, BERTScore |

### Disease Classes (6)
1. Glioma
2. Meningioma
3. Pituitary Tumour
4. No Tumour (Normal)
5. Astrocytoma
6. Ependymoma

---

## 🚀 Quick Start

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI backend
```bash
# From the project root
uvicorn backend.api:app --reload --port 8000
```

The backend runs at: http://localhost:8000  
API docs: http://localhost:8000/docs

### 3. Start the React frontend
```bash
cd frontend
npm install   # only needed once
npm run dev
```

Frontend: http://localhost:5173

---

## 🤖 Enabling Real CNN Inference

By default the app runs in **Demo Mode** (synthetic predictions) when model weights are missing.

To enable real EfficientNet-B4 inference:
1. Train the model on Kaggle using `notebooks/02_cnn_training.ipynb`
2. Download `efficientnet_b4.pth` from Kaggle output
3. Place it in `models/efficientnet_b4.pth`
4. Restart the backend

---

## 🗣️ Enabling Ollama LLM

```bash
# Install from https://ollama.com, then:
ollama pull llama3.2-vision
# or:
ollama pull llama3
```

The backend auto-detects Ollama. If unavailable, falls back to high-quality clinical templates.

---

## 📁 Project Structure

```
brain_tumour_project/
├── frontend/           React + Vite premium UI
├── backend/
│   ├── api.py          FastAPI app
│   ├── classifier.py   EfficientNet-B4 inference
│   ├── severity.py     Severity grading rules
│   ├── report_generator.py  LLM report generation
│   └── evaluator.py    BLEU/ROUGE/BERTScore
├── preprocessing/
│   └── preprocess.py   CLAHE + ROI pipeline
├── models/
│   └── efficientnet_b4.pth  (place trained weights here)
├── prompts/
│   └── report_prompt.txt
└── requirements.txt
```

---

## 🧪 API Endpoints

| Method | Endpoint    | Description |
|--------|-------------|-------------|
| GET    | `/health`   | API + model status |
| POST   | `/predict`  | Upload MRI → classification + report |
| POST   | `/evaluate` | BLEU/ROUGE/BERTScore evaluation |

---

*Generated for Project 22ADP72 | Sooriee | Kongu Engineering College | July 2026*
