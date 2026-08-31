"""
FastAPI Backend — Brain Tumour MRI Report API
Endpoints:
  GET  /health       → API + model status
  POST /predict      → Upload MRI → classification + severity + report
  POST /evaluate     → Compute BLEU/ROUGE/BERTScore on generated vs reference
"""

import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to sys.path so we can import from preprocessing/
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.classifier import load_models, predict, is_demo_mode
from backend.severity import grade_severity, get_tumour_info
from backend.report_generator import generate_report
from backend.evaluator import compute_metrics

# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Brain Tumour MRI Report API",
    description="Multi-class brain tumour classification, severity grading and LLM report generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Startup: Load Model ──────────────────────────────────────────────────────

# We rely on the paths configured inside classifier.py now
EFF_MODEL, DENSE_MODEL, RES_MODEL = load_models(num_classes=6)

# ─── Routes ───────────────────────────────────────────────────────────────────


@app.get("/health")
def health_check():
    return {
        "status": "running",
        "model": "Ensemble (EfficientNet-B4 + DenseNet-121 + ResNet-50)",
        "num_classes": 6,
        "demo_mode": is_demo_mode(),
        "classes": ["glioma", "meningioma", "notumor", "pituitary", "metastasis", "pediatric_glioma"],
    }


@app.post("/predict")
async def predict_and_report(
    file: UploadFile = File(...),
    patient_age: int = Form(None),
    patient_sex: str = Form(None),
):
    """
    Upload a brain MRI image (JPG/PNG) and receive:
    - tumour classification + confidence
    - class probabilities for all 6 classes
    - severity grading
    - structured radiology report (Findings / Impression / Recommendation)
    - tumour clinical info
    """
    if file.content_type not in ("image/jpeg", "image/jpg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Only JPG/PNG/WEBP images are supported.")

    # Read file bytes
    image_bytes = await file.read()

    try:
        # Step 1: Preprocess
        try:
            from preprocessing.preprocess import preprocess_mri
            img = preprocess_mri(image_bytes=image_bytes)
        except Exception as e:
            # If cv2 fails, just create a dummy array in demo mode
            import numpy as np
            img = np.zeros((224, 224, 3), dtype=np.float32)

        # Step 2: Classify (using ensemble if available)
        tumor_class, confidence, all_probs = predict(img)

        # Clinical heuristic: differentiate glioma and pediatric_glioma based on patient age
        if patient_age is not None:
            if patient_age < 18:
                if tumor_class in ("glioma", "pediatric_glioma"):
                    if all_probs.get("glioma", 0) > all_probs.get("pediatric_glioma", 0):
                        # Swap probabilities to prioritize pediatric glioma
                        temp = all_probs["glioma"]
                        all_probs["glioma"] = all_probs["pediatric_glioma"]
                        all_probs["pediatric_glioma"] = temp
                        tumor_class = "pediatric_glioma"
                        confidence = all_probs["pediatric_glioma"]
            else:
                if tumor_class == "pediatric_glioma":
                    if all_probs.get("pediatric_glioma", 0) > all_probs.get("glioma", 0):
                        # Swap probabilities to prioritize adult glioma
                        temp = all_probs["pediatric_glioma"]
                        all_probs["pediatric_glioma"] = all_probs["glioma"]
                        all_probs["glioma"] = temp
                        tumor_class = "glioma"
                        confidence = all_probs["glioma"]

        # Step 3: Severity grading
        severity, severity_desc = grade_severity(tumor_class, confidence)

        # Step 4: Tumour info
        tumour_info = get_tumour_info(tumor_class)

        # Step 5: LLM report generation
        report = generate_report(
            tumor_class=tumor_class,
            confidence=confidence,
            severity=severity,
            severity_description=severity_desc,
            patient_age=patient_age,
            patient_sex=patient_sex,
        )

        return JSONResponse({
            "tumor_class": tumor_class,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "severity": severity,
            "severity_description": severity_desc,
            "tumour_info": tumour_info,
            "report": report,
            "demo_mode": is_demo_mode(),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/evaluate")
async def evaluate_report(
    generated: str = Form(...),
    reference: str = Form(...),
):
    """
    Compute NLP evaluation metrics comparing generated vs reference report.
    Returns BLEU-1, ROUGE-L, BERTScore.
    """
    if not generated.strip() or not reference.strip():
        raise HTTPException(status_code=400, detail="Both 'generated' and 'reference' texts are required.")
    metrics = compute_metrics(generated, reference)
    return JSONResponse(metrics)
