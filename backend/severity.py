"""
Severity Grading Module
Rules based on clinical literature:
- Confidence score + tumour type → severity level
- Maps to Mild / Moderate / Severe / Normal
"""
from typing import Tuple

SEVERITY_RULES = {
    "glioma": {
        "Severe":   (0.85, 1.01),   # High confidence glioma = likely high-grade
        "Moderate": (0.60, 0.85),
        "Mild":     (0.00, 0.60),
    },
    "meningioma": {
        "Severe":   (0.85, 1.01),
        "Moderate": (0.60, 0.85),
        "Mild":     (0.00, 0.60),
    },
    "pituitary": {
        "Severe":   (0.80, 1.01),
        "Moderate": (0.55, 0.80),
        "Mild":     (0.00, 0.55),
    },
    "astrocytoma": {
        "Severe":   (0.85, 1.01),
        "Moderate": (0.60, 0.85),
        "Mild":     (0.00, 0.60),
    },
    "ependymoma": {
        "Severe":   (0.80, 1.01),
        "Moderate": (0.55, 0.80),
        "Mild":     (0.00, 0.55),
    },
    "no_tumor": {
        "Normal":   (0.00, 1.01),
    },
}

SEVERITY_DESCRIPTIONS = {
    "Severe":   "High-grade tumour with significant clinical concern. Immediate specialist referral required.",
    "Moderate": "Moderate-grade findings. Further diagnostic workup recommended.",
    "Mild":     "Low-grade or early-stage findings. Monitoring and follow-up advised.",
    "Normal":   "No tumour detected. Routine follow-up as per clinical protocol.",
}

SEVERITY_COLORS = {
    "Severe":   "#ef4444",   # red
    "Moderate": "#f59e0b",   # amber
    "Mild":     "#22c55e",   # green
    "Normal":   "#3b82f6",   # blue
}

TUMOUR_INFO = {
    "glioma": {
        "full_name": "Glioma",
        "description": "A tumour arising from glial cells in the brain or spine. Can be low-grade or high-grade (glioblastoma).",
        "who_grade": "WHO Grade II–IV",
        "common_location": "Cerebral hemispheres",
    },
    "meningioma": {
        "full_name": "Meningioma",
        "description": "A tumour arising from the meninges (protective layers of the brain and spinal cord). Usually benign.",
        "who_grade": "WHO Grade I–III",
        "common_location": "Brain surface / skull base",
    },
    "pituitary": {
        "full_name": "Pituitary Tumour",
        "description": "Adenoma of the pituitary gland. Most are benign and can affect hormone regulation.",
        "who_grade": "WHO Grade I",
        "common_location": "Sella turcica (pituitary fossa)",
    },
    "astrocytoma": {
        "full_name": "Astrocytoma",
        "description": "A tumour of astrocytes (star-shaped glial cells). Ranges from slow-growing to highly aggressive.",
        "who_grade": "WHO Grade I–IV",
        "common_location": "Cerebral hemispheres, cerebellum",
    },
    "ependymoma": {
        "full_name": "Ependymoma",
        "description": "Arising from ependymal cells lining ventricles. More common in children.",
        "who_grade": "WHO Grade II–III",
        "common_location": "4th ventricle / spinal cord",
    },
    "no_tumor": {
        "full_name": "No Tumour Detected",
        "description": "Normal MRI findings. No evidence of intracranial neoplasm.",
        "who_grade": "N/A",
        "common_location": "N/A",
    },
}


def grade_severity(tumor_class: str, confidence: float) -> Tuple[str, str]:
    """
    Returns (severity_level, severity_description).
    """
    rules = SEVERITY_RULES.get(tumor_class, {})
    for level, (low, high) in rules.items():
        if low <= confidence < high:
            return level, SEVERITY_DESCRIPTIONS[level]
    return "Unknown", "Unable to determine severity."


def get_tumour_info(tumor_class: str) -> dict:
    """Returns clinical info dict for a given tumour class."""
    return TUMOUR_INFO.get(tumor_class, {})
