"""
LLM Report Generator
Tries Ollama (local) first, then falls back to template-based generation.
Install Ollama: https://ollama.com → ollama pull llama3.2-vision
"""
from typing import Optional

import requests
import re

OLLAMA_URL = "http://localhost:11434/api/generate"

REPORT_SYSTEM_PROMPT = """You are an expert neuroradiologist AI assistant. 
Generate a structured clinical radiology report based on the MRI scan classification results provided.
Your report MUST follow this exact format:

FINDINGS:
[2-3 sentences describing what was found in the MRI scan, the tumour location, morphological features, and imaging characteristics based on the tumour type]

IMPRESSION:
[1-2 sentences summarizing the primary diagnosis with clinical significance]

RECOMMENDATION:
[1-2 sentences with clear next steps: further imaging, specialist referral, biopsy, surgery, or routine monitoring]

Rules:
- Be clinically accurate and concise
- Do NOT hallucinate findings not supported by the classification result
- Use formal medical radiological language
- Always mention the tumour class and severity grade in IMPRESSION
- Do not add any text outside the three sections above
"""

# Template-based fallback reports when LLM is unavailable
TEMPLATE_REPORTS = {
    "glioma": {
        "findings": (
            "T1-weighted MRI demonstrates a heterogeneous, infiltrative mass lesion with irregular margins "
            "and surrounding perilesional oedema. Post-contrast imaging reveals heterogeneous enhancement "
            "consistent with neovascularity. Mass effect and midline shift are noted in the dominant hemisphere."
        ),
        "impression": (
            "Imaging findings are consistent with glioma ({severity} grade). "
            "Classifier confidence: {confidence:.1%}. Clinical correlation and histopathological confirmation are essential."
        ),
        "recommendation": {
            "Severe": (
                "Urgent neurosurgical referral is strongly recommended. Stereotactic biopsy or surgical resection "
                "should be planned. Contrast-enhanced MRI with MR spectroscopy and perfusion imaging advised."
            ),
            "Moderate": (
                "Neurology and neuro-oncology consultation recommended. Contrast MRI follow-up within 4–6 weeks. "
                "Consider MR spectroscopy to evaluate metabolite ratios."
            ),
            "Mild": (
                "Close imaging surveillance with repeat MRI in 3 months. Neurology outpatient review advised. "
                "Functional MRI may be considered to map eloquent cortex."
            ),
        },
    },
    "meningioma": {
        "findings": (
            "Extra-axial, well-circumscribed mass with homogeneous signal intensity and a broad dural base. "
            "Post-contrast imaging shows avid, homogeneous enhancement with a dural tail sign. "
            "No significant surrounding oedema; adjacent brain parenchyma appears intact."
        ),
        "impression": (
            "Imaging features are most consistent with meningioma ({severity} grade). "
            "Confidence: {confidence:.1%}. Further characterisation with dedicated MRI protocol recommended."
        ),
        "recommendation": {
            "Severe": (
                "Neurosurgical consultation for resection planning. Pre-operative angiography to assess vascularity. "
                "WHO grading by histopathology essential."
            ),
            "Moderate": (
                "Neurosurgical assessment recommended. Repeat contrast MRI in 6 weeks to assess growth rate."
            ),
            "Mild": (
                "Conservative management with 6-monthly MRI surveillance. Neurological symptom monitoring advised."
            ),
        },
    },
    "pituitary": {
        "findings": (
            "A focal lesion is identified within the sella turcica, consistent with pituitary adenoma. "
            "T1 and T2 signal characteristics are noted. The pituitary stalk demonstrates no deviation "
            "and the optic chiasm appears uninvolved."
        ),
        "impression": (
            "Imaging is consistent with a pituitary tumour ({severity} grade). "
            "Confidence: {confidence:.1%}. Endocrine evaluation is mandatory."
        ),
        "recommendation": {
            "Severe": (
                "Urgent endocrinology and neurosurgery referral. Visual field assessment required. "
                "Trans-sphenoidal surgical resection should be planned."
            ),
            "Moderate": (
                "Endocrinology consultation for hormonal assessment. MRI in 3 months to assess growth."
            ),
            "Mild": (
                "Dynamic pituitary protocol MRI recommended. Baseline hormone panel and annual surveillance."
            ),
        },
    },
    "astrocytoma": {
        "findings": (
            "An infiltrative intra-axial lesion with ill-defined margins is visualised. "
            "T2/FLAIR hyperintensity is present with moderate surrounding oedema. "
            "Enhancement pattern varies with tumour grade; no haemorrhage identified."
        ),
        "impression": (
            "Findings are consistent with astrocytoma ({severity} grade). "
            "Confidence: {confidence:.1%}. Histopathological grading is required to guide management."
        ),
        "recommendation": {
            "Severe": (
                "Immediate neurosurgical referral for surgical planning. IDH mutation and MGMT methylation "
                "analysis recommended. Radiation oncology consultation to follow."
            ),
            "Moderate": (
                "Neuro-oncology referral. MR spectroscopy and perfusion imaging to better characterise grade."
            ),
            "Mild": (
                "3-month MRI follow-up. Neurology outpatient review. Consider IDH status for prognosis."
            ),
        },
    },
    "ependymoma": {
        "findings": (
            "A discrete intra-ventricular or parenchymal mass is identified with T2 heterogeneity. "
            "Calcification and cystic components may be present. Post-contrast enhancement is heterogeneous. "
            "CSF seeding cannot be excluded without spinal imaging."
        ),
        "impression": (
            "Imaging characteristics are consistent with ependymoma ({severity} grade). "
            "Confidence: {confidence:.1%}. Spinal cord MRI should be obtained for staging."
        ),
        "recommendation": {
            "Severe": (
                "Urgent neurosurgical and neuro-oncology referral. Total craniospinal MRI for staging. "
                "Gross total resection is the primary treatment goal."
            ),
            "Moderate": (
                "Neurosurgery and oncology consultation. MRI of the entire neuraxis for leptomeningeal spread assessment."
            ),
            "Mild": (
                "Interval MRI in 3 months. Paediatric neuro-oncology consultation if applicable."
            ),
        },
    },
    "no_tumor": {
        "findings": (
            "MRI of the brain demonstrates normal cortical and subcortical signal characteristics. "
            "No intra-axial or extra-axial mass lesion is identified. Ventricles are of normal size "
            "and configuration. No midline shift, haemorrhage, or abnormal enhancement is detected."
        ),
        "impression": (
            "No intracranial neoplasm detected. Imaging findings are within normal limits. "
            "Confidence: {confidence:.1%}."
        ),
        "recommendation": {
            "Normal": (
                "No further neuroradiological workup required at this time. Routine clinical follow-up "
                "as per referring physician's protocol. Repeat imaging if new neurological symptoms arise."
            ),
        },
    },
}


def _template_report(tumor_class: str, confidence: float, severity: str, patient_age=None, patient_sex=None) -> str:
    """Generate a high-quality template-based report."""
    template = TEMPLATE_REPORTS.get(tumor_class, TEMPLATE_REPORTS["no_tumor"])

    findings = template["findings"]
    impression = template["impression"].format(confidence=confidence, severity=severity)

    # Get recommendation for this severity
    rec_map = template.get("recommendation", {})
    recommendation = rec_map.get(severity, list(rec_map.values())[0] if rec_map else "Clinical follow-up advised.")

    return f"""FINDINGS:
{findings}

IMPRESSION:
{impression}

RECOMMENDATION:
{recommendation}"""


def _try_ollama(prompt: str, model: str = "llama3.2-vision") -> Optional[str]:
    """
    Attempt to call local Ollama LLM. Returns raw text or None on failure.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 400,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception:
        return None


def generate_report(
    tumor_class: str,
    confidence: float,
    severity: str,
    severity_description: str,
    patient_age: int = None,
    patient_sex: str = None,
) -> dict:
    """
    Generate structured radiology report.
    Tries Ollama first, falls back to template generation.
    Returns dict: findings, impression, recommendation, full_report, llm_used
    """
    patient_info = ""
    if patient_age:
        patient_info += f"Patient Age: {patient_age}. "
    if patient_sex:
        patient_info += f"Patient Sex: {patient_sex}. "

    user_prompt = f"""[SYSTEM]: {REPORT_SYSTEM_PROMPT}

[USER]: 
{patient_info}
MRI Classification Result: {tumor_class.replace('_', ' ').title()}
Confidence Score: {confidence:.1%}
Severity Grade: {severity}
Clinical Note: {severity_description}

Based on these classification results, generate the structured radiology report.
"""

    llm_used = "template"
    full_report = None

    # Try Ollama
    full_report = _try_ollama(user_prompt)
    if full_report:
        llm_used = "ollama"
    else:
        # Fall back to template
        full_report = _template_report(tumor_class, confidence, severity, patient_age, patient_sex)
        llm_used = "template"

    sections = parse_report_sections(full_report)
    sections["full_report"] = full_report
    sections["tumor_class"] = tumor_class
    sections["severity"] = severity
    sections["confidence"] = confidence
    sections["llm_used"] = llm_used

    return sections


def parse_report_sections(report_text: str) -> dict:
    """Extract FINDINGS, IMPRESSION, RECOMMENDATION sections from report text."""
    sections = {"findings": "", "impression": "", "recommendation": ""}
    current_section = None

    for line in report_text.split("\n"):
        line_upper = line.strip().upper()
        if "FINDINGS" in line_upper and ":" in line:
            current_section = "findings"
            text = line.split(":", 1)[-1].strip()
            if text:
                sections["findings"] += text + " "
        elif "IMPRESSION" in line_upper and ":" in line:
            current_section = "impression"
            text = line.split(":", 1)[-1].strip()
            if text:
                sections["impression"] += text + " "
        elif "RECOMMENDATION" in line_upper and ":" in line:
            current_section = "recommendation"
            text = line.split(":", 1)[-1].strip()
            if text:
                sections["recommendation"] += text + " "
        elif current_section and line.strip():
            sections[current_section] += line.strip() + " "

    return {k: v.strip() for k, v in sections.items()}
