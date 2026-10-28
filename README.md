# CGate-MRI — Confidence-Gated MRI Analysis

A machine learning-based MRI analysis and report-generation research project that combines Radiomics, CNN, and Vision Transformer (ViT) experts. It features quality-aware expert routing, confidence-based deferral, an interactive Flask web interface, and evidence-grounded Vision-Language Model (VLM) reporting.

**Tech Stack:** Python · PyTorch · Radiomics · CNN · Vision Transformer · Flask · Vision-Language Models · Scikit-learn

## Key Features

* **Multi-Expert Analysis:** Combines radiomics, CNN, and ViT experts for MRI analysis.
* **Quality-Aware Routing:** Uses image-quality metrics to guide expert contributions.
* **Confidence Calibration:** Calibrates model confidence to support more reliable decisions.
* **Confidence Gate:** Releases sufficiently confident cases or defers uncertain cases for review.
* **MRI Report Generation:** Generates research-oriented reports using structured model predictions and supporting evidence.
* **Evidence Grounding:** Tracks supporting evidence and filters unsupported or contradictory generated phrases.
* **Web Interface:** Provides an interface for MRI image submission, analysis, and report viewing.
* **Experimental Evaluation:** Includes B0–B6 experiments, performance metrics, evaluation figures, and prediction records.

## Experimental Results

Results recorded for the BraTS 2021 research evaluation, using seed 0:

| Metric                   | Result |
| ------------------------ | -----: |
| Test Accuracy            |  66.0% |
| Confidence Gate Coverage |  85.2% |
| Confidence Threshold     |  0.551 |

The reported evaluation cohort contains 1,251 cases. Results should be interpreted according to the project's evaluation protocol and do not establish clinical performance.

Detailed metrics and visualizations are available in [`runs/README_RESULTS.txt`](runs/README_RESULTS.txt) and [`runs/figs/`](runs/figs/).

## Installation and Usage

**1. Clone the repository**

```bash
git clone https://github.com/krishpatel1220/CGate-MRI.git
cd CGate-MRI
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Install dependencies**

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\cgate-mri-main[sci,io,torch]"
.\.venv\Scripts\python.exe -m pip install flask
```

**4. Prepare the model**

```powershell
.\.venv\Scripts\python.exe tools\save_system.py
```

This step requires supporting artifacts, including `runs/table` and `runs/cache`, as described in the project documentation.

**5. Launch the web application**

```powershell
.\.venv\Scripts\python.exe web\app.py
```

Open `http://127.0.0.1:5000/` in your browser.

**6. Run the demonstration tests**

```powershell
.\.venv\Scripts\python.exe web\test_demo.py
```

See [`START_HERE.md`](START_HERE.md) for additional setup instructions and requirements.


## Limitations

* Uploaded MRI scans do not include ground-truth labels for individual-case accuracy assessment.
* Results from the research dataset may not generalize to other hospitals, scanners, or patient populations.
* Report-generation quality requires further evaluation against appropriate reference reports and expert assessments.
* High benchmark performance does not establish clinical reliability.

> **Disclaimer:** CGate-MRI is an academic research prototype, not a medical device. It is not intended for clinical diagnosis or treatment. Generated reports must not be treated as verified clinical diagnoses.

## Author

**Krish Patel**

[GitHub Profile](https://github.com/krishpatel1220)
