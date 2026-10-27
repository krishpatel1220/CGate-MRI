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

## Live Deployment (Free, Always-On)

The demo runs as a **Hugging Face Docker Space** built from this repository:

* **Space:** https://huggingface.co/spaces/krishpatel1220/cgate-mri
* **Demo URL:** https://krishpatel1220-cgate-mri.hf.space (available after the ~10 minute build)
* The 174 MB `runs/system_B6.pkl` is fetched during the image build from the [`v1` release](https://github.com/krishpatel1220/CGate-MRI/releases/tag/v1), because GitHub's per-file limit keeps it out of git.
* **Ship an update:** run `.\.venv\Scripts\python.exe deploy\create_space.py` — it prompts for a Hugging Face *write* token (create one at https://huggingface.co/settings/tokens), copies this project to the Space and pushes; the cloud then rebuilds it.

On the free tier expect CPU-only inference (a case takes a minute or two, versus seconds locally), the optional VLM is not installed there — the report page reports it as `unavailable` rather than faking phrases — and results live in memory, so they reset when the Space restarts.

## Dataset and Model

This project uses the BraTS 2021 brain MRI dataset for research evaluation.

The pipeline combines radiomics features, CNN-based analysis, and Vision Transformer representations. Quality-aware routing combines expert predictions, while confidence gating determines whether cases are released or deferred.

The full dataset, large cached artifacts, and the generated `runs/system_B6.pkl` model file are not included in the lightweight repository distribution. Refer to the project documentation for dataset access and artifact preparation instructions.

## Vision-Language Model Reporting

The optional VLM module generates research-oriented MRI report drafts using structured predictions and available evidence.

Generated phrases are subject to evidence-grounding and contradiction-filtering logic. Report generation is evaluated separately from classification performance.

## Limitations

* Uploaded MRI scans do not include ground-truth labels for individual-case accuracy assessment.
* Results from the research dataset may not generalize to other hospitals, scanners, or patient populations.
* Report-generation quality requires further evaluation against appropriate reference reports and expert assessments.
* High benchmark performance does not establish clinical reliability.

> **Disclaimer:** CGate-MRI is an academic research prototype, not a medical device. It is not intended for clinical diagnosis or treatment. Generated reports must not be treated as verified clinical diagnoses.

## Author

**Krish Patel**

[GitHub Profile](https://github.com/krishpatel1220)
