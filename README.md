````markdown
# CGate-MRI

**Confidence-Gated Multimodal MRI Analysis and Report Generation**

CGate-MRI is a final-year B.Tech research project exploring quality-aware expert routing, confidence-based deferral, and evidence-grounded MRI report generation.

The research pipeline combines three expert approaches: radiomics, a convolutional neural network (CNN), and a vision transformer (ViT). A quality router combines their predictions, while a confidence gate determines whether a case should be released or deferred for further review. A grounded reporting module generates research-oriented reports using structured predictions and supporting evidence.

> **Research prototype only. Not a medical device. Not for clinical use.** Generated reports are not verified medical diagnoses.

## Key Features

- **Three-expert ensemble:** Combines radiomics, CNN, and ViT approaches.
- **Quality-aware routing:** Uses image-quality information to guide expert contributions.
- **Confidence calibration:** Calibrates model confidence scores.
- **Confidence gating:** Releases or defers cases according to a confidence threshold.
- **Evidence-grounded reporting:** Tracks supporting evidence and filters unsupported or contradictory generated phrases.
- **Web demonstration:** Provides an interface for MRI image submission, analysis, and report generation.
- **Experimental evaluation:** Includes metrics, figures, predictions, and experiment records.

## Repository Structure

| Path | Description |
|---|---|
| `cgate-mri-main/` | Core research pipeline, implementation, methodology, and documentation |
| `web/` | Flask-based demonstration website |
| `tools/` | Utilities for model preparation and verification |
| `runs/` | Available experimental results, figures, reports, predictions, and logs |
| `START_HERE.md` | Setup and execution guide |

See [`cgate-mri-main/README.md`](cgate-mri-main/README.md) for the methodology and detailed experimental discussion.

For web setup and testing, see [`web/README.md`](web/README.md).

## Headline Experimental Results

**Dataset:** BraTS 2021  
**Reported evaluation cohort:** 1,251 cases  
**Experiment:** Seed 0

| Metric | Result |
|---|---:|
| Test accuracy | 0.660 (66.0%) |
| Gate coverage | 85.2% |
| Confidence threshold | 0.551 |

These values describe the saved research experiment and should be interpreted in the context of its evaluation protocol. They do not establish clinical performance or diagnostic reliability.

For additional metrics and visualizations, see [`runs/README_RESULTS.txt`](runs/README_RESULTS.txt) and [`runs/figs/`](runs/figs/).

## Technology Stack

- Python
- Flask
- NumPy and scientific Python libraries
- PyTorch
- Radiomics and machine-learning components
- Vision-language model integration
- Automated testing and evaluation utilities

Refer to `pyproject.toml` and the project documentation for the exact dependencies and supported configuration.

## Quick Start: Web Demonstration

The following commands assume that the repository has been cloned and Python is installed.

### 1. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
````

### 2. Install the research package

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\cgate-mri-main[sci,io,torch]"
```

### 3. Install Flask

```powershell
.\.venv\Scripts\python.exe -m pip install flask
```

### 4. Prepare the model

```powershell
.\.venv\Scripts\python.exe tools\save_system.py
```

This step requires the input artifacts documented by the project, including `runs/table` and `runs/cache`. These artifacts are not included in the lightweight repository distribution.

### 5. Start the web application

```powershell
.\.venv\Scripts\python.exe web\app.py
```

Open the following address in your browser:

`http://127.0.0.1:5000/`

### 6. Run the demonstration tests

```powershell
.\.venv\Scripts\python.exe web\test_demo.py
```

Successful execution depends on the required model artifacts and test fixtures being available.

For complete setup instructions, including Linux/macOS guidance, see [`START_HERE.md`](START_HERE.md).

## Dataset and Model Artifacts

Some large or restricted assets are intentionally excluded from this repository.

| Asset                           | Reason for exclusion                                                | Requirements                                                               |
| ------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| BraTS 2021 dataset              | Large dataset with access and licensing requirements                | Obtain access through the official dataset distribution process            |
| `runs/system_B6.pkl`            | Approximately 174 MB, exceeding GitHub's standard 100 MB file limit | Prepare it using the documented workflow and required input artifacts      |
| `runs/cache/` and `runs/table/` | Large derived artifacts                                             | Generate them using the documented data-processing and experiment pipeline |
| User-uploaded MRI images        | Potentially sensitive data                                          | Do not commit personal or identifiable medical images                      |

**Reproducibility note:** The lightweight repository does not contain every dataset, cached feature, or trained-model artifact required to reproduce the complete research pipeline. Follow the project setup documentation to identify and prepare the required assets.

## Vision-Language Model Reporting

The optional VLM component supports research-oriented report drafting. Generated phrases are subject to the project's evidence and contradiction-filtering logic.

Report generation should not be interpreted as independent clinical verification. The quality of generated text, evidence grounding, and deferral behavior must be evaluated separately from classification accuracy.

## Limitations

* Uploaded scans are processed without ground-truth labels; therefore, an individual uploaded case cannot be assigned a verified accuracy score.
* Report-generation evaluation focuses on structural properties such as evidence grounding, deferrals, and phrase filtering.
* The BraTS research dataset does not provide reference radiology reports for direct evaluation of report-text accuracy.
* Performance on the research cohort does not establish generalization to other hospitals, scanners, populations, or clinical settings.
* The system requires additional validation before any consideration of clinical use.

## Future Work

Potential extensions include broader external validation, improved report evaluation, reproducible model packaging, and further investigation of quality-aware routing and confidence-based deferral.

## Disclaimer

CGate-MRI is an academic research and demonstration project. It is not intended to diagnose, treat, or prevent disease, and it must not be used to make clinical decisions. All generated reports require appropriate expert review.

```
```
