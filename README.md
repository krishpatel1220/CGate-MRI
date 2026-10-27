# CGate-MRI

**Confidence-gated multi-modal MRI reporting** — a final-year B.Tech project.
A three-expert ensemble (radiomics + CNN + ViT) predicts MRI findings, a
*quality router* weighs the experts per case, and a *confidence gate* either
releases the case or defers it for review; a grounded generator then writes a
report in which every sentence carries the evidence behind it.

> Research prototype for demonstration purposes. **Not a medical device, not
> for clinical use.**

## What's in this repository

| Path | What it is |
|------|------------|
| [`cgate-mri-main/`](cgate-mri-main/) | The research pipeline: data ingest → feature views → three experts → calibration → quality router → confidence gate → grounded reporting. **Full write-up, method description and results: [`cgate-mri-main/README.md`](cgate-mri-main/README.md)** |
| [`web/`](web/) | The Flask demo site (Home → MRI analysis → Generated report). **Run/test instructions and viva script: [`web/README.md`](web/README.md)** |
| [`tools/`](tools/) | `save_system.py` — refits and verifies the served model, then saves it as `runs/system_B6.pkl` |
| [`runs/`](runs/) | Deliverables: figures (`figs/`), 250 template reports (`reports/`), 60 VLM reports (`reports_vlm/`), `predictions.json` (the exported reference rows), metrics and run logs |
| [`START_HERE.md`](START_HERE.md) | Step-by-step guide for running the bundle on a fresh machine or server |

## Headline results (BraTS 2021, 1,251 cases, seed 0)

- Test accuracy **0.660** (multi-finding, five findings per case)
- Confidence gate releases **85.2%** of cases; threshold fitted at 0.551
- Every generated sentence is grounded — findings, image-quality metrics, the
  gate decision and vetted VLM phrases are tracked as evidence, and phrases
  that contradict the calibrated predictions are vetoed

See [`runs/README_RESULTS.txt`](runs/README_RESULTS.txt) and
[`runs/figs/`](runs/figs/) for the full numbers and plots.

## Quick start (web demo)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e "cgate-mri-main[sci,io,torch]"
.\.venv\Scripts\python.exe -m pip install flask
.\.venv\Scripts\python.exe tools\save_system.py      # one-time: fit + verify the model
.\.venv\Scripts\python.exe web\app.py                # http://127.0.0.1:5000/
.\.venv\Scripts\python.exe web\test_demo.py          # 24 checks vs runs/predictions.json
```

Linux/mac: use `.venv/bin/python` instead. Serving to others: change
`host="127.0.0.1"` to `"0.0.0.0"` in `web/app.py` (details in
[`START_HERE.md`](START_HERE.md)).

## Not included in this repository (and why)

| Thing | Why | How to get it |
|-------|-----|---------------|
| `data/` — BraTS 2021 (12.7 GB) | The BraTS license requires registration and forbids redistribution | Register at [Medical Segmentation Decathlon / BraTS](https://www.synapse.org/#!Synapse:syn25829067) |
| `runs/system_B6.pkl` (174 MB) | Exceeds GitHub's 100 MB file limit | `tools\save_system.py` (≈7 min, needs `runs/table` + `runs/cache` from a full run) |
| `runs/cache/`, `runs/table/` (11 GB) | Derived bulk data | Produced by `cgate run` / ingest |
| `runs/uploads/`, `runs/test_uploads/` | User uploads; fixtures are derived from BraTS | `web\test_demo.py` rebuilds the fixtures automatically when `data/` is present |

## Honest limitations

- Uploaded scans are analysed **without ground truth** (no segmentation is
  uploaded); accuracy numbers exist only for the research cohort.
- Report quality is evaluated structurally (grounding, deferrals, vetoes) —
  BraTS has no reference reports to compare text against.
- The optional VLM (`Qwen/Qwen2-VL-2B-Instruct`) is advisory; its phrases pass
  a veto filter before appearing on a report.
