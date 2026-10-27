# CGate-MRI web demo

A simple, three-page Flask website for demonstrating the project. It runs the
**real fitted model** on an MRI case you upload — there are no precomputed or
mocked results anywhere in it.

| Page | Route | What it shows |
|------|-------|---------------|
| **A. Home** | `/` | Project title, short description, the 5-step pipeline, headline numbers |
| **B. MRI analysis** | `/analyze` | Upload form, preview of the uploaded scan, Analyze button + loading state, then the real prediction, confidence, routing decision and gate status |
| **C. Generated report** | `/report/<token>` | The grounded report (every sentence with its evidence chips), the grounding ledger, optional vetted VLM phrases, and download buttons (`.md` / `.json`) |

## Files (kept few and readable for a viva)

```
web/
  app.py           Flask routes, pages, report HTML rendering  (~250 lines)
  inference.py     adapter: upload -> the existing cgate pipeline -> report
  templates/       base.html, index.html, analyze.html, report.html
  static/          style.css (light/white, blue accents), app.js (plain fetch)
  test_demo.py     end-to-end test against real scans
  README.md        this file
tools/
  save_system.py   one-time: fit + verify + save runs/system_B6.pkl
```

`inference.py` contains **no model logic of its own**. Every step calls the
same functions the research run used (IQMs -> feature views -> slice cache ->
`TrainedSystem.predict` -> gate -> `cgate.reporting.render_report`), so the
website reports exactly what the project computes.

---

## Requirements — what is and is not included

| Needed | Status |
|--------|--------|
| Python environment (`.\.venv`) with torch/CUDA, nibabel, timm, transformers | already built for the research run |
| **Flask** | **new dependency**: `.\.venv\Scripts\python.exe -m pip install flask` (already installed here) |
| **`runs\system_B6.pkl`** (the fitted model) | **not created by the research run** — `cgate run` keeps systems in memory and writes only metrics. Create it once with `tools\save_system.py` (see below). Without it the site loads but Analyze returns a clear "model file not found" message instead of pretending to work. |
| `runs\table` + `runs\cache` (case table + slice cache) | present; `save_system.py` reuses them — **no re-ingest, no retraining of anything else** |
| VLM (optional checkbox) | `Qwen/Qwen2-VL-2B-Instruct` is cached and loads on first use (~5 s, ~4.4 GB VRAM). `google/medgemma-4b-it` is gated on Hugging Face (HTTP 401 without an accepted license + login), so the code falls back to Qwen automatically. Without any VLM the reports are template-only and the page says so. |

### One-time: fit and save the model

```powershell
cd "C:\Users\abc\Documents\Default Project"
.\.venv\Scripts\python.exe tools\save_system.py
```

~7 minutes. It refits **only the B6 row** from the cached table with the same
splits, seeds and label thresholds as the research run, then **verifies itself
against `runs\predictions.json`** before writing the pickle: on this machine it
reported `per_case_findings_agreement=1.0000`, `gate_decision_agreement=1.0000`,
`mean_abs_confidence_delta=0.0000` — i.e. the served system is the evaluated
system. (It re-runs only if `runs\system_B6.pkl` is missing.)

---

## Run (PowerShell)

```powershell
cd "C:\Users\abc\Documents\Default Project"
.\.venv\Scripts\python.exe web\app.py
```

Then open **http://127.0.0.1:5000/** in a browser.

* The server binds `127.0.0.1` only (local demo, not exposed to the network).
* Results are kept in server memory (no database); they disappear on restart.
* After editing any `web\*.py` file, restart `app.py` (the server runs without
  the debug reloader, on purpose).

## Test (PowerShell)

```powershell
# 24 checks: pages, two real scans vs runs\predictions.json, report +
# downloads, three bad-upload error paths, and the four-files-at-once upload
.\.venv\Scripts\python.exe web\test_demo.py

# 29 checks: additionally the VLM phrase pass on BraTS2021_00009
# (a case that yields both kept phrases and a veto)
.\.venv\Scripts\python.exe web\test_demo.py --vlm
```

Fixtures (zips of real cases) are built automatically into `runs\test_uploads\`.

The key check is this one — the live website must reproduce the research
run's exported row for the uploaded scan:

```
[PASS] result matches runs/predictions.json for this case -- conf=0.562 gate=True (max prob delta 0.000194)
[PASS] different scan, different result -- conf 0.562 vs 0.529, gate True vs False
```

---

## 3-minute demonstration script (for the viva)

1. **Home** — walk the 5 pipeline steps; point at the headline numbers
   (1,251 cases, test accuracy 0.660, gate releases 85.2%).
2. **MRI analysis** — upload `runs\test_uploads\BraTS2021_00009.zip`
   (or Ctrl-select the four `.nii.gz` files from any case folder in `data\`
   — multi-select works too) → **Upload**. The four-panel
   mosaic that appears *is* the input the networks read.
3. Tick **"Include VLM phrases"** → **Analyze** (loading message while the
   three experts, routing and gating run). Point at, in order:
   * the confidence bar vs the fitted threshold → **RELEASED / DEFERRED**;
   * the five findings with their real probabilities;
   * the routing table — which expert the quality router trusted, per finding;
   * the VLM phrases and, if one is struck through, the veto rule that
     removed it (e.g. *"phrase denies enhancement while et_present is
     predicted present"*).
4. **View full report** — the grounding ledger: every sentence with the
   `finding:` / `iqm:` / `gate:` / `vlm:` evidence behind it → **Download
   report (.md)**.
5. **Error demo** — upload `runs\test_uploads\not_a_scan.txt` or
   `missing_t1ce.zip` and show the clear error message.
6. **Different scan, different result** — upload
   `runs\test_uploads\BraTS2021_00005.zip` → this one is **DEFERRED**
   (confidence 0.529 < 0.551), proving the output belongs to the scan.

## Supported upload formats

* **`.zip`** containing one case's four co-registered, skull-stripped
  sequences: `*_t1.nii.gz`, `*_t1ce.nii.gz`, `*_t2.nii.gz`, `*_flair.nii.gz`
  (files may sit in a nested folder; an extra `*_seg.nii.gz` is ignored).
* **the four `.nii/.nii.gz` files selected together** — the file dialog allows
  multi-select (`Ctrl`/`Cmd`-click), and the four sequences of one case are
  uploaded in one request; the case id is derived from the file names
  (`BraTS2021_00002_t1.nii.gz` → `BraTS2021_00002`).
* **one 4-D `.nii/.nii.gz`** whose four volumes are the sequences.

Anything else — a single 3-D volume, a missing sequence, a non-MRI file — is
rejected with a message that says what was wrong and what is needed. The model
is a four-sequence fusion model, so one picture cannot be analysed; that is a
property of the system, not of the website.

## Honest limitations (say these if asked)

* **Research prototype.** Not a medical device, not for clinical use — the
  label appears in the footer of every page.
* Uploaded cases are analysed **without ground truth** (no segmentation is
  uploaded); labels/accuracy only exist for the research cohort, where they
  are computed from the BraTS masks.
* Report quality is evaluated structurally (grounding, deferral, vetoes),
  never by text similarity — BraTS has no reference reports.
* The half-resolution zip `BraTS2021_00009_demo.zip` in `runs\test_uploads\`
  exists only because the browser automation used for the UI test cannot
  transfer files larger than 5 MiB; it is a real scan at 120×120×78 and runs
  the identical pipeline, but its numbers differ from the full-resolution
  case in `runs\predictions.json`.
