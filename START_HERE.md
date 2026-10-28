# CGate-MRI — start here (fresh machine / different server)

This bundle contains the **whole project**: the research pipeline (`source/`),
the Flask website (`web/`), the results and sample reports, and the scripts.

```
source/    the cgate package: ingest -> experts -> router -> gate -> reporting
web/       the Flask site (app.py, inference.py, templates/, static/)
tools/     save_system.py (fit + save the model), run scripts
results/   metrics, figures, predictions.json (the reference rows)
reports/   sample generated reports (template + VLM)
runs/      model file + test fixtures   [full bundle only — see below]
```

There is **no database, no server config, no fake data**: one Python process,
one in-memory session dict.

---

## 1. Requirements

| Needed | Notes |
|--------|-------|
| Python **3.10+** (3.11 used here) | `python --version` |
| ~4 GB disk for the virtualenv | torch is the big one |
| GPU | optional — CPU works, slower; needed only for the optional VLM. The shipped model file loads fine without a GPU. |
| The BraTS dataset | **not needed to run or test** — only to retrain |

## 2. Install (PowerShell on Windows, bash on Linux/mac)

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e "source[sci,io,torch]"
.\.venv\Scripts\python.exe -m pip install flask
```

```bash
# Linux / mac
python3 -m venv .venv
. .venv/bin/activate
pip install -e "source[sci,io,torch]"
pip install flask
```

## 3. The model file (the one thing a source-only zip cannot carry)

Put the fitted model at **`runs/system_B6.pkl`** (~174 MB).

* In the **full bundle** (`cgate-mri-full-demo.zip`) it is already there.
* In the **light zip** (`cgate-mri-results.zip`) it is deliberately excluded;
  ask whoever sent you this for `runs/system_B6.pkl`.
* It **cannot be regenerated from this zip**: `tools/save_system.py` refits
  from `runs/table` + `runs/cache` (8.8 GB, not shipped), and building those
  from scratch needs the BraTS dataset and a full `cgate run` (~5 hours on the
  original machine).
* Without it the site still starts and every page loads, but Analyze answers
  with an honest error (`model file runs/system_B6.pkl not found`) — the demo
  never invents a prediction.

## 4. Run

```powershell
.\.venv\Scripts\python.exe web\app.py      # Windows
. .venv/bin/activate && python web/app.py   # Linux / mac
```

Open **http://127.0.0.1:5000/** — Home → MRI analysis (upload a case from
`runs/test_uploads/`, e.g. `BraTS2021_00009_demo.zip`) → Generated report.

The report page also shows a **Tumor Localization** panel below the grounding
ledger: upload `runs/test_uploads/BraTS2021_00005.zip` (it carries the
ground-truth `_seg.nii.gz`) to see the uploaded T1ce slice next to the same
slice with the mask overlaid. Uploads without a mask get an honest
"no segmentation" message instead of a faked region.

### On a remote server (so others can open it)

`web/app.py` binds `127.0.0.1` on purpose (local demo). To serve it to
others, either:

* change the last line of `web/app.py` to `app.run(host="0.0.0.0", port=5000, threaded=True)`
  and allow port 5000 in the firewall, **or**
* run behind gunicorn (recommended on Linux; keep **1 worker** — results and
  the loaded model live in process memory):

```bash
pip install gunicorn
cd web && gunicorn -w 1 --threads 4 -b 0.0.0.0:5000 app:app
```

If port 5000 is busy, change `port=5000` in `web/app.py`.

## 5. Test (proves the live site reproduces the research numbers)

```powershell
.\.venv\Scripts\python.exe web\test_demo.py          # 29 checks
.\.venv\Scripts\python.exe web\test_demo.py --vlm    # 34 checks (needs GPU + transformers)
```

The key line — the website's answer for an uploaded scan must match the
exported row in `runs/predictions.json`:

```
[PASS] result matches runs/predictions.json for this case -- conf=0.562 gate=True (max prob delta 0.000194)
```

Fixtures live in `runs/test_uploads/` (included in the full bundle). In the
light zip they are rebuilt automatically from `data/` if you have BraTS.

## 6. Optional: VLM phrases on the report

```powershell
.\.venv\Scripts\python.exe -m pip install transformers
```

The report page's checkbox then loads `Qwen/Qwen2-VL-2B-Instruct`
(~5 GB download, ~4.4 GB VRAM, GPU strongly recommended). Without it the
reports are template-only and the page says so — nothing is faked.

## 7. Implementing it differently

The split is deliberate, so you can rewrite the presentation without touching
the science:

* **`source/` has zero web code** — it is the pipeline the paper describes
  (ingest → radiomics/CNN/ViT experts → calibration → quality router →
  confidence gate → grounded reporting). Reuse it as a library:
  `pip install -e "source[sci,io,torch]"`, then `from cgate... import ...`.
* **`web/` is only a thin shell** (~500 lines total): `app.py` = routes,
  `inference.py` = the adapter that calls the same cgate functions the
  training run used. Rewrite `web/` in your own framework (Django, FastAPI,
  plain CGI, a different template design…) and keep calling
  `inference.validate_upload / build_case_table / analyze` — those four
  functions are the entire contract with the pipeline.
* Changing the *model* itself means editing `source/cgate/...` and retraining
  with `cgate run` on BraTS (dataset registration required; it cannot be
  redistributed with this bundle).

## 8.limitations 

Research prototype for a final-year project — **not a medical device, not for
clinical use**. Uploaded cases are analysed without ground truth; accuracy
numbers exist only for the research cohort, where BraTS segmentations are
available.
