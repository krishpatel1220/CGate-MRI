"""Flask demo server for CGate-MRI -- three pages, as the project brief asks.

  A. Home                  /
  B. MRI analysis          /analyze   upload -> preview -> Analyze -> results
  C. Generated report      /report/<token>  (+ Markdown / JSON downloads)

The server contains no analysis logic. /api/upload and /api/analyze call
`web/inference.py`, which calls the very same cgate functions the research
pipeline used (ingest -> experts -> calibration -> router -> gate -> report),
so the website reports what the project actually computes -- never a mock.

State is an in-memory dict (no database): a result lives until the server
restarts, which is all a demo needs.

Run:
    .venv\\Scripts\\python.exe web\\app.py
    then open http://127.0.0.1:5000/
"""

from __future__ import annotations

import shutil
import sys
import traceback
import uuid
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
# Locate the cgate package wherever this bundle was unpacked: the working
# tree calls it cgate-mri-main/, the shareable zip calls it source/.
for _cand in (WS / "cgate-mri-main" / "src", WS / "source" / "src", WS / "src"):
    if (_cand / "cgate").is_dir():
        sys.path.insert(0, str(_cand))
        break
sys.path.insert(0, str(WS / "web"))

from flask import (  # noqa: E402
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename  # noqa: E402

import inference  # noqa: E402  (web/inference.py, the pipeline adapter)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 400 * 1024 * 1024  # zips of one case are ~50 MB

UPLOADS = WS / "runs" / "uploads"

# token -> {"pid", "table", "preview_url", "result", "report", "md", "json",
#           "vlm_status"}. In-memory on purpose: no database for a demo.
RESULTS: dict[str, dict] = {}
MAX_SESSIONS = 20


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/analyze")
def analyze_page():
    return render_template("analyze.html", msg=request.args.get("msg", ""))


@app.get("/report/<token>")
def report_page(token: str):
    entry = RESULTS.get(token)
    if entry is None or entry.get("report") is None:
        return redirect(url_for("analyze_page", msg="No result for that link -- upload and analyze a scan first."))
    return render_template(
        "report.html",
        token=token,
        pid=entry["pid"],
        accepted=entry["result"]["accepted"],
        confidence=entry["result"]["confidence"],
        threshold=entry["result"]["gate_threshold"],
        vlm_status=entry.get("vlm_status", "not requested"),
        sections_html=_report_sections_html(entry["report"]),
        rejected_html=_rejected_html(entry["report"]),
        ledger=_ledger_rows(entry["report"]),
    )


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.post("/api/upload")
def api_upload():
    """Validate the upload, build the one-case table, return a preview.

    Accepts either ONE file (.zip or a 4-D .nii/.nii.gz) or the FOUR
    .nii/.nii.gz sequences of one case selected together -- the browser
    sends them under the same form field, so both shapes arrive here.
    """
    uploads = [f for f in request.files.getlist("file") if f and f.filename]
    if not uploads:
        return jsonify(
            error="No file chosen. Select a .zip, a 4-D .nii/.nii.gz, or the "
            "four sequence files (.nii/.nii.gz) together."
        ), 400
    if len(uploads) > 8:
        return jsonify(
            error=f"got {len(uploads)} files at once. A case is one .zip or "
            "the four sequences (t1, t1ce, t2, flair)."
        ), 400

    token = uuid.uuid4().hex[:12]
    work = UPLOADS / token
    work.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for f in uploads:
        name = secure_filename(Path(f.filename).name)
        if not name:
            shutil.rmtree(work, ignore_errors=True)
            return jsonify(error=f"could not use the file name {f.filename!r}."), 400
        dest = work / name
        if dest.exists():  # the same file selected twice would be ambiguous
            shutil.rmtree(work, ignore_errors=True)
            return jsonify(
                error=f"{name} was selected twice. Choose each sequence once."
            ), 400
        f.save(dest)
        saved.append(dest)

    try:
        if len(saved) == 1:
            vols, pid = inference.validate_upload(saved[0], work)
        else:
            vols, pid = inference.validate_upload_many(saved, work)
        table = inference.build_case_table(vols, pid, work)
        preview_url = inference.make_preview(table)
    except inference.UploadError as exc:
        shutil.rmtree(work, ignore_errors=True)
        return jsonify(error=str(exc)), 400
    except Exception as exc:  # unexpected: still tell the user something real
        shutil.rmtree(work, ignore_errors=True)
        return jsonify(error=f"could not read the file ({type(exc).__name__}: {exc})"), 400

    while len(RESULTS) >= MAX_SESSIONS:  # oldest session first (dict order)
        RESULTS.pop(next(iter(RESULTS)))
    RESULTS[token] = {
        "pid": pid,
        "table": table,
        "preview_url": preview_url,
        "result": None,
        "report": None,
        "md": None,
        "json": None,
        "vlm_status": None,
    }
    return jsonify(token=token, patient_id=pid, preview_url=preview_url)


@app.post("/api/analyze")
def api_analyze():
    """Run the fitted system on the uploaded case and return the result."""
    payload = request.get_json(silent=True) or {}
    token = str(payload.get("token", ""))
    entry = RESULTS.get(token)
    if entry is None:
        return jsonify(error="This session expired. Upload the scan again."), 400

    try:
        result = inference.analyze(entry["table"], use_vlm=bool(payload.get("use_vlm")))
    except inference.InferenceError as exc:
        return jsonify(error=str(exc)), 500
    except Exception as exc:
        # Full traceback stays on the server console (the browser only ever
        # gets the message below); a demo you can debug beats one that lies.
        traceback.print_exc()
        return jsonify(error=f"analysis failed ({type(exc).__name__}: {exc})"), 500

    # Keep the big/renderable parts server-side; the browser gets the rest.
    report = result.pop("_report")
    entry["md"] = result.pop("report_markdown")
    entry["json"] = result.pop("report_json")
    entry["report"] = report
    entry["result"] = result
    entry["vlm_status"] = result["vlm"]["status"]
    result["report_url"] = url_for("report_page", token=token)
    return jsonify(result)


@app.get("/download/<token>/<kind>")
def download(token: str, kind: str):
    entry = RESULTS.get(token)
    if entry is None or entry.get("md") is None:
        abort(404)
    if kind == "md":
        body, mime, suffix = entry["md"], "text/markdown; charset=utf-8", "md"
    elif kind == "json":
        body, mime, suffix = entry["json"], "application/json", "json"
    else:
        abort(404)
    resp = Response(body, mimetype=mime)
    resp.headers["Content-Disposition"] = f"attachment; filename={entry['pid']}_report.{suffix}"
    return resp


@app.get("/uploads/<path:rel>")
def uploaded_file(rel: str):
    """Serve a preview image (or anything else) from runs/uploads/."""
    target = (UPLOADS / rel).resolve()
    if UPLOADS.resolve() not in target.parents or not target.is_file():
        abort(404)
    return send_file(target)


@app.get("/api/status")
def api_status():
    """Honest availability of the pieces the demo depends on."""
    return jsonify(inference.status())


# ---------------------------------------------------------------------------
# Report rendering (HTML for the report page; the Markdown download comes
# from cgate.reporting.render, i.e. the same source text the ledger lists)
# ---------------------------------------------------------------------------


def _report_sections_html(report) -> str:
    """Sections as sentences with their evidence chips underneath."""
    from markupsafe import escape

    parts: list[str] = []
    for title, sents in report._sections():  # same grouping markdown() uses
        parts.append(f"<h3>{escape(title)}</h3>")
        parts.append('<ul class="sentences">')
        for s in sents:
            chips = "".join(
                '<span class="chip chip-{src}" title="{d}">{src}{label}</span>'.format(
                    src=escape(e.source),
                    d=escape(e.detail or e.source),
                    label=f" &middot; {escape(e.detail)}" if e.detail else "",
                )
                for e in s.evidence
            )
            parts.append(f"<li>{escape(s.text)}<div class='chips'>{chips}</div></li>")
        parts.append("</ul>")
    return "\n".join(parts)


def _rejected_html(report) -> str:
    from markupsafe import escape

    if not report.vlm_rejected:
        return ""
    items = "".join(
        f"<li><s>{escape(p)}</s> <span class='veto'>vetoed: {escape(r)}</span></li>"
        for p, r in report.vlm_rejected
    )
    return f"<h3>Vetted-out VLM phrases</h3><ul class='sentences'>{items}</ul>"


def _ledger_rows(report) -> list[dict]:
    return [
        {
            "i": i,
            "text": s.text,
            "evidence": "; ".join(
                f"{e.source}" + (f" ({e.detail})" if e.detail else "")
                for e in s.evidence
            ),
        }
        for i, s in enumerate(report.sentences, start=1)
    ]


if __name__ == "__main__":
    print("CGate-MRI web demo  ->  http://127.0.0.1:5000/")
    model_state = (
        "found" if inference.SYSTEM_PKL.exists()
        else "MISSING (run tools\\save_system.py)"
    )
    print(f"  model file: {model_state}")
    app.run(host="127.0.0.1", port=5000, threaded=True)
