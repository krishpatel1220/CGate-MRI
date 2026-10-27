"""End-to-end test of the CGate-MRI web demo against real uploaded scans.

Run the server first, then:

    .venv\\Scripts\\python.exe web\\test_demo.py            # without VLM
    .venv\\Scripts\\python.exe web\\test_demo.py --vlm      # also test Qwen phrases

What it checks (nothing is mocked):

  1. the pages load (/, /analyze)
  2. uploading BraTS201000002.zip (four sequences, NO seg) and analyzing it
     reproduces that case's row in runs/predictions.json -- probabilities,
     confidence and gate decision -- which is what proves the result belongs
     to the uploaded scan and not to some fixed output
  3. a different scan (BraTS2021_00005) gives a different, equally-matching
     result (this one the gate defers)
  4. the report page and the .md / .json downloads work
  5. bad uploads fail with clear messages: non-zip file, missing sequence,
     single 3-D volume
  6. the same case uploaded as FOUR separate .nii.gz files (no zip) gives
     the same reference-matching result
  7. with --vlm: the VLM phrase pass runs (or reports itself unavailable)
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "cgate-mri-main" / "src"))

BASE = "http://127.0.0.1:5000"
DATA = WS / "data"
PREDICTIONS = WS / "runs" / "predictions.json"
FIXTURES = WS / "runs" / "test_uploads"

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only: the venv has no `requests`)
# ---------------------------------------------------------------------------


def get(path: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def post_json(path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def post_files(path: str, field: str, file_paths: list[Path]) -> tuple[int, dict]:
    """POST several files under the SAME form field (what the page does)."""
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for p in file_paths:
        parts += [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{field}"; '
            f'filename="{p.name}"\r\n'.encode(),
            b"Content-Type: application/octet-stream\r\n\r\n",
            p.read_bytes(),
            b"\r\n",
        ]
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        BASE + path,
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def post_file(path: str, field: str, file_path: Path) -> tuple[int, dict]:
    return post_files(path, field, [file_path])


def wait_for_server(timeout_s: int = 60) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            status, _ = get("/")
            if status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def build_fixtures() -> None:
    """Zip two real cases (no code involved: pure file operations)."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for pid, include_seg in (
        ("BraTS2021_00002", False),
        ("BraTS2021_00005", True),
        ("BraTS2021_00009", False),  # demo case: yields VLM phrases + a veto
    ):
        dest = FIXTURES / f"{pid}.zip"
        if dest.exists():
            continue
        src = DATA / pid
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as zf:
            for f in sorted(src.glob("*.nii.gz")):
                is_seg = f.name.endswith("_seg.nii.gz")
                if is_seg and not include_seg:
                    continue
                # 00005 is zipped as a folder (like a right-click "compress")
                arc = f"{pid}/{f.name}" if include_seg else f.name
                zf.write(f, arcname=arc)
        print(f"  built {dest.relative_to(WS)}")
    # single 3-D file + non-zip + missing sequence, for the error paths
    solo = FIXTURES / "solo_flair.nii.gz"
    if not solo.exists():
        solo.write_bytes((DATA / "BraTS2021_00002" / "BraTS2021_00002_flair.nii.gz").read_bytes())
    txt = FIXTURES / "not_a_scan.txt"
    if not txt.exists():
        txt.write_text("hello - this is not an MRI", encoding="utf8")
    bad = FIXTURES / "missing_t1ce.zip"
    if not bad.exists():
        with zipfile.ZipFile(bad, "w", zipfile.ZIP_STORED) as zf:
            for mod in ("t1", "t2", "flair"):
                f = DATA / "BraTS2021_00006" / f"BraTS2021_00006_{mod}.nii.gz"
                zf.write(f, arcname=f.name)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def upload_and_analyze(name: str, use_vlm: bool) -> tuple[dict | None, str]:
    status, up = post_file("/api/upload", "file", FIXTURES / f"{name}.zip")
    if status != 200:
        return None, f"upload failed: HTTP {status} {up.get('error', '')}"
    status, res = post_json("/api/analyze", {"token": up["token"], "use_vlm": use_vlm})
    if status != 200:
        return None, f"analyze failed: HTTP {status} {res.get('error', '')}"
    res["_token"] = up["token"]
    res["_preview"] = up["preview_url"]
    return res, ""


def matches_reference(result: dict) -> tuple[bool, str]:
    """Compare a live analysis against the research run's exported row."""
    from cgate.schema import FINDING_SCHEMA

    ref = {
        c["patient_id"]: c
        for c in json.loads(PREDICTIONS.read_text(encoding="utf8"))["cases"]
    }[result["patient_id"]]

    problems = []
    if result["accepted"] != ref["accepted"]:
        problems.append(f"gate {result['accepted']} != reference {ref['accepted']}")
    if abs(result["confidence"] - ref["confidence"]) > 1e-3:
        problems.append(
            f"confidence {result['confidence']:.4f} != reference {ref['confidence']:.4f}"
        )
    worst = 0.0
    for j, spec in enumerate(FINDING_SCHEMA):
        live = [result["findings"][j]["probs"][c] for c in spec.classes]
        worst = max(worst, max(abs(a - b) for a, b in zip(live, ref["probs"][j])))
    if worst > 1e-3:
        problems.append(f"probabilities differ by up to {worst:.4f}")
    return not problems, "; ".join(problems) or (
        f"conf={result['confidence']:.3f} gate={result['accepted']} "
        f"(max prob delta {worst:.6f})"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", action="store_true", help="also test the VLM phrase pass")
    args = ap.parse_args()

    print("Waiting for the server ...")
    if not wait_for_server():
        print("Server did not come up. Start it first: .venv\\Scripts\\python.exe web\\app.py")
        return 2
    print(f"  {BASE} is up\n")

    print("Fixtures")
    build_fixtures()

    print("\n1. Pages")
    status, body = get("/")
    check("home page loads", status == 200 and b"CGate-MRI" in body)
    status, body = get("/analyze")
    check("analysis page loads", status == 200 and b"Upload an MRI case" in body)
    status, body = get("/api/status")
    st = json.loads(body) if status == 200 else {}
    check("fitted model present", st.get("system_saved") is True, str(st.get("system_path")))

    print("\n2. Real scan #1: BraTS2021_00002 (flat zip, four sequences, no seg)")
    r1, err = upload_and_analyze("BraTS2021_00002", use_vlm=False)
    if not check("upload + analyze succeeded", r1 is not None, err):
        return finish()
    check("preview image rendered", r1["_preview"].startswith("/uploads/"))
    ok, detail = matches_reference(r1)
    check("result matches runs/predictions.json for this case", ok, detail)
    check(
        "report link present",
        bool(r1.get("report_url")) and r1["report_url"].startswith("/report/"),
        r1.get("report_url", ""),
    )

    print("\n3. Real scan #2: BraTS2021_00005 (zipped as a folder, with seg)")
    r2, err = upload_and_analyze("BraTS2021_00005", use_vlm=False)
    if not check("upload + analyze succeeded", r2 is not None, err):
        return finish()
    ok, detail = matches_reference(r2)
    check("result matches runs/predictions.json for this case", ok, detail)
    check(
        "different scan, different result",
        r2["patient_id"] != r1["patient_id"]
        and abs(r2["confidence"] - r1["confidence"]) > 1e-6,
        f"conf {r1['confidence']:.3f} vs {r2['confidence']:.3f}, "
        f"gate {r1['accepted']} vs {r2['accepted']}",
    )

    print("\n4. Report page and downloads")
    token = r1["_token"]
    status, body = get(f"/report/{token}")
    html = body.decode("utf8", "replace")
    check("report page loads", status == 200)
    check("report shows gate status", "RELEASED" in html or "DEFERRED" in html)
    check("grounding ledger present", "Grounding ledger" in html)
    status, md = get(f"/download/{token}/md")
    text = md.decode("utf8", "replace")
    check(
        "markdown download",
        status == 200 and text.startswith("# Automated MRI report"),
        text.splitlines()[0] if text else f"HTTP {status}",
    )
    status, raw = get(f"/download/{token}/json")
    try:
        payload = json.loads(raw)
        check("json download with sentences", status == 200 and len(payload.get("sentences", [])) > 0,
              f"{len(payload.get('sentences', []))} sentences")
    except Exception:
        check("json download with sentences", False, f"HTTP {status}")

    print("\n5. Bad uploads fail with clear messages")
    status, err1 = post_file("/api/upload", "file", FIXTURES / "not_a_scan.txt")
    check(
        "non-zip file rejected",
        status == 400 and "unsupported file type" in err1.get("error", ""),
        err1.get("error", "")[:90],
    )
    status, err2 = post_file("/api/upload", "file", FIXTURES / "missing_t1ce.zip")
    check(
        "missing sequence rejected",
        status == 400 and "t1ce" in err2.get("error", ""),
        err2.get("error", "")[:90],
    )
    status, err3 = post_file("/api/upload", "file", FIXTURES / "solo_flair.nii.gz")
    check(
        "single 3-D volume rejected",
        status == 400 and "four" in err3.get("error", ""),
        err3.get("error", "")[:90],
    )

    print("\n6. The same scan uploaded as four separate .nii.gz files")
    if (DATA / "BraTS2021_00002").is_dir():
        four = sorted(
            p for p in (DATA / "BraTS2021_00002").glob("*.nii.gz")
            if not p.name.endswith("_seg.nii.gz")
        )
    else:
        # No dataset on this machine (e.g. the shareable bundle): unpack the
        # shipped fixture zip instead -- same four files, no data/ needed.
        four_dir = FIXTURES / "_four_files"
        four_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(FIXTURES / "BraTS2021_00002.zip") as zf:
            zf.extractall(four_dir)
        four = sorted(four_dir.glob("*.nii.gz"))
    check("four sequence files available", len(four) == 4,
          ", ".join(p.name for p in four))
    status, up = post_files("/api/upload", "file", four)
    if check("four-file upload accepted", status == 200, up.get("error", "")):
        check("case id derived from the file names",
              up["patient_id"] == "BraTS2021_00002", up["patient_id"])
        status, res = post_json("/api/analyze", {"token": up["token"], "use_vlm": False})
        if check("four-file analyze succeeded", status == 200, res.get("error", "")):
            ok, detail = matches_reference(res)
            check("four-file result matches runs/predictions.json", ok, detail)
    # only two of the four -> a clear message, not a crash
    status, err4 = post_files("/api/upload", "file", four[:2])
    check(
        "two files rejected with a clear message",
        status == 400 and "four sequences" in err4.get("error", ""),
        err4.get("error", "")[:90],
    )

    if args.vlm:
        print("\n7. VLM phrase pass (Qwen2-VL-2B, a few seconds)")
        # 00009 is the case that yields both kept phrases and a veto in the
        # reference batch, so this exercises the report's positive path too.
        rv, err = upload_and_analyze("BraTS2021_00009", use_vlm=True)
        if check("VLM analyze succeeded", rv is not None, err):
            vlm = rv["vlm"]
            print(f"    status : {vlm['status']}")
            print(f"    kept   : {vlm['phrases']}")
            print(f"    vetoed : {[(v['phrase'][:40], v['reason'][:40]) for v in vlm['vetoed']]}")
            check(
                "VLM status is honest (ready or explicit unavailable)",
                "ready" in vlm["status"] or "unavailable" in vlm["status"],
                vlm["status"][:100],
            )
            # The report page must show exactly what the filter decided:
            # the vetoed-phrases section appears iff something was vetoed.
            status, body = get(f"/report/{rv['_token']}")
            html = body.decode("utf8", "replace")
            check("report page loads", status == 200)
            if vlm["vetoed"]:
                check(
                    "vetoed phrases shown on the report page",
                    "Vetted-out VLM phrases" in html,
                    f"{len(vlm['vetoed'])} vetoed in JSON",
                )
            else:
                check(
                    "no empty veto section when nothing was vetoed",
                    "Vetted-out VLM phrases" not in html,
                )
            if vlm["phrases"]:
                check(
                    "kept phrases appear in the report findings",
                    any(p[:40] in html for p in vlm["phrases"]),
                )

    return finish()


def finish() -> int:
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
