"""Uploaded scan -> the fitted CGate-MRI system -> grounded report.

This is the only module where the website touches the research code, and it
contains no model logic of its own. Every step calls the same functions the
training ingest and the evaluation run used, so an uploaded scan travels the
identical path a BraTS case did:

    read NIfTI        nibabel, canonical RAS       (cgate.data.brats conventions)
    brain mask        any modality has signal       (brats.load_case rule)
    IQMs              compute_iqms on FLAIR         (ingest_cases: modality 3)
    feature views     extract_views                 (texture / local / global)
    slice cache       SliceCacheWriter, same config as training
    inference         TrainedSystem.predict         (experts -> temps -> router)
    gate              ConfidenceGate.accept          (threshold fitted at save time)
    report            cgate.reporting.render_report (+ optional VLM phrases)

There is deliberately no second "web" pipeline: a simplified copy would drift
from the research one and quietly serve different numbers.

Required on disk (built by tools/save_system.py):
    runs/system_B6.pkl   the fitted B6 system (experts + calibration + gate)
"""

from __future__ import annotations

import json
import os
import pickle
import re
import threading
import zipfile
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
UPLOADS = WS / "runs" / "uploads"
SYSTEM_PKL = WS / "runs" / "system_B6.pkl"
TRAINING_CACHE = WS / "runs" / "cache"

# Limits: a BraTS case is ~50 MB of NIfTI; anything far past that is either a
# whole-dataset zip or a non-compressed export, and should fail fast rather
# than be read into RAM.
MAX_ZIP_BYTES = 400 * 1024 * 1024
MAX_NIFTI_BYTES = 300 * 1024 * 1024


class UploadError(Exception):
    """A problem with the uploaded file. The message is shown to the user."""


class InferenceError(Exception):
    """A problem running the pipeline. The message is shown to the user."""


# ---------------------------------------------------------------------------
# Loading the upload
# ---------------------------------------------------------------------------


def _require_nib():
    try:
        import nibabel  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise InferenceError(
            "nibabel is not installed. Run: pip install -e '.[sci,io,torch]'"
        ) from exc


def _find_modality_file(root: Path, modality: str) -> Path | None:
    """Locate one sequence's file anywhere below `root` (zip may nest a folder).

    Same suffix priority as cgate.data.brats._find_one, applied recursively.
    """
    from cgate.data.brats import MODALITY_PATTERNS

    for suffix in MODALITY_PATTERNS[modality]:
        for pattern in (f"*_{suffix}.nii*", f"*-{suffix}.nii*"):
            hits = sorted(p for p in root.rglob(pattern) if p.is_file())
            if hits:
                return hits[0]
    return None


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract, refusing paths that would escape `dest`."""
    for member in zf.infolist():
        name = member.filename
        if name.startswith(("/", "\\")) or ".." in Path(name).parts:
            raise UploadError(f"zip contains an unsafe path: {name!r}")
    zf.extractall(dest)


def _load_zip(src: Path, work: Path) -> tuple[dict[str, Path], str]:
    try:
        zf = zipfile.ZipFile(src)
    except zipfile.BadZipFile as exc:
        raise UploadError("not a valid .zip file") from exc
    dest = work / "extracted"
    dest.mkdir(parents=True, exist_ok=True)
    with zf:
        _safe_extract(zf, dest)
    files = {}
    missing = []
    for modality in ("t1", "t1ce", "t2", "flair"):
        hit = _find_modality_file(dest, modality)
        if hit is None:
            missing.append(modality)
        else:
            files[modality] = hit
    if missing:
        found = sorted(p.name for p in dest.rglob("*.nii*"))
        raise UploadError(
            "missing MRI sequence(s): "
            + ", ".join(missing)
            + ". A case must contain all four co-registered sequences "
            "(t1, t1ce, t2, flair) as .nii/.nii.gz files. "
            + (f"Files found: {', '.join(found[:8])}." if found else "Zip was empty.")
        )
    # The case id comes from the uploaded file's name (BraTS2021_00002.zip ->
    # BraTS2021_00002), which is what the user sees on screen.
    return files, src.stem


def _stack_sequences(files: dict[str, Path]) -> np.ndarray:
    """Read the four named .nii/.nii.gz files into one (4, H, W, D) array.

    Shared by the .zip path and the "four files selected together" path so
    both read volumes exactly the same way (canonical RAS, float32, shape
    agreement enforced).
    """
    _require_nib()
    import nibabel as nib

    from cgate.data.brats import MODALITY_ORDER

    volumes = []
    shapes = []
    for modality in MODALITY_ORDER:
        path = files[modality]
        if path.stat().st_size > MAX_NIFTI_BYTES:
            raise UploadError(f"{path.name} is too large to be a single volume")
        img = nib.as_closest_canonical(nib.load(str(path)))
        arr = np.asarray(img.dataobj, dtype=np.float32)
        if arr.ndim != 3:
            raise UploadError(
                f"{path.name}: expected a 3-D volume, got shape {arr.shape}"
            )
        volumes.append(arr)
        shapes.append(arr.shape)
    if len(set(shapes)) != 1:
        raise UploadError(
            "the four sequences have different shapes "
            + ", ".join(str(s) for s in shapes)
            + " — they must be co-registered (BraTS-style) before upload."
        )
    return np.stack(volumes, axis=0)


def _load_volumes(src: Path, work: Path) -> tuple[np.ndarray, str]:
    """Return ((4, H, W, D) float32 volumes in canonical RAS, patient id)."""
    _require_nib()
    import nibabel as nib

    suffix = src.suffix.lower()
    if suffix == ".zip":
        files, patient_id = _load_zip(src, work)
        return _stack_sequences(files), patient_id

    if suffix in (".nii", ".gz"):  # .nii.gz arrives with suffix .gz
        if src.stat().st_size > MAX_NIFTI_BYTES:
            raise UploadError("file is too large (max 300 MB per volume)")
        img = nib.as_closest_canonical(nib.load(str(src)))
        arr = np.asarray(img.dataobj, dtype=np.float32)
        if arr.ndim == 4:
            axes = [i for i, s in enumerate(arr.shape) if s == 4]
            if len(axes) != 1:
                raise UploadError(
                    f"4-D file with shape {arr.shape}: expected exactly one "
                    "axis of size 4 (the four sequences)."
                )
            arr = np.moveaxis(arr, axes[0], 0)
        elif arr.ndim == 3:
            raise UploadError(
                "this is a single 3-D volume, but the system needs all four "
                "co-registered sequences (t1, t1ce, t2, flair). Select the "
                "four sequence files together, zip them into one .zip, or "
                "upload a 4-D file with the four sequences."
            )
        else:
            raise UploadError(f"unsupported shape {arr.shape}")
        if len(set(arr.shape[1:])) != 1:
            raise UploadError(f"sequence axes disagree: {arr.shape}")
        return arr, src.stem.replace(".nii", "")

    raise UploadError(
        f"unsupported file type '{src.name}'. Upload a .zip containing one "
        "case's four .nii/.nii.gz sequences, or a single 4-D .nii/.nii.gz."
    )


_NII_SUFFIX_RE = re.compile(r"\.nii(\.gz)?$", re.IGNORECASE)
_SEQ_NAME_RE = re.compile(r"[_\-.](t1ce|t1|t2|flair)$", re.IGNORECASE)


def _find_all_modality_files(root: Path, modality: str) -> list[Path]:
    """Every file below `root` matching one sequence, without duplicates."""
    from cgate.data.brats import MODALITY_PATTERNS

    hits: list[Path] = []
    for suffix in MODALITY_PATTERNS[modality]:
        for pattern in (f"*_{suffix}.nii*", f"*-{suffix}.nii*"):
            hits.extend(p for p in root.rglob(pattern) if p.is_file())
    seen: set[Path] = set()
    out = []
    for p in sorted(hits):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _patient_id_from(files: dict[str, Path]) -> str:
    """Case id from the file names: BraTS2021_00002_t1.nii.gz -> BraTS2021_00002."""
    stems = []
    for path in files.values():
        stem = _NII_SUFFIX_RE.sub("", path.name)
        stems.append(_SEQ_NAME_RE.sub("", stem))
    prefix = os.path.commonprefix(stems).rstrip("_-. ")
    return prefix or "upload"


def validate_upload_many(sources: list[Path], work: Path) -> tuple[np.ndarray, str]:
    """Validate the four sequence files selected together (no zip involved).

    The files were saved flat into `work`; each is located by its modality
    suffix, exactly as the zip path does after extraction, then read by the
    same `_stack_sequences` and checked by the same brain/shape rules.
    """
    if len(sources) < 4:
        raise UploadError(
            f"got {len(sources)} file(s), but a case needs all four sequences "
            "(t1, t1ce, t2, flair). Select the four .nii/.nii.gz files "
            "together, or upload one .zip containing them."
        )
    bad = [p.name for p in sources if not _NII_SUFFIX_RE.search(p.name)]
    if bad:
        raise UploadError(
            "only .nii/.nii.gz volumes can be combined (found: "
            + ", ".join(bad[:6])
            + "). To mix file types, zip the four sequences into one .zip instead."
        )

    files: dict[str, Path] = {}
    duplicates = []
    missing = []
    for modality in ("t1", "t1ce", "t2", "flair"):
        hits = _find_all_modality_files(work, modality)
        if len(hits) > 1:
            duplicates.append(f"{modality} ({', '.join(h.name for h in hits)})")
        elif hits:
            files[modality] = hits[0]
        else:
            missing.append(modality)
    if duplicates:
        raise UploadError(
            "more than one file matches the same sequence: " + ", ".join(duplicates)
        )
    if missing:
        found = sorted(p.name for p in work.rglob("*.nii*"))
        raise UploadError(
            "missing MRI sequence(s): "
            + ", ".join(missing)
            + ". A case must contain all four co-registered sequences "
            "(t1, t1ce, t2, flair) as .nii/.nii.gz files. "
            + (
                f"Files selected: {', '.join(found[:8])}."
                if found
                else "No .nii files were uploaded."
            )
        )
    return _validate_vols(_stack_sequences(files), _patient_id_from(files))


def _validate_vols(vols: np.ndarray, patient_id: str) -> tuple[np.ndarray, str]:
    """Shared post-load checks (both upload shapes take this path)."""
    if vols.shape[0] != 4:  # defensive; the loaders guarantee 4
        raise UploadError(f"expected 4 sequences, got {vols.shape[0]}")
    if not np.all(np.isfinite(vols)):
        raise UploadError("volume contains NaN/Inf values")

    brain = np.any(vols > 0, axis=0)
    frac = float(brain.mean())
    if frac < 0.01:
        raise UploadError(
            "no brain signal found (less than 1% of voxels are non-zero). "
            "The upload appears empty or fully zero-masked."
        )
    if frac > 0.95:
        raise UploadError(
            "the image does not look skull-stripped (over 95% of voxels carry "
            "signal). This system expects BraTS-style skull-stripped, "
            "co-registered volumes."
        )
    # patient ids become file names in the slice cache; sanitise like the cache does
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in patient_id)
    return vols, safe[:80] or "upload"


def validate_upload(src: Path, work: Path) -> tuple[np.ndarray, str]:
    """Full read + validation. Raises UploadError with a user-facing message."""
    vols, patient_id = _load_volumes(src, work)
    return _validate_vols(vols, patient_id)


# ---------------------------------------------------------------------------
# Preprocessing: identical steps to ingest_cases (minus labels/masks)
# ---------------------------------------------------------------------------


def build_case_table(vols: np.ndarray, patient_id: str, work: Path):
    """One-case CaseTable with a slice cache built from the same config.

    Mirrors `ingest_cases` for a single case: IQMs on FLAIR (modality 3),
    feature views from the volumes, slices written with the training cache's
    own config so the CNN/ViT receive exactly the representation they saw.
    """
    from cgate.data.features import extract_views
    from cgate.data.slices import SliceCacheConfig, SliceCacheWriter
    from cgate.pipeline import CaseTable
    from cgate.quality import ROUTER_FEATURES, compute_iqms, iqm_vector

    brain = np.any(vols > 0, axis=0)
    iqms = compute_iqms(vols[3].astype(np.float64), brain)
    iqm_row = iqm_vector(iqms, ROUTER_FEATURES)

    views = extract_views(vols, brain)
    views_out = {name: row[None, :] for name, row in views.items()}

    # Same slice-selection config as the training cache (read from its index,
    # not re-defaulted: a different n_slices would silently change what the
    # networks see relative to their training).
    if (TRAINING_CACHE / "index.json").exists():
        cfg_dict = json.loads(
            (TRAINING_CACHE / "index.json").read_text(encoding="utf8")
        ).get("config", {})
        config = SliceCacheConfig(**cfg_dict)
    else:
        config = SliceCacheConfig()

    from cgate.data.brats import BratsCase

    case = BratsCase(
        patient_id=patient_id,
        volumes=vols,
        mask=np.zeros(vols.shape[1:], dtype=np.int16),  # no ground truth at upload time
        brain_mask=brain,
        cohort="upload",
        quality_level=float("nan"),
        truth={},
    )
    cache_root = work / "cache"
    writer = SliceCacheWriter(cache_root, config)
    writer.add(case)
    cache = writer.close()
    views_out["volume"] = cache.rows_for([patient_id]).reshape(-1, 1)

    table = CaseTable(
        patient_ids=np.asarray([patient_id]),
        cohorts=np.asarray(["upload"]),
        quality_level=np.asarray([np.nan]),
        iqms=iqm_row[None, :],
        views=views_out,
        # Labels/measurements need a ground-truth segmentation, which an
        # uploaded scan does not have and inference never reads.
        measurements=[{}],
        iqm_names=ROUTER_FEATURES,
        slice_cache=cache,
    )
    return table


# ---------------------------------------------------------------------------
# The fitted system and the VLM (both lazy, both optional-safe)
# ---------------------------------------------------------------------------

_SYSTEM = None
_VLM = None
_VLM_STATUS = "not requested"
_LOAD_LOCK = threading.Lock()  # guards the two lazy globals
_RUN_LOCK = threading.Lock()   # one inference at a time: single GPU


class _TorchCacheSwap:
    """Point every torch expert's cache at this upload's slice cache.

    CNN/ViT experts read pixels through the SliceCache they were constructed
    with (the training cache). A new scan lives in its own one-case cache, so
    the pointer is swapped for the duration of the call. The caller holds
    _RUN_LOCK, so nothing else can observe the swap.
    """

    def __init__(self, system, cache):
        self.system, self.cache = system, cache
        self.saved = []

    def __enter__(self):
        for expert in self.system.experts:
            if hasattr(expert, "cache"):
                self.saved.append((expert, expert.cache))
                expert.cache = self.cache
        return self

    def __exit__(self, *exc):
        for expert, cache in self.saved:
            expert.cache = cache
        return False


def get_system():
    """Load the fitted B6 system once. Fails loudly if it was never saved."""
    global _SYSTEM
    if _SYSTEM is not None:
        return _SYSTEM
    with _LOAD_LOCK:
        if _SYSTEM is not None:
            return _SYSTEM
        if not SYSTEM_PKL.exists():
            raise InferenceError(
                "model file runs/system_B6.pkl not found. Fit it once with: "
                ".venv\\Scripts\\python.exe tools\\save_system.py"
            )
        try:
            with open(SYSTEM_PKL, "rb") as fh:
                payload = pickle.load(fh)
        except Exception as exc:  # wrong env / CUDA missing / truncated file
            raise InferenceError(
                f"could not load runs/system_B6.pkl ({type(exc).__name__}: {exc}). "
                "It must be read by the same Python/torch environment that wrote "
                "it; re-fit with tools\\save_system.py if it is gone."
            ) from exc
        _SYSTEM = payload["system"]
        _SYSTEM.fit_log = getattr(_SYSTEM, "fit_log", {})
        _SYSTEM.saved_stats = payload.get("stats", {})
        return _SYSTEM


def get_vlm():
    """Lazily load the VLM. Returns (reporter_or_None, status_string).

    Attempted at most once per process: a load that fails (no GPU memory,
    transformers missing, gated model) fails the same way next time, and
    retrying would stall every analysis behind the same error.
    """
    global _VLM, _VLM_STATUS
    with _LOAD_LOCK:
        if _VLM is not None:
            return _VLM, _VLM_STATUS
        if _VLM_STATUS != "not requested":
            return _VLM, _VLM_STATUS
        try:
            from cgate.reporting.vlm import VLMConfig, VLMReporter

            reporter = VLMReporter(config=VLMConfig())
            if reporter.ensure_loaded():
                _VLM = reporter
                _VLM_STATUS = f"ready ({reporter.loaded_model_id})"
            else:
                _VLM_STATUS = f"unavailable ({reporter._error or 'load failed'})"
        except Exception as exc:  # transformers missing, OOM, no network
            _VLM_STATUS = f"unavailable ({type(exc).__name__}: {exc})"
        return _VLM, _VLM_STATUS


def status() -> dict:
    """Honest availability of what the demo needs (surfaced on the pages)."""
    return {
        "system_saved": SYSTEM_PKL.exists(),
        "system_path": str(SYSTEM_PKL.relative_to(WS)),
        "vlm": _VLM_STATUS,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _findings_context(case) -> str:
    return "\n".join(
        f"- {name}: {cls} (p={p:.2f})" for name, (cls, p) in case.predicted.items()
    )


def make_preview(table) -> str:
    """Save the 2x2 sequence mosaic the networks actually read; returns URL.

    Built from the upload's own slice cache, so the picture shown on screen
    is the input the prediction comes from.
    """
    from cgate.reporting.vlm import build_mosaic

    preview_path = table.slice_cache.root.parent / "preview.png"
    build_mosaic(table.slice_cache.get(0)).save(preview_path)
    return f"/uploads/{table.slice_cache.root.parent.name}/preview.png"


def analyze(table, use_vlm: bool = False) -> dict:
    """Run the fitted system on the prepared one-case table.

    Returns a JSON-serialisable result: findings, confidence, gate decision,
    routing weights, the grounding ledger, and (when available) VLM phrases.
    `_report` (the Report object) is for the caller's report page only --
    pop it before sending anything to the browser.
    """
    from cgate.pipeline import case_confidence
    from cgate.reporting.grounding import ReportCase, filter_phrases
    from cgate.reporting.render import render_report

    with _RUN_LOCK:
        system = get_system()
        idx = np.zeros(1, dtype=int)

        with _TorchCacheSwap(system, table.slice_cache):
            fused = system.predict(table, idx)
            conf = float(case_confidence(fused)[0])
            accepted = bool(system.gate.accept(np.asarray([conf]))[0])
            q = system.standardizer.transform(table.iqms)  # (1, n_router_features)
            # Per-case expert weights from the quality router: what "routing
            # decision" means concretely for this scan.
            if system.router is not None:
                weights = system.router.weights(q)[0]  # (n_findings, n_experts)
            else:
                weights = None

        report_case = ReportCase(
            patient_id=str(table.patient_ids[0]),
            cohort="upload",
            probs=[np.asarray(f[0]) for f in fused],  # fused[j] is (n, K) for finding j
            confidence=conf,
            gate_threshold=float(system.gate.threshold_),
            accepted=accepted,
            iqms={
                name: float(table.iqms[0, k])
                for k, name in enumerate(table.iqm_names)
            },
            q_std=tuple(float(z) for z in q[0]),
            q_names=tuple(table.iqm_names),
        )

        # Optional VLM phrases over the same mosaic; veto filter runs on the
        # calibrated findings, exactly as in cgate-report.
        phrases, rejected, vlm_note, vlm_model = [], [], None, None
        if use_vlm:
            reporter, status = get_vlm()
            vlm_note = status
            if reporter is not None:
                from cgate.reporting.vlm import build_mosaic

                err_before = reporter._error
                raw = reporter.describe(
                    build_mosaic(table.slice_cache.get(0)),
                    _findings_context(report_case),
                )
                if reporter._error != err_before:
                    # describe() swallows exceptions into _error; say so rather
                    # than showing a silent empty phrase list as if the model
                    # simply had nothing to add.
                    vlm_note = (
                        f"error during generation ({reporter._error}) — "
                        "report generated from findings only"
                    )
                kept, vetoed = filter_phrases(raw, report_case)
                phrases = kept
                rejected = [(p, r.why or r.name) for p, r in vetoed]
                vlm_model = reporter.loaded_model_id
            else:
                vlm_note = status + " — report generated from findings only"

        report = render_report(
            report_case,
            vlm_phrases=phrases,
            vlm_rejected=rejected,
            vlm_model=vlm_model,
        )

        expert_names = list(system.expert_names)
        from cgate.schema import CRITICAL_MASK, FINDING_SCHEMA

        findings = []
        for j, spec in enumerate(FINDING_SCHEMA):
            probs = np.asarray(fused[j][0])
            k = int(np.argmax(probs))
            findings.append(
                {
                    "name": spec.name,
                    "label": spec.description.split(".")[0],
                    "class": spec.classes[k],
                    "p": float(probs[k]),
                    "probs": {c: float(v) for c, v in zip(spec.classes, probs)},
                    "critical": bool(spec.critical),
                }
            )

        routing = None
        if weights is not None:
            routing = [
                {
                    "finding": spec.name,
                    "weights": {
                        name: float(weights[j, e])
                        for e, name in enumerate(expert_names)
                    },
                }
                for j, spec in enumerate(FINDING_SCHEMA)
            ]

        ledger = [
            {
                "text": s.text,
                "evidence": [
                    {"source": e.source, "detail": e.detail} for e in s.evidence
                ],
            }
            for s in report.sentences
        ]

        return {
            "patient_id": report_case.patient_id,
            "system": "B6: Radiomics + CNN + ViT, quality-routed (ours)",
            "findings": findings,
            "confidence": conf,
            "gate_threshold": float(system.gate.threshold_),
            "accepted": accepted,
            "critical_mask": list(CRITICAL_MASK),
            "routing": routing,
            "expert_names": expert_names,
            "iqms": report_case.iqms,
            "q_mean_abs_z": float(np.mean(np.abs(q[0]))),
            "vlm": {
                "status": vlm_note or "not requested",
                "phrases": phrases,
                "vetoed": [{"phrase": p, "reason": r} for p, r in rejected],
            },
            "ledger": ledger,
            "report_markdown": report.markdown(),
            "report_json": report.to_json(),
            # Not for the browser: the report page renders from this object.
            "_report": report,
        }
