"""Refit the serving system (B6) from the cached table and persist it.

Why this exists
---------------
The research run (`cgate run`) fits every ablation row in memory and writes
only metrics -- fitted systems were never saved, because the experiment is the
artefact and a pickle of a 3-expert system is not. The Flask demo needs a
*durable* system to serve real inference on uploaded scans.

This refits exactly one row (B6, the quality-routed system) with the same
inputs the original run used -- same table, same slice cache, same split
function and seed, same label thresholds, same expert/router config -- so the
served predictions are the research system's predictions, not a new model.
No re-ingest: the expensive pass (2.9 h) is already on disk.

Before writing the pickle it re-predicts the test split and compares against
`runs/predictions.json` (exported by the original run). If the two disagree
materially the pickle is NOT written: serving a system that does not match the
reported one would make every number on the website a claim about a model that
was never evaluated.

Usage:
    .venv\\Scripts\\python.exe tools\\save_system.py
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
for _cand in (WS / "cgate-mri-main" / "src", WS / "source" / "src", WS / "src"):
    if (_cand / "cgate").is_dir():
        sys.path.insert(0, str(_cand))
        break

from cgate.data.splits import stratified_patient_split  # noqa: E402
from cgate.experiments.ablations import ABLATIONS, build_system  # noqa: E402
from cgate.findings import FindingExtractor  # noqa: E402
from cgate.pipeline import CaseTable, case_confidence  # noqa: E402
from cgate.routing import RouterConfig  # noqa: E402

# Exactly the flags `cgate run --experts torch --pretrained --arch resnet50
# --epochs 12 --batch-size 32` used; everything else was argparse defaults.
TABLE_ROOT = WS / "runs" / "table"
CACHE_ROOT = WS / "runs" / "cache"
PREDICTIONS = WS / "runs" / "predictions.json"
OUT = WS / "runs" / "system_B6.pkl"

MIN_VOXELS = 500  # non-synthetic default (run.py)
TARGET_COVERAGE = 0.8


def _sanity_check(system, table, splits) -> dict:
    """Compare this fit against the original run's exported predictions.

    Returns a metrics dict; raises RuntimeError on material disagreement.
    """
    exported = json.loads(PREDICTIONS.read_text(encoding="utf8"))
    by_id = {c["patient_id"]: c for c in exported["cases"]}

    test_idx = splits["test"]
    fused = system.predict(table, test_idx)
    conf = case_confidence(fused)
    accepted = system.gate.accept(conf)

    agree, n, conf_deltas, gate_agree = 0, 0, [], 0
    for i, row in enumerate(test_idx):
        pid = str(table.patient_ids[row])
        ref = by_id.get(pid)
        if ref is None:
            continue
        n += 1
        # A case matches only when every finding's argmax matches: the gate
        # operates on all critical findings at once, so partial credit would
        # hide exactly the disagreements that matter.
        case_ok = all(
            int(np.argmax(fused[j][i])) == int(np.argmax(np.asarray(ref["probs"][j])))
            for j in range(len(fused))
        )
        agree += int(case_ok)
        conf_deltas.append(abs(float(conf[i]) - float(ref["confidence"])))
        gate_agree += int(bool(accepted[i]) == bool(ref["accepted"]))

    stats = {
        "n_test": n,
        "per_case_findings_agreement": agree / max(n, 1),
        "mean_abs_confidence_delta": float(np.mean(conf_deltas)) if conf_deltas else 1.0,
        "max_abs_confidence_delta": float(np.max(conf_deltas)) if conf_deltas else 1.0,
        "gate_decision_agreement": gate_agree / max(n, 1),
    }
    # Thresholds: every finding must agree on essentially every case, and gate
    # decisions must match almost exactly -- the gate is the product.
    if stats["per_case_findings_agreement"] < 0.98:
        raise RuntimeError(
            f"refit disagrees with exported predictions on findings: {stats}"
        )
    if stats["gate_decision_agreement"] < 0.95:
        raise RuntimeError(f"refit disagrees on gate decisions: {stats}")
    if stats["mean_abs_confidence_delta"] > 0.05:
        raise RuntimeError(f"refit confidences drift from exported: {stats}")
    return stats


def main() -> int:
    t0 = time.time()
    print(f"Loading table from {TABLE_ROOT} ...", flush=True)
    table = CaseTable.load(TABLE_ROOT, slice_cache_root=CACHE_ROOT, require_slices=True)
    print(f"  {table.n_cases} cases, cache: {table.slice_cache.n_slices} slices/case")

    splits = stratified_patient_split(
        table.patient_ids,
        table.cohorts,
        {"train": 0.6, "calib": 0.2, "test": 0.2},
        seed=0,
    )
    print(
        "  splits: "
        + ", ".join(f"{k}={len(v)}" for k, v in splits.items())
    )

    # Labels are re-derived exactly as run.py does (they are not saved in the
    # table by design -- tercile edges belong to a run, not to the dataset).
    extractor = FindingExtractor(min_voxels=MIN_VOXELS).fit_from_measurements(
        [table.measurements[i] for i in splits["train"]]
    )
    table.assign_labels(extractor)
    print(f"  labels fitted: ncr_present cut = {extractor.ncr_cut:,.0f} voxels")

    from cgate.experts.torch_experts import TorchExpertConfig

    torch_config = TorchExpertConfig(
        arch="resnet50",
        img_size=128,
        vit_arch="vit_small_patch16_224",
        vit_img_size=224,
        pretrained=True,
        epochs=12,
        batch_size=32,
        amp=None,
        device=None,
        seed=0,
    )
    spec = next(s for s in ABLATIONS if s.key == "B6")
    print(f"Fitting {spec.key}: {spec.label} (radiomics + CNN + ViT, routed) ...", flush=True)
    system = build_system(
        spec,
        seed=0,
        router=RouterConfig(seed=0),
        cache=table.slice_cache,
        torch_config=torch_config,
    )
    system.fit(
        table, splits["train"], splits["calib"], target_coverage=TARGET_COVERAGE
    )
    print(f"  fitted in {time.time() - t0:.0f}s; gate threshold = {system.gate.threshold_:.4f}")

    if not PREDICTIONS.exists():
        raise SystemExit(
            f"{PREDICTIONS} not found; run `cgate run --predictions ...` first so "
            "the refit can be verified against the reported system."
        )
    print("Verifying against the original run's exported predictions ...", flush=True)
    stats = _sanity_check(system, table, splits)
    print(
        "  OK: "
        + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in stats.items())
    )

    with open(OUT, "wb") as fh:
        pickle.dump({"system": system, "stats": stats, "ncr_cut": extractor.ncr_cut,
                     "min_voxels": MIN_VOXELS, "target_coverage": TARGET_COVERAGE},
                    fh, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = OUT.stat().st_size / 1e6
    print(f"\nWrote {OUT} ({size_mb:.1f} MB) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
