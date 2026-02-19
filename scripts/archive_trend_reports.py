#!/usr/bin/env python3
"""Archive bulky trend_based report artifacts to keep the live reports tree clean.

Design goals
- Keep `reports/trend_based/` as the *live* surface.
- Keep decision bundles in `reports/trend_based/decisions/` but allow moving heavy `raw/` payloads out.
- Default is safe + inspectable: dry-run unless `--execute`.

Current policy
- For every decision bundle under `reports/trend_based/decisions/*/raw/`, move that `raw/` directory into:

    reports/_archive_<UTCSTAMP>/trend_based_decisions_raw/<bundle_name>/raw/

- Leave summaries/CSVs/DECISION.md in place.

If you later want hard deletion, add a second pass that deletes archived payloads.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
REPORTS = WORKSPACE / "reports"
TREND = REPORTS / "trend_based"
DECISIONS = TREND / "decisions"


@dataclass
class MovePlan:
    src: Path
    dst: Path
    bytes_est: int


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def _dir_size_bytes(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for fn in files:
            try:
                total += (Path(root) / fn).stat().st_size
            except FileNotFoundError:
                pass
    return total


def build_plans(archive_root: Path) -> list[MovePlan]:
    plans: list[MovePlan] = []
    if not DECISIONS.exists():
        return plans

    for bundle in sorted(DECISIONS.iterdir()):
        if not bundle.is_dir():
            continue
        raw = bundle / "raw"
        if not raw.exists() or not raw.is_dir():
            continue

        size_b = _dir_size_bytes(raw)
        dst = archive_root / "trend_based_decisions_raw" / bundle.name / "raw"
        plans.append(MovePlan(src=raw, dst=dst, bytes_est=size_b))

    return plans


def apply_plans(plans: list[MovePlan], execute: bool) -> dict:
    moved = []
    skipped = []

    for plan in plans:
        if plan.dst.exists():
            skipped.append({
                "src": str(plan.src),
                "dst": str(plan.dst),
                "reason": "dst_exists",
            })
            continue

        if not execute:
            moved.append({
                "src": str(plan.src),
                "dst": str(plan.dst),
                "bytes_est": plan.bytes_est,
                "status": "dry_run",
            })
            continue

        plan.dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(plan.src), str(plan.dst))
        moved.append({
            "src": str(plan.src),
            "dst": str(plan.dst),
            "bytes_est": plan.bytes_est,
            "status": "moved",
        })

    return {"moved": moved, "skipped": skipped}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="Apply moves (default: dry-run)")
    ap.add_argument(
        "--archive-stamp",
        default=None,
        help="Override archive timestamp (default: current UTC).",
    )
    ap.add_argument(
        "--min-mb",
        type=float,
        default=0.0,
        help="Only archive raw/ dirs >= this size (MB). Default: 0 (all).",
    )
    args = ap.parse_args()

    stamp = args.archive_stamp or _utc_stamp()
    archive_root = REPORTS / f"_archive_{stamp}"

    plans = build_plans(archive_root)

    # size filter
    if args.min_mb > 0:
        thr = int(args.min_mb * 1024 * 1024)
        plans = [p for p in plans if p.bytes_est >= thr]

    result = apply_plans(plans, execute=args.execute)

    out = {
        "stamp": stamp,
        "execute": bool(args.execute),
        "archive_root": str(archive_root),
        "n_plans": len(plans),
        "result": result,
    }

    print(json.dumps(out, indent=2))

    if args.execute and plans:
        archive_root.mkdir(parents=True, exist_ok=True)
        (archive_root / "MANIFEST_trend_based_decisions_raw.json").write_text(
            json.dumps(out, indent=2) + "\n"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
