"""Reproduce the entire WEZ-XAI pipeline from raw CSVs to figures + tables.

Runs notebooks/01_eda.py through notebooks/08_rnez_portability.py in order,
with per-step timing and a fail-fast contract: if any script returns non-zero,
the run aborts.

Usage:
    .venv/bin/python run_all.py
    .venv/bin/python run_all.py --from 04   # resume from step 04
    .venv/bin/python run_all.py --only 03   # run only step 03
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable  # use the interpreter that launched this script
NOTEBOOKS = [
    "01_eda.py",
    "02_features.py",
    "03_models.py",
    "04_xai_local.py",
    "05_xai_global.py",
    "06_model_specific.py",
    "07_advanced.py",
    "08_rnez_portability.py",
]


def step_num(name: str) -> str:
    return name.split("_")[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_", default=None,
                        help="Resume from step (e.g. 04).")
    parser.add_argument("--only", default=None,
                        help="Run only this step (e.g. 03).")
    args = parser.parse_args()

    to_run = NOTEBOOKS
    if args.only:
        to_run = [n for n in NOTEBOOKS if step_num(n) == args.only]
    elif args.from_:
        to_run = [n for n in NOTEBOOKS if step_num(n) >= args.from_]

    if not to_run:
        print("No matching steps.")
        return 1

    print(f"Will run {len(to_run)} step(s) with {PY}")
    print(f"  {' → '.join(step_num(n) for n in to_run)}")

    overall_t0 = time.time()
    timings = []
    for name in to_run:
        script = ROOT / "notebooks" / name
        print(f"\n{'='*80}\n▶  {name}\n{'='*80}")
        t0 = time.time()
        rc = subprocess.run([PY, str(script)], cwd=ROOT).returncode
        dt = time.time() - t0
        timings.append((name, dt, rc))
        if rc != 0:
            print(f"\n✗ {name} exited with code {rc} after {dt:.1f}s. Aborting.")
            return rc

    total = time.time() - overall_t0
    print(f"\n{'='*80}\n✓ Pipeline complete in {total/60:.1f} min")
    for name, dt, _ in timings:
        print(f"    {name:35s}  {dt:6.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
