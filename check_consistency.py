#!/usr/bin/env python3
"""
Consistency check for DOE, classifier, and results CSVs
— compares numbers at six-significant-digit precision.

Rules
  • DOE row ↔ exactly one classifier row.
  • Successful row ↔ exactly 30 results rows.
  • Failed row    ↔ zero results rows.
  • No unmatched rows anywhere.
Exit 0 if all good, 1 otherwise.
"""

import csv, re, sys
from pathlib import Path
from collections import defaultdict

DOE = Path("data/DOE_output_results/1MKOH_input_parameters_DOE_maximin_lhs.csv")
CLS = Path("data/COMSOL/results_3D_GE_Applied_Current_1MKOH_63_03_1MKOH_input_parameters_DOE_maximin_lhs_classifier_001.csv")
RES = Path("data/COMSOL/results_3D_GE_Applied_Current_1MKOH_63_03_1MKOH_input_parameters_DOE_maximin_lhs_success_001.csv")

unit_re = re.compile(r"\[.*?]")

def strip_units(t: str) -> str:
    return unit_re.sub("", t).strip()

def norm(v: str) -> str:
    v = strip_units(v)
    try:
        return f"{float(v):.6g}"
    except ValueError:
        return v

def load(path: Path, keep_all_cols=False):
    with path.open(newline='') as f:
        rdr = csv.reader(f)
        hdr_raw = next(rdr)
        hdr = [strip_units(c) for c in hdr_raw]
        rows = [[norm(x) for x in r] for r in rdr]
    return (hdr, rows) if keep_all_cols else (hdr[:12], [r[:12] for r in rows])

def rownum(i):           # 1-based position in file (header is row 1)
    return i + 2

# ---------- load & preprocess ----------------------------------------------
doe_hdr, doe_rows = load(DOE)
cls_hdr, cls_rows_full = load(CLS, keep_all_cols=True)
cls_rows = [r[:12] for r in cls_rows_full]
cls_success = [r[12] for r in cls_rows_full]          # "0"/"1"

res_hdr, res_rows_full = load(RES, keep_all_cols=True)
res_rows = [r[:12] for r in res_rows_full]

# pre-count results per parameter set
res_count = defaultdict(int)
for key in map(tuple, res_rows):
    res_count[key] += 1

errors = 0
# ---------- duplicates: ONLY DOE & classifier ------------------------------
for name, rows in (("DOE", doe_rows), ("classifier", cls_rows)):
    seen = set()
    for i, r in enumerate(rows):
        k = tuple(r)
        if k in seen:
            print(f"✗ duplicate row in {name} (first repeat at line {rownum(i)})")
            errors += 1
            break
        seen.add(k)

# ---------- walk DOE -------------------------------------------------------
for i_doe, doe in enumerate(doe_rows):
    key = tuple(doe)

    # classifier presence
    cls_hits = [j for j, r in enumerate(cls_rows) if tuple(r) == key]
    if not cls_hits:
        print(f"✗ DOE row {rownum(i_doe)} missing from classifier")
        errors += 1
        continue
    if len(cls_hits) > 1:
        places = ", ".join(str(rownum(j)) for j in cls_hits)
        print(f"✗ DOE row {rownum(i_doe)} duplicated in classifier rows {places}")
        errors += 1

    success = cls_success[cls_hits[0]]
    n_res = res_count[key]

    if success == "1":
        if n_res == 30:
            pass
        elif n_res == 0:
            print(f"✗ success at DOE row {rownum(i_doe)} has NO results")
            errors += 1
        elif n_res < 30:
            print(f"✗ success at DOE row {rownum(i_doe)} has only {n_res}/30 results")
            errors += 1
        else:
            print(f"✗ success at DOE row {rownum(i_doe)} duplicated ({n_res} results)")
            errors += 1
    else:          # marked as failure
        if n_res:
            print(f"✗ failed run at DOE row {rownum(i_doe)} nevertheless has {n_res} result rows")
            errors += 1

# ---------- stray results not in DOE ---------------------------------------
unused = [k for k in res_count if k not in map(tuple, doe_rows)]
if unused:
    print(f"✗ {len(unused)} result keys do not correspond to any DOE row")
    errors += 1

# ---------- summary ---------------------------------------------------------
if errors:
    print(f"\nFinished with {errors} problem(s) found.")
    sys.exit(1)

print("✓ All three files consistent (values compared at 6-sig-digit precision)")
sys.exit(0)
