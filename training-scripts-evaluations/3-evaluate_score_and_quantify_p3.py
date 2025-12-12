# Prompt Engineering Experiments — Step 3 (Evaluate P3)
# SPICE Netlist Metrics: syntax_validity, coverage_score, exact_match_score

import csv
import json
import re
from pathlib import Path

BENCHMARK_PATH = Path("benchmark/spice_benchmark.json")
RUNS_DIR = Path("runs/P3")
RESULTS_DIR = Path("results")

CSV_OUT = RESULTS_DIR / "P3_results.csv"
SUMMARY_OUT = RESULTS_DIR / "P3_summary.json"

def load_and_parse_json(file_path):
    # load a json file and parse it into objects
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    data = json.loads(text_data)
    return data

def normalize_text(text):
    # normalize the text
    t = text.replace("\r\n", "\n").replace("\r", "\n").lower()
    t = re.sub(r"[ \t]+", " ", t)   # collapse horizontal whitespace
    t = re.sub(r"\n+", "\n", t)     # collapse blank lines
    return t.strip()

def must_contain_score(text_norm, must_contain):
    # return (found, total) for must_contain checks
    total = len(must_contain)
    found = 0

    for item in must_contain:
        if normalize_text(item) in text_norm:
            found += 1

    return found, total

def must_contain_any_score(text_norm, must_contain_any):
    # each group counts as 1, if any option appears then OKAY
    total = len(must_contain_any)
    found = 0

    for group in must_contain_any:
        ok = any(normalize_text(opt) in text_norm for opt in group)
        if ok:
            found += 1

    return found, total

def compute_coverage_score(text_norm, entry):
    must_contain = entry.get("must_contain", [])
    must_contain_any = entry.get("must_contain_any", [])

    found_must, total_must = must_contain_score(text_norm, must_contain)
    found_any, total_any = must_contain_any_score(text_norm, must_contain_any)

    total = total_must + total_any
    if total == 0:
        return 0.0

    return (found_must + found_any) / total

def default_key_exact(entry):
    ## having issues here ##
    # use line-level 'must_contain' items that are NOT directives - so, not starting with '.'
    # this focuses exact match on component parameter lines like V1/R1/R2

    keys = []
    for s in entry.get("must_contain", []):
        s_norm = normalize_text(s)
        if s_norm.startswith("."):
            continue
        keys.append(s)
    return keys

def compute_exact_match_score(text_norm, entry):
    key_exact = entry.get("key_exact", None)
    if key_exact is None:
        key_exact = default_key_exact(entry)

    total = len(key_exact)
    if total == 0:
        return 0.0

    found = 0
    for item in key_exact:
        if normalize_text(item) in text_norm:
            found += 1

    return found / total

def syntax_validity(text_raw, entry):
    # must contain .end
    # must contain at least one element line (R/C/L/V/I...)
    # if benchmark requires a directive, then it must be there

    t_norm = normalize_text(text_raw)

    if ".end" not in t_norm:
        return False

    lines = [ln.strip() for ln in text_raw.splitlines() if ln.strip()]
    element_ok = any(
        re.match(r"^(r|c|l|v|i|d|q|m|x|e|f|g|h)\w*\b", ln, re.IGNORECASE)
        for ln in lines
    )
    if not element_ok:
        return False

    must = [normalize_text(x) for x in entry.get("must_contain", [])]
    for directive in [".op", ".ac", ".tran", ".dc"]:
        expects = any(m == directive or m.startswith(directive + " ") for m in must)
        if expects and directive not in t_norm:
            return False

    return True

def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def missing_must_contain(text_norm, must_contain):
    missing = []
    for item in must_contain:
        if normalize_text(item) not in text_norm:
            missing.append(item)
    return missing

def missing_must_contain_any(text_norm, must_contain_any):
    missing_groups = []
    for group in must_contain_any:
        if not any(normalize_text(opt) in text_norm for opt in group):
            missing_groups.append(group)
    return missing_groups

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_and_parse_json(BENCHMARK_PATH)
    if not isinstance(data, list):
        raise ValueError("Benchmark JSON must be a list of objects.")

    rows = []
    syntax_pass_count = 0
    coverage_sum = 0.0
    exact_sum = 0.0

    for entry in data:
        case_id = entry["id"]
        category = entry.get("category", "")
        out_path = RUNS_DIR / f"{case_id}.cir"

        if not out_path.exists():
            text_raw = ""
            err = "missing_output_file"
        else:
            text_raw = out_path.read_text(encoding="utf-8", errors="replace")
            err = ""

        text_norm = normalize_text(text_raw)

        miss_must = missing_must_contain(text_norm, entry.get("must_contain", []))
        miss_any = missing_must_contain_any(text_norm, entry.get("must_contain_any", []))

        syn = syntax_validity(text_raw, entry)
        cov = compute_coverage_score(text_norm, entry)
        exm = compute_exact_match_score(text_norm, entry)

        syntax_pass_count += int(syn)
        coverage_sum += cov
        exact_sum += exm

        rows.append({
            "case_id": case_id,
            "category": category,
            "syntax_pass": int(syn),
            "coverage_score": round(cov, 4),
            "exact_match_score": round(exm, 4),
            "missing_must_contain": "|".join(miss_must),
            "missing_must_contain_any": "|".join([",".join(g) for g in miss_any]),
            "output_file": str(out_path),
            "error": err,
        })

    n = len(rows)
    summary = {
        "num_cases": n,
        "syntax_pass_rate": (syntax_pass_count / n) if n else 0.0,
        "avg_coverage_score": (coverage_sum / n) if n else 0.0,
        "avg_exact_match_score": (exact_sum / n) if n else 0.0,
    }

    save_csv(CSV_OUT, rows)
    save_json(SUMMARY_OUT, summary)

    print(f"Wrote: {CSV_OUT}")
    print(f"Wrote: {SUMMARY_OUT}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
