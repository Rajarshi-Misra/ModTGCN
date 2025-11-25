import pandas as pd
from pathlib import Path
import itertools

# Root directory
results_dir = Path("results")

# Ignore these directories
ignore_dirs = {"best_hyperparams_fine_tuned", "hyperparameter_plots_fine_tuned"}

# Fixed possible values
graph_constructions = ["cosine", "gaussian"]
weight_adjs = ["True", "False"]
label_types = ["true_label", "pred_label"]

# Datasets (columns)
datasets = ["mr", "r8", "r52", "ohsumed", "20ng"]

# Map existing directories
case_map = {}
for sub_dir in results_dir.iterdir():
    if not (sub_dir.is_dir() and sub_dir.name.endswith("_fine_tuned")):##NOTE: THIS IS FOR SUMMARIZING fine_tuned results only
        continue
    parts = sub_dir.name.split("_")
    if len(parts) != 6:  # Must be: true_label_False_gaussian_fine_tuned (6 parts)
        continue
    label_type = parts[0] + "_" + parts[1]     # true_label, pred_label
    weight_adj = parts[2]                      # True/False
    graph_construction = parts[3]              # gaussian/cosine
    case_map[(graph_construction, weight_adj, label_type)] = sub_dir

# Generate all possible combinations
all_combos = itertools.product(graph_constructions, weight_adjs, label_types)

# Helper to compute mean ± std
def summarize_metrics(file_list):
    dfs = []
    for f in file_list:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return "Missing"
    full = pd.concat(dfs, ignore_index=True)

    # Assuming columns "micro_f1" and "macro_f1" exist
    result = {}
    for metric in ["micro_f1", "macro_f1"]:
        if metric in full.columns:
            mean = full[metric].mean()
            std = full[metric].std()
            result[metric] = f"{mean:.4f} ± {std:.4f}"
        else:
            result[metric] = "NA"
    return result

# Build rows
rows = []
for gc, wa, lt in all_combos:
    row = {
        "Graph_Construction": gc,
        "Weight_Adj": wa,
        "Label_Type": lt,
    }
    dir_path = case_map.get((gc, wa, lt))  # might not exist
    for dataset in datasets:
        if dir_path:
            # Collect all seed files for this dataset
            files = list(dir_path.glob(f"{dataset}_*_results.csv"))
            summary = summarize_metrics(files)
            if summary == "Missing":
                row[dataset] = "Missing"
            else:
                row[dataset] = f"micro: {summary['micro_f1']}, macro: {summary['macro_f1']}"
        else:
            row[dataset] = "Missing"
    rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

# Ensure column order
df = df[["Graph_Construction", "Weight_Adj", "Label_Type"] + datasets]

# Save CSV
csv_file = "results/summary/results_summary.csv"
df.to_csv(csv_file, index=False)

print(f"CSV created: {csv_file}")
