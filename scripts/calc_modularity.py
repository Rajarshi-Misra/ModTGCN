import os
import sys
import argparse
from typing import Dict, Optional, List
import json
import numpy as np
import pandas as pd
import glob
import re

# make repo importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.graph.modularity import AdjacencyMatrixCreatorForModularity

def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def find_splits_for_dataset(root: str, name: str) -> Optional[Dict[str, str]]:
    """
    Look for train/val/test CSVs for dataset 'name' under root.
    Supported patterns:
      - <root>/<name>/train.csv, val.csv, test.csv
      - <root>/<name>/*train*.csv, *val* or *dev* or *validation*, *test*.csv
      - <root>/*{name}*train*.csv etc
    Returns dict with keys 'train','val','test' -> file paths or None if not found.
    """
    candidates = {}
    dataset_dir = os.path.join(root, name)
    patterns = []
    if os.path.isdir(dataset_dir):
        patterns = glob.glob(os.path.join(dataset_dir, "*.csv"))
    else:
        patterns = glob.glob(os.path.join(root, "*.csv")) + glob.glob(os.path.join(root, "**", f"*{name}*.csv"), recursive=True)

    if not patterns:
        return None

    for p in patterns:
        fname = os.path.basename(p).lower()
        if re.search(r"(^|[^a-z0-9])(train)([^a-z0-9]|$)", fname) or "train" in fname:
            candidates.setdefault("train", []).append(p)
        if re.search(r"(^|[^a-z0-9])(val|dev|validation)([^a-z0-9]|$)", fname) or "val" in fname:
            candidates.setdefault("val", []).append(p)
        if re.search(r"(^|[^a-z0-9])(test)([^a-z0-9]|$)", fname) or "test" in fname:
            candidates.setdefault("test", []).append(p)

    # pick first match for each split (deterministic sort)
    out = {}
    for split in ("train", "val", "test"):
        files = sorted(candidates.get(split, []))
        out[split] = files[0] if files else None

    # require train + test at minimum
    if out["train"] is None or out["test"] is None:
        return None
    return out

def process_dataset(root: str, name: str, method: str, llm: str, sigma: float):
    splits = find_splits_for_dataset(root, name)
    if splits is None:
        return {"dataset": name, "error": "split files not found"}
    df_train = load_df(splits["train"])
    df_val = load_df(splits["val"]) if splits["val"] else pd.DataFrame(columns=df_train.columns)
    df_test = load_df(splits["test"])

    for df in (df_train, df_val, df_test):
        if "label" not in df.columns or "text" not in df.columns:
            return {"dataset": name, "error": "missing 'text' or 'label' columns in splits"}

    creator = AdjacencyMatrixCreatorForModularity(method=method)
    run_args = argparse.Namespace(llm=llm, sigma=sigma, k=0, increase=1.0, decrease=1.0, modify_graph=False)

    adj = creator.create(run_args, df_train=df_train, df_val=df_val, df_test=df_test)
    adj = adj.detach().cpu().numpy() ##Dealing in Numpy
    labels_vec = np.concatenate([
        df_train["label"].values,
        df_val["label"].values,
        df_test["label"].values
    ]).astype(object)

    # one-hot encoding
    classes, labels_idx = np.unique(labels_vec, return_inverse=True)
    S = np.eye(len(classes))[labels_idx]

    D_W = np.sum(adj,axis=0)
    e_W = (np.sum(adj)/2)
    B_W = (adj - np.outer(D_W,D_W)/(2*e_W))
    modularity = np.trace(S.T@B_W@S) * (1/(2*e_W))
    return { "dataset" : name, "modularity": modularity }


def main():
    p = argparse.ArgumentParser(description="Compute homophily for SBERT-based cosine or gaussian graphs (basic graphs only).")
    p.add_argument("--method", choices=["from_sbert_embeddings_cosine", "from_sbert_embeddings_gaussian"], required=True)
    p.add_argument("--datasets-root", default="datasets", help="root folder where dataset subfolders or csvs live")
    p.add_argument("--datasets", nargs="+", default=["mr", "20ng", "r8", "r52", "ohsumed"], help="dataset names to analyze (default common set).")
    p.add_argument("--llm", default="all-mpnet-base-v2")
    p.add_argument("--sigma", type=float, default=1.0, help="sigma for gaussian kernel (only for gaussian method)")
    p.add_argument("--output", help="output json file to save homophily results", default=None)
    args = p.parse_args()
    results: List[Dict] = []
    for ds in args.datasets:
        res = process_dataset(args.datasets_root, ds, args.method, args.llm, args.sigma)
        results.append(res)
    pretty = json.dumps(results, indent=2, ensure_ascii=False)
    print(pretty)
    if args.output:
        outdir = os.path.dirname(args.output)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()