import os
import sys
import argparse
import json
from typing import Union, Dict, Optional, List

import numpy as np
import torch
from scipy import sparse
import pandas as pd
import glob
import re

# make repo importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.graph.modularity import AdjacencyMatrixCreatorForModularity


def _to_numpy_adj(adj: Union[np.ndarray, torch.Tensor, sparse.spmatrix]) -> np.ndarray:
    if isinstance(adj, torch.Tensor):
        return adj.detach().cpu().numpy()
    if sparse.issparse(adj):
        return adj.toarray()
    if isinstance(adj, np.ndarray):
        return adj
    raise TypeError(f"Unsupported adjacency type: {type(adj)}")


def homophily_per_node(adj: Union[np.ndarray, torch.Tensor, sparse.spmatrix],
                       labels: Union[np.ndarray, list, torch.Tensor],
                       weighted: bool = False) -> np.ndarray:
    A = _to_numpy_adj(adj).copy()
    # keep original label types (strings or ints)
    labels = np.array(labels, dtype=object)
    if A.shape[0] != labels.shape[0]:
        raise ValueError("Adjacency size and labels length must match")

    np.fill_diagonal(A, 0)
    n = A.shape[0]
    hom = np.full(n, np.nan, dtype=float)

    if weighted:
        row_sums = A.sum(axis=1)
        for i in range(n):
            total = row_sums[i]
            if total == 0:
                continue
            same_mask = (labels == labels[i]).astype(float)
            same_weight = (A[i] * same_mask).sum()
            hom[i] = same_weight / total
    else:
        for i in range(n):
            neighbors = np.nonzero(A[i])[0]
            if neighbors.size == 0:
                continue
            same_count = (labels[neighbors] == labels[i]).sum()
            hom[i] = same_count / neighbors.size

    return hom


def global_homophily(adj, labels, weighted=True, ignore_isolated=True) -> float:
    node_h = homophily_per_node(adj, labels, weighted=weighted)
    return float(np.nanmean(node_h)) if ignore_isolated else float(np.nanmean(node_h))


def class_homophily(adj, labels, weighted=False) -> Dict[str, float]:
    node_h = homophily_per_node(adj, labels, weighted=weighted)
    labels = np.array(labels, dtype=object)
    classes = np.unique(labels)
    out: Dict[str, float] = {}
    for c in classes:
        mask = labels == c
        vals = node_h[mask]
        out[str(c)] = float(np.nanmean(vals)) if vals.size > 0 else float("nan")
    return out


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


def process_dataset(root: str, name: str, method: str, llm: str, sigma: float, weighted: bool):
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
    labels = np.concatenate([df_train["label"].values, df_val["label"].values, df_test["label"].values]).astype(object)

    node_h = homophily_per_node(adj, labels, weighted=weighted)
    return {
        "dataset": name,
        "method": method,
        "global_homophily": global_homophily(adj, labels, weighted=weighted),
        "class_homophily": class_homophily(adj, labels, weighted=weighted),
        "node_homophily_sample": node_h[:min(50, len(node_h))].tolist()
    }


def main():
    p = argparse.ArgumentParser(description="Compute homophily for SBERT-based cosine or gaussian graphs (basic graphs only).")
    p.add_argument("--method", choices=["from_sbert_embeddings_cosine", "from_sbert_embeddings_gaussian"], required=True)
    p.add_argument("--datasets-root", default="datasets", help="root folder where dataset subfolders or csvs live")
    p.add_argument("--datasets", nargs="+", default=["mr", "20ng", "r8", "r52", "ohsumed"], help="dataset names to analyze (default common set).")
    p.add_argument("--llm", default="all-mpnet-base-v2")
    p.add_argument("--sigma", type=float, default=1.0, help="sigma for gaussian kernel (only for gaussian method)")
    p.add_argument("--weighted", action="store_true", help="compute weighted homophily (default is unweighted)", default=True)
    p.add_argument("--output", help="output json file to save homophily results", default=None)
    args = p.parse_args()

    results: List[Dict] = []
    for ds in args.datasets:
        res = process_dataset(args.datasets_root, ds, args.method, args.llm, args.sigma, args.weighted)
        results.append(res)

    print(json.dumps(results, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
# filepath: /home/aditya/Modified_TextGCN/Modularity-TextGCN/scripts/homophily.py
import os
import sys
import argparse
import json
from typing import Union, Dict, Optional, List

import numpy as np
import torch
from scipy import sparse
import pandas as pd
import glob
import re

# make repo importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.graph.modularity import AdjacencyMatrixCreatorForModularity


def _to_numpy_adj(adj: Union[np.ndarray, torch.Tensor, sparse.spmatrix]) -> np.ndarray:
    if isinstance(adj, torch.Tensor):
        return adj.detach().cpu().numpy()
    if sparse.issparse(adj):
        return adj.toarray()
    if isinstance(adj, np.ndarray):
        return adj
    raise TypeError(f"Unsupported adjacency type: {type(adj)}")


def homophily_per_node(adj: Union[np.ndarray, torch.Tensor, sparse.spmatrix],
                       labels: Union[np.ndarray, list, torch.Tensor],
                       weighted: bool = False) -> np.ndarray:
    A = _to_numpy_adj(adj).copy()
    # keep original label types (strings or ints)
    labels = np.array(labels, dtype=object)
    if A.shape[0] != labels.shape[0]:
        raise ValueError("Adjacency size and labels length must match")

    np.fill_diagonal(A, 0)
    n = A.shape[0]
    hom = np.full(n, np.nan, dtype=float)

    if weighted:
        row_sums = A.sum(axis=1)
        for i in range(n):
            total = row_sums[i]
            if total == 0:
                continue
            same_mask = (labels == labels[i]).astype(float)
            same_weight = (A[i] * same_mask).sum()
            hom[i] = same_weight / total
    else:
        for i in range(n):
            neighbors = np.nonzero(A[i])[0]
            if neighbors.size == 0:
                continue
            same_count = (labels[neighbors] == labels[i]).sum()
            hom[i] = same_count / neighbors.size

    return hom


def global_homophily(adj, labels, weighted=False, ignore_isolated=True) -> float:
    node_h = homophily_per_node(adj, labels, weighted=weighted)
    return float(np.nanmean(node_h)) if ignore_isolated else float(np.nanmean(node_h))


def class_homophily(adj, labels, weighted=False) -> Dict[str, float]:
    node_h = homophily_per_node(adj, labels, weighted=weighted)
    labels = np.array(labels, dtype=object)
    classes = np.unique(labels)
    out: Dict[str, float] = {}
    for c in classes:
        mask = labels == c
        vals = node_h[mask]
        out[str(c)] = float(np.nanmean(vals)) if vals.size > 0 else float("nan")
    return out


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


def process_dataset(root: str, name: str, method: str, llm: str, sigma: float, weighted: bool):
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
    labels = np.concatenate([df_train["label"].values, df_val["label"].values, df_test["label"].values]).astype(object)

    node_h = homophily_per_node(adj, labels, weighted=weighted)
    return {
        "dataset": name,
        "method": method,
        "global_homophily": global_homophily(adj, labels, weighted=weighted),
        "class_homophily": class_homophily(adj, labels, weighted=weighted),
        "node_homophily_sample": node_h[:min(50, len(node_h))].tolist()
    }


def main():
    p = argparse.ArgumentParser(description="Compute homophily for SBERT-based cosine or gaussian graphs (basic graphs only).")
    p.add_argument("--method", choices=["from_sbert_embeddings_cosine", "from_sbert_embeddings_gaussian"], required=True)
    p.add_argument("--datasets-root", default="datasets", help="root folder where dataset subfolders or csvs live")
    p.add_argument("--datasets", nargs="+", default=["mr", "20ng", "r8", "r52"], help="dataset names to analyze (default common set).")
    p.add_argument("--llm", default="all-mpnet-base-v2")
    p.add_argument("--sigma", type=float, default=1.0, help="sigma for gaussian kernel (only for gaussian method)")
    p.add_argument("--weighted", action="store_true", help="compute weighted homophily (default is unweighted)", default=False)
    p.add_argument("--output", help="output json file to save homophily results", default=None)
    args = p.parse_args()

    results: List[Dict] = []
    for ds in args.datasets:
        res = process_dataset(args.datasets_root, ds, args.method, args.llm, args.sigma, args.weighted)
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