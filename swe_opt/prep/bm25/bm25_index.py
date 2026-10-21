import os
import sys
import json
import logging
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

from tqdm.auto import tqdm
from pyserini.search.lucene import LuceneSearcher

from swe_opt.prep.utils import is_optimization_candidate, ContextManager, clone_repo
from swe_opt.prep.data_loader import DatasetLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Query helpers
def build_query(row: Dict) -> str:
    """Build BM25 query from dataset row"""
    v = row.get("task")
    return str(v) if v else ""

def resolve_id(row: Dict) -> str:
    """Resolve instance identifier."""
    if "scenario_id" in row:
        return str(row["scenario_id"])
    raise KeyError("No instance ID field found")

# Index collection writers
def write_file_collection(repo_root: str, out_dir: Path) -> None:
    """Write file-level Python documents for Lucene indexing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "documents.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for dirpath, _, filenames in os.walk(repo_root):
            for fname in filenames:
                path = Path(dirpath) / fname
                rel = os.path.relpath(path, repo_root)
                if not is_optimization_candidate(rel):
                    continue
                try:
                    code = path.read_text(encoding="utf-8")
                except:
                    continue
                f.write(json.dumps({"id": rel, "contents": code}, ensure_ascii=False) + "\n")

# Index builder
def build_lucene_index(collection_dir: Path, index_dir: Path, threads: int = 4) -> None:
    """Build Lucene BM25 index (if not already built)."""
    if index_dir.exists() and any(index_dir.iterdir()):
        return

    index_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "pyserini.index",
        "--collection", "JsonCollection",
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        "--input", collection_dir.as_posix(),
        "--index", index_dir.as_posix(),
        "--storePositions",
        "--storeDocvectors",
        "--storeContents"
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Lucene index build failed\n{proc.stdout}\n{proc.stderr}")

# Search helper
def load_entity_meta(collection_dir: Path) -> dict:
    meta_path = collection_dir / "meta.jsonl"
    if not meta_path.exists():
        return {}

    meta_map = {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                meta_map[rec["id"]] = rec.get("meta", {})
    except Exception as e:
        logger.warning(f"Failed to read {meta_path}: {e}")

    return meta_map

def search_lucene_bm25(
    *, index_dir: Path, collection_dir: Path, query: str, k: int
) -> List[Dict]:
    searcher = LuceneSearcher(index_dir.as_posix())
    searcher.set_bm25(k1=1.0 , b=0.3) # BLUiR's k1=1.0, b=0.3 configuration discovered via grid search for code retrival
    hits = searcher.search(query, k=k)

    meta = load_entity_meta(collection_dir)

    results = []
    for hit in hits:
        hit_id = hit.docid
        doc = searcher.doc(hit_id)
        if not doc:
            continue

        results.append({
            "path": hit_id,
            "score": float(hit.score),
            "code": doc.contents(),
            "metadata": meta.get(hit_id, {})
        })

    return results

def build_bm25_retrieval_dataset(
    *, dataset_name_or_path: str, output_dir: str, top_k: int = 20
):
    loader = DatasetLoader(dataset_name_or_path, mode="raw")
    dataset = loader.data
    dataset_name = loader.name

    output_root = Path(output_dir) / dataset_name
    repos_root = Path(output_dir).parent / "repos"
    indexes_root = output_root / "lucene_indexes"
    collections_root = output_root / "lucene_collections"

    repos_root.mkdir(parents=True, exist_ok=True)
    indexes_root.mkdir(parents=True, exist_ok=True)
    collections_root.mkdir(parents=True, exist_ok=True)

    output_file = output_root / f"bm25_file_retrievals.jsonl"

    seen = set()
    with open(output_file, "w", encoding="utf-8") as f:
        for row in tqdm(dataset, desc="BM25 retrieval"):
            scenario_id = resolve_id(row)
            if scenario_id in seen:
                raise ValueError(f"Duplicate scenario_id detected: {scenario_id}")
            seen.add(scenario_id)

            repo_slug = row["repo"]
            commit = row["baseline_sha"]
            query = build_query(row)

            repo_dir = clone_repo(
                repo_slug=repo_slug,
                root_dir=repos_root,
                token=os.environ.get("GITHUB_TOKEN"),
                base_url=os.environ.get("REPO_BASE_URL", "https://github.com"),
            )

            index_key = f"{scenario_id}__{commit[:12]}"
            collection_dir = collections_root / index_key
            index_dir = indexes_root / index_key / "index"

            try:
                with ContextManager(repo_dir.as_posix(), commit):
                    
                    write_file_collection(repo_dir.as_posix(), collection_dir)
                    build_lucene_index(collection_dir, index_dir, threads=4)
                    hits = search_lucene_bm25(
                        index_dir=index_dir,
                        collection_dir=collection_dir,
                        query=query,
                        k=top_k)

            except Exception as e:
                logger.error(f"BM25 failed for {scenario_id}: {e}")
                hits = []

            record = {
                "scenario_id": scenario_id,
                "repo": repo_slug,
                "baseline_sha": commit,
                "query": query,
                "hits": hits,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Saved BM25 retrievals → {output_file}")

def main():
    parser = ArgumentParser(description="Merged Lucene BM25 retrieval dataset (per sample isolation)")

    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default="SWE-Opt/SWE-Opt",
        help="Path to base dataset (JSON / JSONL / HF dataset)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./retrieval_results_lucene",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of files retrieved per instance",
    )

    args = parser.parse_args()

    build_bm25_retrieval_dataset(
        dataset_name_or_path=args.dataset_name_or_path,
        output_dir=args.output_dir,
        top_k=args.top_k
    )

if __name__ == "__main__":
    main()
