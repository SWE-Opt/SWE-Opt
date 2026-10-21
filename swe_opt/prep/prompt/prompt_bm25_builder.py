import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Any

from swe_opt.prep.data_loader import DatasetLoader
from swe_opt.prep.prompt.prompt_tokenizer import (
    get_tokenizer,
    count_tokens,
    truncate_to_token_budget,
)
from swe_opt.prep.prompt.prompt_frame import (
    build_prompt_prefix,
    build_prompt_suffix
)
from swe_opt.prep.utils import normalize_prompt_dataset_meta, prompt_dataset_filename

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BM25PromptBuilder:
    """
    BM25PromptBuilder assembles model prompts from precomputed BM25 retrievals
    (see ``swe_opt/prep/bm25/bm25_index.py``).         
    It performs no retrieval and no static analysis at prompt-build time.

    Core contract
    -------------
    - Input: one dataset row plus its BM25 hits.
    - Output: a single prompt string composed of:
      1) a fixed prefix (task + instructions),
      2) BM25-retrieved files, and
      3) a fixed suffix (response format + constraints),
      all subject to a global token budget.

    Notes
    -----
    - Which entities appear in the prompt is entirely determined upstream
      by the BM25 pipeline. This builder only formats and packs them.
    - This builder only formats and packs the provided context under
        the specified token budget.
    """

    def __init__(
        self,
        row: Dict[str, Any],
        hits: List[Dict[str, Any]],
        *,
        max_tokens: int = 8000,
    ) -> None:

        self.row = row
        self.hits = hits or []
        self.max_tokens = int(max_tokens)

    def _file_meta(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        """Return file-level metadata (e.g. module-level info) from a BM25 hit."""

        md = hit.get("metadata")
        if isinstance(md, dict):
            return md.get("file_meta") or {}
        return {}

    def _score(self, hit: Dict[str, Any]) -> float:
        """Return the BM25 score for a hit, or -inf if it is missing/invalid."""

        s = hit.get("score")
        try:
            return float(s) if s is not None else float("-inf")
        except Exception:
            return float("-inf")

    def _file_path(self, hit: Dict[str, Any]) -> str:
        """Extract the normalized file path for a hit from object or file metadata."""

        fm = self._file_meta(hit)
        return str(fm.get("file_path") or "").strip()
    
    def _build_file_blocks(self, *, tokenizer_spec, remaining: int, sep_cost: int) -> List[str]:
        """
        Select and format full-file hits ordered by score under a token budget.
        """
        
        blocks: List[str] = []
        used = 0

        for h in sorted(self.hits, key=self._score, reverse=True):
            fp = str(h.get("path") or "").strip()
            code = (h.get("code") or "").rstrip("\n")
            if not fp or not code:
                continue

            body = f"```python\n{code}\n```"
            start_marker = f"[start of {fp}]\n"
            end_marker = f"\n[end of {fp}]\n"
            block = start_marker + body + end_marker

            block_cost = count_tokens(block, tokenizer_spec)
            extra = block_cost + (sep_cost if blocks else 0)  # charge join cost
            if used + extra <= remaining:
                blocks.append(block)
                used += extra
                continue

            fence = "```python\n"
            fence_end = "\n```"
            overhead = count_tokens(start_marker + end_marker + fence + fence_end, tokenizer_spec)
            budget_for_code = remaining - used - overhead - (sep_cost if blocks else 0)
            if budget_for_code <= 0:
                break

            truncated_code = truncate_to_token_budget(code, tokenizer_spec, budget_for_code)
            blocks.append(start_marker + fence + truncated_code.rstrip("\n") + fence_end + end_marker)
            break

        return blocks
    
    def build_prompt_with_budget(self, *, tokenizer_spec) -> str:
        """Construct a BM25-based prompt (prefix + context + suffix) under the token budget."""

        prefix = build_prompt_prefix(self.row)
        suffix = build_prompt_suffix()

        sep = "\n\n"
        sep_cost = count_tokens(sep, tokenizer_spec)

        remaining = (self.max_tokens
                    - count_tokens(prefix, tokenizer_spec)
                    - count_tokens(suffix, tokenizer_spec)
                    - 2 * sep_cost)
        if remaining <= 0:
            raise ValueError("Token budget too small for prompt skeleton")

        
        blocks = self._build_file_blocks(tokenizer_spec=tokenizer_spec, remaining=remaining, sep_cost=sep_cost)

        if not blocks:
            raise ValueError("No BM25 blocks fit into token budget")

        context = "\n".join(blocks)
        prompt = "\n".join([prefix, context, suffix])
        return prompt

def build_bm25_prompt_dataset(
    *,
    dataset_name_or_path: str,
    bm25_retrievals_path: str,
    output_path: str,
    tokenizer_name: str = "cl100k_base",
    max_tokens: int = 100_000):
    """Build BM25-based prompts for each dataset row and write them to a JSON file."""

    tokenizer_spec = get_tokenizer(tokenizer_name)
    loader = DatasetLoader(dataset_name_or_path, mode="raw")

    bm25_by_id: Dict[str, Dict[str, Any]] = {}
    with open(bm25_retrievals_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("scenario_id")
            if sid is None:
                continue
            bm25_by_id[str(sid)] = rec

    rows_out: List[Dict[str, Any]] = []

    for row in loader:
        scenario_id = row.get("scenario_id")
        if scenario_id is None:
            logger.warning("Row without scenario_id encountered, skipping")
            continue

        scenario_id = str(scenario_id)
        bm25_rec = bm25_by_id.get(scenario_id)
        if not bm25_rec:
            logger.warning(f"{scenario_id}: no BM25 retrieval record found, skipping")
            continue

        hits = bm25_rec.get("hits") or []
        if not hits:
            logger.warning(f"{scenario_id}: BM25 hits list is empty, skipping")
            continue

        builder = BM25PromptBuilder(
            row=row,
            hits = hits,
            max_tokens=max_tokens)

        try:
            prompt = builder.build_prompt_with_budget(tokenizer_spec=tokenizer_spec)
        except Exception as e:
            logger.warning(f"{scenario_id}: failed to build BM25 prompt: {e}")
            continue
        
        prompt_tokens = count_tokens(prompt, tokenizer_spec)
        logger.info(f"{scenario_id}: prompt_tokens={prompt_tokens} (max={max_tokens})")
        if prompt_tokens > max_tokens:
            logger.warning(f"{scenario_id}: prompt exceeds budget! {prompt_tokens} > {max_tokens}")

        out_row = dict(row)
        out_row["text"] = prompt
        out_row.pop("oracle_context", None)
        rows_out.append(out_row)

    meta = normalize_prompt_dataset_meta({
        "builder": "bm25",
        "tokenizer": tokenizer_name,
        "max_tokens": int(max_tokens),
        "dataset_path": dataset_name_or_path,
        "bm25_retrievals_path": bm25_retrievals_path,
    })

    output_obj = {"meta": meta, "samples": rows_out}

    if output_path is None:
        ds_path = Path(dataset_name_or_path).resolve()
        out_dir = ds_path.parent / "prompt_dataset_bm25"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / prompt_dataset_filename(meta))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Saved {len(rows_out)} BM25 prompts → {out_path}")
    return out_path

def main() -> None:
    parser = ArgumentParser(description="Generate BM25-based prompts with token budget")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="SWE-Opt/SWE-Opt",
        help="Path to base dataset")
    
    parser.add_argument(
        "--bm25_retrievals_path",
        type=str,
        default="retrieval_results_lucene/SWE-Opt/bm25_file_retrievals.jsonl",
        help="Path to BM25 retrieval JSONL")
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write dataset with BM25 prompts (JSON)")
    
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        choices=[
        # tiktoken / cl100k
        "cl100k", "cl100k_base",
        # HF aliases
        "qwen3.5", "deepseek", "phi4", "glm5.1",
        "kimik2.5", "minimaxm2.5", "minimaxm2.7"],
        help=(
        "Tokenizer to use. Fixed choices: cl100k, cl100k_base, llama3, llama3.1, "
        "qwen3.5, deepseek, phi4, glm5.1, kimik2.5, minimaxm2.5, minimaxm2.7. "
        "Also accepts prefixed forms: hf:<model_id>, gemini:<model_name>, claude:<model_name>."))
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        required=True,
        help="Max total tokens per prompt")

    args = parser.parse_args()

    build_bm25_prompt_dataset(
        dataset_name_or_path=args.dataset_path,
        bm25_retrievals_path=args.bm25_retrievals_path,
        output_path=args.output_path,
        tokenizer_name=args.tokenizer,
        max_tokens=args.max_tokens)

if __name__ == "__main__":
    main()
