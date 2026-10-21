from typing import Dict, List, Any
from pathlib import Path
import json
import logging
from argparse import ArgumentParser

from swe_opt.prep.data_loader import *
from swe_opt.prep.prompt.prompt_tokenizer import get_tokenizer, count_tokens, truncate_to_token_budget
from swe_opt.prep.utils import normalize_prompt_dataset_meta

from swe_opt.prep.prompt.prompt_frame import (
    build_prompt_prefix,
    build_prompt_suffix
)

logger = logging.getLogger(__name__) 
logging.basicConfig(level=logging.INFO)

class OraclePromptBuilder:
    """
        OraclePromptBuilder assembles model prompts from precomputed oracle context.
        It performs no retrieval and no static analysis at prompt-build time.

        Core contract
        -------------
        - Input: one dataset row containing ``oracle_context``.
        - Output: a single prompt string composed of:
        1) a fixed prefix (task + instructions),
        2) file context, and
        3) a fixed suffix (response format + constraints),
        all subject to a global token budget.
        
        Notes
        -----
        - This builder only formats and packs the provided context under
        the specified token budget.
    """

    def __init__(
        self,
        row: Dict[str, Any],
        *,
        max_tokens: int = 8000
    ):
        self.row = row
        self.max_tokens = max_tokens
        baseline_ctx = row.get("oracle_context") or {}
        self.file_contexts = baseline_ctx.get("file_context", [])

    def build_file_context(self) -> List[Dict[str, Any]]:
        sections: List[Dict[str, Any]] = []

        for file_unit in self.file_contexts:
            file_path = file_unit.get("file_path")
            raw_code = file_unit.get("raw_code")

            if not isinstance(raw_code, str) or not raw_code:
                continue
            raw_code = raw_code.rstrip("\n")

            parts: List[str] = []

            parts.append("### Full file context")
            parts.append(f"```python\n{raw_code}\n```")

            sections.append({"file_path": file_path, "text": "\n".join(parts)})

        return sections

    def build_prompt_with_budget(
        self,
        *,
        tokenizer_spec,
    ) -> str:
        """
        Build prompt (prefix + context + suffix).
        """

        sections = self.build_file_context()
        if not sections:
            raise ValueError("no context sections generated")

        prefix = build_prompt_prefix(self.row)
        suffix = build_prompt_suffix()
        sep = "\n\n"

        prepared_chunks: List[Dict[str, str]] = []

        for sec in sections:
            path = sec.get("file_path")
            code = sec.get("text")

            if not path or code is None:
                continue

            code_str = str(code).rstrip()
            if not code_str:
                continue

            start_marker = f"[start of {path}]\n"
            end_marker = f"\n[end of {path}]\n"
            full_chunk = start_marker + code_str + end_marker

            prepared_chunks.append(
                {
                    "file_path": path,
                    "start_marker": start_marker,
                    "end_marker": end_marker,
                    "code_str": code_str,
                    "full_chunk": full_chunk,
                }
            )

        if not prepared_chunks:
            raise ValueError("No code fits into prompt")

        # Phase 1: try the full prompt first

        full_context = sep.join(chunk["full_chunk"] for chunk in prepared_chunks)
        full_prompt = sep.join([prefix, full_context, suffix])
        tokens = count_tokens(full_prompt, tokenizer_spec)
        if tokens <= self.max_tokens:
            return full_prompt

        # Phase 2: budget-aware packing
        sep_cost = count_tokens(sep, tokenizer_spec)

        # prefix <sep> context <sep> suffix  => 2 fixed separators
        fixed_tokens = (
            count_tokens(prefix, tokenizer_spec)
            + count_tokens(suffix, tokenizer_spec)
            + 2 * sep_cost
        )

        remaining = self.max_tokens - fixed_tokens
        if remaining <= 0:
            raise ValueError("Token budget too small even for prefix+suffix")

        selected_chunks: List[str] = []
        used = 0

        for chunk in prepared_chunks:
            full_chunk = chunk["full_chunk"]
            join_cost = sep_cost if selected_chunks else 0
            chunk_cost = count_tokens(full_chunk, tokenizer_spec)
            extra = join_cost + chunk_cost

            # Case 1: whole chunk fits
            if used + extra <= remaining:
                selected_chunks.append(full_chunk)
                used += extra
                continue

            # Case 2: only part of this final chunk can fit
            budget_for_this_chunk = remaining - used - join_cost
            if budget_for_this_chunk <= 0:
                break

            start_marker = chunk["start_marker"]
            end_marker = chunk["end_marker"]
            code_str = chunk["code_str"]

            marker_cost = (
                count_tokens(start_marker, tokenizer_spec)
                + count_tokens(end_marker, tokenizer_spec)
            )
            budget_for_code = budget_for_this_chunk - marker_cost
            if budget_for_code <= 0:
                break

            truncated_code = truncate_to_token_budget(
                code_str,
                tokenizer_spec,
                budget_for_code,
            ).rstrip()

            # avoid ending on a broken last line
            if "\n" in truncated_code:
                truncated_code = truncated_code.rsplit("\n", 1)[0].rstrip()

            if truncated_code:
                selected_chunks.append(start_marker + truncated_code + end_marker)

            break

        if not selected_chunks:
            raise ValueError("No code fits into token budget")

        context = sep.join(selected_chunks)
        prompt = sep.join([prefix, context, suffix])

        # Final safety check
        if count_tokens(prompt, tokenizer_spec) > self.max_tokens:
            prompt = truncate_to_token_budget(prompt, tokenizer_spec, self.max_tokens).rstrip()

        return prompt

def build_output_filename(
    *,
    tokenizer_name: str,
    max_tokens: int,
    prefix: str = "dataset") -> str:

    """Return an output filename that encodes the prompt configuration."""

    return (
        f"{prefix}"
        f"__tok={tokenizer_name}"
        f"__max={int(max_tokens)}"
        f".json"
    )

def build_prompt_dataset(
    *,
    dataset_name_or_path: str,
    output_path: str,
    tokenizer_name: str = "cl100k_base",
    max_tokens: int = 8000):

    """Build oracle prompts for each dataset row and write them to a JSON file."""

    tokenizer_spec = get_tokenizer(tokenizer_name)
    loader = DatasetLoader(dataset_name_or_path)

    rows_out: List[Dict[str, Any]] = []
    truncation_filtered = []

    for row in loader:
        scenario_id = row.get("scenario_id")
        ctx = row.get("oracle_context") or {}
        if not ctx:
            logger.warning(f"{scenario_id}: no usable context")
            continue

        prompter = OraclePromptBuilder(
            row,
            max_tokens=max_tokens)
        try:
            prompt = prompter.build_prompt_with_budget(
                        tokenizer_spec=tokenizer_spec)

        except Exception as e:
            logger.warning(f"{scenario_id}: {e}")
            continue

        prompt_tokens = count_tokens(prompt, tokenizer_spec)
        logger.info(f"{scenario_id}: prompt_tokens={prompt_tokens}")
        out_row = dict(row)
        out_row["text"] = prompt

        # Remove heavy oracle/baseline context from final dataset
        out_row.pop("oracle_context", None)

        rows_out.append(out_row)
    
    meta = normalize_prompt_dataset_meta({
        "builder": "oracle",
        "tokenizer": tokenizer_name,
        "max_tokens": int(max_tokens),
        "dataset_path": dataset_name_or_path,
        "truncation_filtered_count": len(truncation_filtered),
        "truncation_filtered": truncation_filtered, 
    })

    output_obj = {"meta": meta, "samples": rows_out}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(rows_out)} prompts → {output_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate oracle prompts with token budget")

    # Core arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="SWE-Opt/SWE-Opt",
        help="Path to base dataset")
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="out",
        help="Output JSON path. If omitted, it is auto-generated under <dataset_parent>/prompt_dataset/.")
    
    parser.add_argument(
        "--tokenizer",
        type=str,
        #required=True,
        choices=[
        # tiktoken / cl100k
        "cl100k", "cl100k_base",
        # HF aliases
        "qwen3.5", "deepseek", "phi4", "glm5.1",
        "kimik2.5", "minimaxm2.5", "minimaxm2.7"],
        default="cl100k_base",
        help=(
        "Tokenizer to use. Fixed choices: cl100k, cl100k_base, llama3, llama3.1, "
        "qwen3.5, deepseek, phi4, glm5.1, kimik2.5, minimaxm2.5, minimaxm2.7. "
        "Also accepts prefixed forms: hf:<model_id>, gemini:<model_name>, claude:<model_name>."))
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        #required=True,
        default=8000,
        help="Max total tokens per prompt")

    args = parser.parse_args()

    # --- auto output naming in the same style as BM25 + fixed folder name ---
    if args.output_path is None:
        ds_path = Path(args.dataset_path).resolve()
        parent = ds_path.parent

        out_dir = parent / "prompt_dataset_oracle"
        out_dir.mkdir(parents=True, exist_ok=True)

        fname = build_output_filename(
            tokenizer_name=args.tokenizer,
            max_tokens=args.max_tokens,
            prefix="dataset",
        )
        args.output_path = str(out_dir / fname)

    build_prompt_dataset(
        dataset_name_or_path=args.dataset_path,
        output_path=args.output_path,
        tokenizer_name=args.tokenizer,
        max_tokens=args.max_tokens
    )