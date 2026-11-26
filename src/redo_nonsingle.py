"""Repeat format-ineligible calls with their recorded prompt, options and temperature."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from .config import MODELS, RunConfig
from .export_csv import convert as export_csv
from .llm_client import call_model
from .response_parser import parse_response_detailed
from .run_experiment import DEFAULT_XLSX, PROJECT_ROOT, per_call_seed
from .data_loader import load_cases
from .prompt_builder import build_prompt


def _is_clean(rec: dict) -> bool:
    return rec.get("parse_method") == "single_letter"


async def _retry_cell(client, model_cfg, case, rec, run_cfg, sem, max_retries):
    """Retry one cell until the model returns a single letter or we cap out."""
    condition = rec["condition"]
    rep = rec["repetition"]
    seed = per_call_seed(run_cfg.master_seed, model_cfg.name, case.case_id, condition, rep)
    rng = random.Random(seed)
    prompt, option_mapping, correct_letter, distractor_letter = build_prompt(case, condition, rng)
    if (prompt, option_mapping, correct_letter, distractor_letter) != (
        rec["prompt"], rec["option_mapping"], rec["correct_letter"], rec["distractor_letter"]
    ):
        raise ValueError(f"Prompt or options differ from the recorded call {rec['call_id']}")

    last_result = None
    last_parse = None
    for attempt in range(1, max_retries + 1):
        res = await call_model(
            client, model_cfg, rec["prompt"], rec["temperature"], sem,
            max_retries=run_cfg.max_retries,
        )
        parse = parse_response_detailed(res.raw_response or "", option_mapping)
        last_result, last_parse = res, parse
        if parse.method == "single_letter":
            break

    parsed = last_parse.letter
    return {
        **rec,
        "raw_response": last_result.raw_response,
        "parsed_answer": parsed,
        "parse_method": last_parse.method,
        "is_correct": parsed == correct_letter,
        "chose_distractor": parsed == distractor_letter,
        "finish_reason": last_result.finish_reason,
        "attempts": last_result.attempts,
        "latency_ms": last_result.latency_ms,
        "error": last_result.error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retries_used": attempt,
        "forced_single_letter": True,
    }


async def main_async(model_name: str, max_retries: int, concurrency: int, xlsx_path: Path, results_dir=None):
    if model_name not in MODELS:
        raise SystemExit(f"Unknown model {model_name!r}. Known: {sorted(MODELS)}")
    model_cfg = MODELS[model_name]
    run_cfg = RunConfig(concurrency=concurrency)

    if max_retries < 1 or concurrency < 1:
        raise ValueError("Retry and concurrency counts must be positive")
    directory = Path(results_dir) if results_dir is not None else PROJECT_ROOT / run_cfg.results_dir
    jsonl_path = directory / f"{model_name}.jsonl"
    if not jsonl_path.exists():
        raise SystemExit(f"Not found: {jsonl_path}")

    records = [json.loads(l) for l in jsonl_path.open("r", encoding="utf-8") if l.strip()]
    targets = [(i, r) for i, r in enumerate(records) if not _is_clean(r)]
    print(f"Model: {model_name} | total records: {len(records)} | "
          f"non-single-letter targets: {len(targets)} | max-retries: {max_retries}")
    if not targets:
        print("Nothing to do.")
        return

    cases_by_id = {c.case_id: c for c in load_cases(xlsx_path)}

    sem = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(run_cfg.request_timeout_s)
    limits = httpx.Limits(max_connections=concurrency * 2)

    async def task(i, rec):
        case = cases_by_id[rec["case_id"]]
        new_rec = await _retry_cell(client, model_cfg, case, rec, run_cfg, sem, max_retries)
        records[i] = new_rec
        return new_rec

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        coros = [task(i, r) for i, r in targets]
        results = await tqdm_asyncio.gather(*coros, desc=f"{model_name}", unit="cell")

    tmp = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, jsonl_path)

    csv_path = jsonl_path.with_suffix(".csv")
    n_csv = export_csv(jsonl_path, csv_path, include_prompt=False)

    now_single = sum(1 for r in results if r["parse_method"] == "single_letter")
    still_non = len(results) - now_single
    total_api_calls = sum(r["retries_used"] for r in results)
    print(f"\nDone. {now_single}/{len(results)} now single_letter; "
          f"{still_non} still non-single after {max_retries} retries.")
    print(f"Total API calls made: {total_api_calls}.")
    print(f"Re-wrote {jsonl_path.name} ({len(records)} lines) and {csv_path.name} ({n_csv} rows).")


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(description="Re-run non-single-letter cells until clean.")
    parser.add_argument("--model", required=True, choices=sorted(MODELS),
                        help="Model key from src/config.py MODELS")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Maximum format re-queries per response (default: 5)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--out", type=Path, help="Directory containing the collection JSONL files.")
    args = parser.parse_args(argv)

    asyncio.run(main_async(args.model, args.max_retries, args.concurrency, args.xlsx, args.out))


if __name__ == "__main__":
    main()
