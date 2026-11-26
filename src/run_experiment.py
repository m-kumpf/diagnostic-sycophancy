"""Collect one response per model, vignette, condition and repetition."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from .config import MODELS, ModelConfig, RunConfig
from .data_loader import Case, load_cases
from .llm_client import call_model
from .prompt_builder import CONDITIONS, Condition, build_prompt
from .response_parser import parse_first_response


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XLSX = PROJECT_ROOT / "case_vignettes.xlsx"


def per_call_seed(master_seed: int, model: str, case_id: str, condition: str, rep: int) -> int:
    h = hashlib.sha256(f"{master_seed}|{model}|{case_id}|{condition}|{rep}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def _output_path(results_dir: Path, model_name: str, pilot: bool) -> Path:
    suffix = ".pilot.jsonl" if pilot else ".jsonl"
    return results_dir / f"{model_name}{suffix}"


def _build_work_list(
    cases: list[Case], model_name: str, run_cfg: RunConfig,
) -> list[tuple[Case, Condition, int]]:
    if run_cfg.pilot:
        selected_cases = cases[: run_cfg.pilot_n_cases]
        n_reps = run_cfg.pilot_n_reps
    else:
        selected_cases = cases
        n_reps = run_cfg.n_reps

    work: list[tuple[Case, Condition, int]] = []
    for case in selected_cases:
        for condition in CONDITIONS:
            for rep in range(1, n_reps + 1):
                work.append((case, condition, rep))
    return work


def _already_done_keys(output_path: Path) -> set[tuple[str, str, int]]:
    if not output_path.exists():
        return set()
    keys: set[tuple[str, str, int]] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Corrupt line — ignore; it will be re-run.
                continue
            keys.add((rec["case_id"], rec["condition"], int(rec["repetition"])))
    return keys


async def _run_one(
    client: httpx.AsyncClient,
    model_cfg: ModelConfig,
    case: Case,
    condition: Condition,
    rep: int,
    run_cfg: RunConfig,
    semaphore: asyncio.Semaphore,
) -> dict:
    seed = per_call_seed(run_cfg.master_seed, model_cfg.name, case.case_id, condition, rep)
    rng = random.Random(seed)
    prompt, option_mapping, correct_letter, distractor_letter = build_prompt(case, condition, rng)

    result = await call_model(
        client, model_cfg, prompt, run_cfg.temperature, semaphore,
        max_retries=run_cfg.max_retries,
    )

    parse_res = parse_first_response(result.raw_response or "", option_mapping)
    parsed = parse_res.letter
    return {
        "call_id": str(uuid.uuid4()),
        "case_id": case.case_id,
        "condition": condition,
        "repetition": rep,
        "model": model_cfg.name,
        "model_id": model_cfg.model_id,
        "temperature": run_cfg.temperature,
        "option_mapping": option_mapping,
        "correct_letter": correct_letter,
        "distractor_letter": distractor_letter,
        "prompt": prompt,
        "raw_response": result.raw_response,
        "parsed_answer": parsed,
        "parse_method": parse_res.method,
        "is_correct": parsed == correct_letter,
        "chose_distractor": parsed == distractor_letter,
        "finish_reason": result.finish_reason,
        "attempts": result.attempts,
        "latency_ms": result.latency_ms,
        "error": result.error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
    }


async def run(model_name: str, run_cfg: RunConfig, xlsx_path: Path) -> None:
    if model_name not in MODELS:
        raise SystemExit(f"Unknown model {model_name!r}. Known: {sorted(MODELS)}")
    model_cfg = MODELS[model_name]

    cases = load_cases(xlsx_path)
    print(f"Loaded {len(cases)} cases from {xlsx_path.name}")

    work = _build_work_list(cases, model_name, run_cfg)
    total = len(work)

    output_path = _output_path(run_cfg.results_dir, model_name, run_cfg.pilot)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys = _already_done_keys(output_path)
    remaining = [(c, cond, rep) for (c, cond, rep) in work
                 if (c.case_id, cond, rep) not in done_keys]

    print(
        f"Model: {model_name} ({model_cfg.model_id})  |  "
        f"pilot={run_cfg.pilot}  |  total={total}  done={len(done_keys)}  "
        f"to-run={len(remaining)}  |  concurrency={run_cfg.concurrency}"
    )
    print(f"Writing to: {output_path}")

    if not remaining:
        print("Nothing to do. All work already completed.")
        return

    semaphore = asyncio.Semaphore(run_cfg.concurrency)
    timeout = httpx.Timeout(run_cfg.request_timeout_s)
    limits = httpx.Limits(max_connections=run_cfg.concurrency * 2)

    write_lock = asyncio.Lock()

    with output_path.open("a", encoding="utf-8") as out_f:
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:

            async def task(case: Case, condition: Condition, rep: int) -> None:
                rec = await _run_one(client, model_cfg, case, condition, rep, run_cfg, semaphore)
                async with write_lock:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()

            coros = [task(c, cond, rep) for (c, cond, rep) in remaining]
            await tqdm_asyncio.gather(*coros, desc=f"{model_name}", unit="call")

    _print_summary(output_path)


def _print_summary(output_path: Path) -> None:
    n_total = 0
    n_correct = 0
    n_chose_distractor = 0
    n_unparseable = 0
    n_errors = 0
    by_cond: dict[str, dict[str, int]] = {c: {"n": 0, "correct": 0, "distractor": 0, "unparseable": 0, "errors": 0}
                                          for c in ("neutral", "mild", "severe")}
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_total += 1
            cond = rec["condition"]
            d = by_cond.setdefault(cond, {"n": 0, "correct": 0, "distractor": 0, "unparseable": 0, "errors": 0})
            d["n"] += 1
            if rec.get("error"):
                n_errors += 1
                d["errors"] += 1
            if rec["parsed_answer"] == "unparseable":
                n_unparseable += 1
                d["unparseable"] += 1
            if rec["is_correct"]:
                n_correct += 1
                d["correct"] += 1
            if rec["chose_distractor"]:
                n_chose_distractor += 1
                d["distractor"] += 1

    print(f"\nSummary for {output_path.name}:")
    print(f"  total calls: {n_total}")
    if n_total:
        print(f"  accuracy:        {n_correct}/{n_total} = {n_correct / n_total:.1%}")
        print(f"  chose DD_a:      {n_chose_distractor}/{n_total} = {n_chose_distractor / n_total:.1%}")
        print(f"  unparseable:     {n_unparseable}/{n_total} = {n_unparseable / n_total:.1%}")
        print(f"  errors:          {n_errors}/{n_total} = {n_errors / n_total:.1%}")
    for cond in ("neutral", "mild", "severe"):
        d = by_cond[cond]
        if d["n"]:
            print(f"  [{cond:7}] n={d['n']:4}  acc={d['correct']/d['n']:.1%}  "
                  f"DD_a={d['distractor']/d['n']:.1%}  unparseable={d['unparseable']/d['n']:.1%}")


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the sycophancy experiment.")
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODELS),
        help="Model key from src/config.py MODELS (required).",
    )
    parser.add_argument("--pilot", action="store_true", help="Use the pilot case and repetition counts")
    parser.add_argument("--concurrency", type=int, default=None, help="Override RunConfig.concurrency")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX, help="Path to the case xlsx")
    parser.add_argument("--out", type=Path, default=None, help="Override results dir")
    args = parser.parse_args(argv)

    base_cfg = RunConfig()
    run_cfg = RunConfig(
        n_reps=base_cfg.n_reps,
        temperature=base_cfg.temperature,
        concurrency=args.concurrency if args.concurrency is not None else base_cfg.concurrency,
        pilot=args.pilot,
        pilot_n_cases=base_cfg.pilot_n_cases,
        pilot_n_reps=base_cfg.pilot_n_reps,
        results_dir=(args.out if args.out is not None else PROJECT_ROOT / base_cfg.results_dir),
        master_seed=base_cfg.master_seed,
        max_retries=base_cfg.max_retries,
        request_timeout_s=base_cfg.request_timeout_s,
    )

    try:
        asyncio.run(run(args.model, run_cfg, args.xlsx))
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
