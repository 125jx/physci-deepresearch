"""Baseline prediction generator for PhySciBench (`python -m eval.predict`).

Produces the `predictions.jsonl` that `eval.score` consumes: one JSON object per
line, `{"id": "physci-001", "response": "..."}`. This is the *model-under-test*
side of the pipeline (distinct from the LLM-as-judge configured via `JUDGE_LLM_*`).

It is a faithful, dependency-light port of the internal baseline runner
(`baseline/boyue_model.py`): for each benchmark item it builds a single user
message from the question plus any referenced figures/PDFs (base64-attached from
`<data-dir>/files/`), calls an OpenAI-compatible `chat.completions` endpoint with
bounded concurrency + retry, and streams results to disk.

This baseline is a convenience, not a requirement — you may generate
`predictions.jsonl` with any system as long as it matches the two-field schema
above and the `id`s line up with `physcibench.json`.

Model-under-test connection (CLI flag overrides env; env precedence shown):
    --model      / PREDICT_LLM_MODEL     > OPENAI_MODEL
    --base-url   / PREDICT_LLM_BASE_URL  > OPENAI_BASE_URL
    --api-key    / PREDICT_LLM_API_KEY   > OPENAI_API_KEY

Example:
    python -m eval.predict --data-dir PhySciBench --out predictions.jsonl \
        --model gpt-5.2 --concurrency 16

File attachment requires an endpoint that accepts PDF/image content parts; for a
text-only endpoint (or JSON-only download without `files/`) pass
`--no-attach-files` and only the question text is sent.
"""

import argparse
import asyncio
import base64
import json
import logging
import mimetypes
import os
import pathlib

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from .data import Sample, load_benchmark

logger = logging.getLogger("eval.predict")

# Exceptions worth retrying (transient / server-side); mirrors boyue_model.py.
_RETRYABLE = (APIError, RateLimitError, APITimeoutError)

_IMAGE_MIME_PREFIX = "image/"


def _env(*names: str, default: str | None = None) -> str | None:
    """Return the first set environment variable among `names` (precedence order)."""
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return default


def _encode_file(path: pathlib.Path) -> str:
    """Read `path` and return its base64-encoded contents."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_message_content(
    sample: Sample,
    files_dir: pathlib.Path,
    *,
    attach_files: bool,
    missing_warned: set[str],
):
    """Build the OpenAI `content` for one sample: question text + attached files.

    Returns a plain string when there is nothing to attach (maximally compatible
    with endpoints that reject content arrays), otherwise a list of content parts:
    PDFs become `type:"file"` parts and images become `type:"image_url"` parts,
    mirroring the internal baseline. Remote (`http(s)://`) references and files
    absent from `files_dir` are skipped with a warning — so the JSON-only dataset
    (no `files/`) still runs, just without attachments.
    """
    text = sample.raw_question or ""
    if not attach_files or not sample.files:
        return text

    parts: list[dict] = [{"type": "text", "text": text}]
    for file_name in sample.files:
        if file_name.startswith(("http://", "https://")):
            # A base LLM cannot fetch remote references; skip (documented behavior).
            logger.debug("Sample %s: skipping remote reference %s", sample.id, file_name)
            continue

        path = files_dir / file_name
        if not path.is_file():
            if file_name not in missing_warned:
                logger.warning(
                    "Sample %s: referenced file not found, skipping attachment: %s "
                    "(download files with `python scripts/download.py --with-files`)",
                    sample.id,
                    path,
                )
                missing_warned.add(file_name)
            continue

        b64 = _encode_file(path)
        mime, _ = mimetypes.guess_type(path.as_posix())
        mime = mime or "application/octet-stream"
        if mime.startswith(_IMAGE_MIME_PREFIX):
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        else:  # pdf and any other document
            parts.append(
                {
                    "type": "file",
                    "file": {"file_data": f"data:{mime};base64,{b64}", "filename": path.name},
                }
            )

    # Nothing attachable resolved -> fall back to plain text.
    return parts if len(parts) > 1 else text


async def _predict_one(
    client: AsyncOpenAI,
    model: str,
    sample: Sample,
    content,
    *,
    temperature: float | None,
    max_retries: int,
) -> str | None:
    """Call the model for one sample with exponential-backoff retry.

    Returns the response text, or `None` if all attempts failed (the caller then
    leaves the item out of the output so `--resume` can retry it later and the
    scorer reports it as missing rather than scoring a spurious error string).
    """
    messages = [{"role": "user", "content": content}]
    extra = {"temperature": temperature} if temperature is not None else {}

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            completion = await client.chat.completions.create(model=model, messages=messages, **extra)
            text = completion.choices[0].message.content
            if not text:
                logger.warning("Sample %s: model returned an empty response.", sample.id)
                return ""
            return text
        except _RETRYABLE as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            # 30s, 60s, 60s, ... (matches boyue_model.py's wait_exponential(min=30, max=60)).
            delay = min(30 * (2 ** (attempt - 1)), 60)
            logger.warning(
                "Sample %s: API error on attempt %d/%d (%s); retrying in %ds.",
                sample.id, attempt, max_retries, exc, delay,
            )
            await asyncio.sleep(delay)
        except Exception as exc:  # noqa: BLE001 - non-retryable; fail this sample only
            logger.error("Sample %s: non-retryable error: %s", sample.id, exc)
            return None

    logger.error("Sample %s: giving up after %d attempts (last error: %s).", sample.id, max_retries, last_exc)
    return None


def _load_done_ids(out_path: pathlib.Path) -> set[str]:
    """Return the set of sample ids already present in an existing `--out` file."""
    done: set[str] = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line)["id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


async def run_predictions(
    *,
    benchmark_path: str | pathlib.Path,
    out_path: str | pathlib.Path,
    files_dir: str | pathlib.Path,
    model: str,
    base_url: str | None,
    api_key: str | None,
    concurrency: int = 8,
    limit: int | None = None,
    attach_files: bool = True,
    temperature: float | None = None,
    max_retries: int = 4,
    timeout: float = 3600.0,
    resume: bool = False,
) -> dict:
    """Generate predictions for every benchmark sample and stream them to `out_path`.

    Results are appended as they complete (JSONL), so a crash keeps finished work;
    pass `resume=True` to skip ids already written. Returns a small run summary.
    """
    out_path = pathlib.Path(out_path)
    files_dir = pathlib.Path(files_dir)

    samples = load_benchmark(benchmark_path)
    if limit is not None:
        samples = samples[:limit]

    done_ids = _load_done_ids(out_path) if resume else set()
    todo = [s for s in samples if s.id not in done_ids]
    logger.info(
        "Loaded %d samples; %d already done, %d to predict (model=%s, attach_files=%s).",
        len(samples), len(done_ids), len(todo), model, attach_files,
    )

    client = AsyncOpenAI(api_key=api_key or "xxx", base_url=base_url, timeout=timeout)
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    missing_warned: set[str] = set()
    counters = {"written": 0, "failed": 0}

    # Append in resume mode (keep prior lines), truncate otherwise.
    mode = "a" if (resume and out_path.exists()) else "w"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, mode, encoding="utf-8") as out_f:

        async def worker(sample: Sample) -> None:
            content = build_message_content(
                sample, files_dir, attach_files=attach_files, missing_warned=missing_warned
            )
            async with semaphore:
                response = await _predict_one(
                    client, model, sample, content, temperature=temperature, max_retries=max_retries
                )
            if response is None:
                counters["failed"] += 1
                return
            async with write_lock:
                out_f.write(json.dumps({"id": sample.id, "response": response}, ensure_ascii=False) + "\n")
                out_f.flush()
                counters["written"] += 1
                done = counters["written"] + counters["failed"]
                if done % 10 == 0 or done == len(todo):
                    logger.info("Progress: %d/%d (%d failed).", done, len(todo), counters["failed"])

        try:
            await asyncio.gather(*(worker(s) for s in todo))
        finally:
            await client.close()

    summary = {
        "num_samples": len(samples),
        "num_skipped_resume": len(done_ids),
        "num_written": counters["written"],
        "num_failed": counters["failed"],
        "out_path": out_path.as_posix(),
    }
    if counters["failed"]:
        logger.warning(
            "%d sample(s) failed and were NOT written; re-run with --resume to retry them.",
            counters["failed"],
        )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhySciBench baseline prediction generator")
    parser.add_argument("--data-dir", default="PhySciBench", help="Data dir holding physcibench.json and files/ (default: PhySciBench)")
    parser.add_argument("--benchmark", default=None, help="Path to physcibench.json (default: <data-dir>/physcibench.json)")
    parser.add_argument("--out", default="predictions.jsonl", help="Output predictions path (default: predictions.jsonl)")
    parser.add_argument("--model", default=None, help="Model id (default: env PREDICT_LLM_MODEL / OPENAI_MODEL)")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (default: env PREDICT_LLM_BASE_URL / OPENAI_BASE_URL)")
    parser.add_argument("--api-key", default=None, help="API key (default: env PREDICT_LLM_API_KEY / OPENAI_API_KEY)")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent requests (default: 8)")
    parser.add_argument("--limit", type=int, default=None, help="Predict only the first N samples")
    parser.add_argument("--no-attach-files", dest="attach_files", action="store_false", help="Send question text only (no figures/PDFs)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (omitted from the request if unset)")
    parser.add_argument("--max-retries", type=int, default=4, help="Retry attempts per sample on transient API errors (default: 4)")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Per-request timeout in seconds (default: 3600)")
    parser.add_argument("--resume", action="store_true", help="Skip ids already present in --out and append")
    parser.set_defaults(attach_files=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s - %(message)s")
    args = _parse_args(argv)

    data_dir = pathlib.Path(args.data_dir)
    benchmark_path = pathlib.Path(args.benchmark) if args.benchmark else data_dir / "physcibench.json"

    model = args.model or _env("PREDICT_LLM_MODEL", "OPENAI_MODEL")
    if not model:
        raise SystemExit("No model configured. Pass --model or set PREDICT_LLM_MODEL / OPENAI_MODEL.")
    base_url = args.base_url or _env("PREDICT_LLM_BASE_URL", "OPENAI_BASE_URL")
    api_key = args.api_key or _env("PREDICT_LLM_API_KEY", "OPENAI_API_KEY")

    summary = asyncio.run(
        run_predictions(
            benchmark_path=benchmark_path,
            out_path=args.out,
            files_dir=data_dir / "files",
            model=model,
            base_url=base_url,
            api_key=api_key,
            concurrency=args.concurrency,
            limit=args.limit,
            attach_files=args.attach_files,
            temperature=args.temperature,
            max_retries=args.max_retries,
            timeout=args.timeout,
            resume=args.resume,
        )
    )
    logger.info("Done. %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
