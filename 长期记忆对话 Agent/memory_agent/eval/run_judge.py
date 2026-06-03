import argparse
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from memory_agent.eval.core.llm_client import LLMClient
from memory_agent.eval.core.metrics import exact_match, f1_score


JUDGE_SYSTEM = (
    "You are a strict but fair grading assistant. You grade whether a predicted answer "
    "captures the same information as the reference answer. Output one JSON object only."
)

JUDGE_PROMPT = """Grade the prediction against the reference.

Question: {question}
Reference answer: {reference}
Predicted answer: {prediction}

Rubric:
- CORRECT: same key information as the reference.
- PARTIAL: right topic and partial overlap, but not fully accurate.
- WRONG: missing, contradictory, hallucinated, or off topic.

Respond with JSON only:
{{"reasoning": "<one short sentence>", "label": "CORRECT" | "PARTIAL" | "WRONG"}}"""

LABEL_SCORE = {"CORRECT": 1.0, "PARTIAL": 0.5, "WRONG": 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge prediction files with an OpenAI-compatible LLM.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--judge_base_url", default=None)
    parser.add_argument("--judge_model", default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.judge_base_url:
        os.environ["LLM_BASE_URL"] = args.judge_base_url
    if args.judge_model:
        os.environ["LLM_MODEL"] = args.judge_model

    pred_path = Path(args.predictions)
    out_path = Path(args.output) if args.output else pred_path.with_name("results_judge.json")
    with pred_path.open(encoding="utf-8") as f:
        predictions = json.load(f)
    if args.limit:
        predictions = predictions[: args.limit]

    client = LLMClient(temperature=0.0)
    print(f"[Judge] model={client.model} url={client.base_url} workers={args.num_workers} n={len(predictions)}")
    started = time.perf_counter()

    graded = [None] * len(predictions)
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(judge_one, client, item): i for i, item in enumerate(predictions)}
        for done, future in enumerate(as_completed(futures), start=1):
            idx = futures[future]
            try:
                graded[idx] = future.result()
            except Exception as exc:
                graded[idx] = fallback_grade(predictions[idx], f"judge_call_failed: {exc}")
            if done % 20 == 0 or done == len(predictions):
                elapsed = time.perf_counter() - started
                print(f"  judged {done}/{len(predictions)} ({elapsed:.1f}s)")

    results = aggregate(graded, client, str(pred_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print_table(results)
    print(f"Saved -> {out_path}")


def judge_one(client: LLMClient, entry: dict) -> dict:
    if entry.get("error"):
        return fallback_grade(entry, f"generation_error: {entry['error']}")
    raw = client.tracked_generate(
        JUDGE_PROMPT.format(
            question=str(entry.get("question", "")),
            reference=str(entry.get("reference", "")),
            prediction=str(entry.get("prediction", "")),
        ),
        max_tokens=128,
        system=JUDGE_SYSTEM,
        temperature=0.0,
    )
    parsed = parse_judge_output(raw)
    return {
        **entry,
        "judge_raw": raw,
        "judge_label": parsed["label"],
        "judge_reasoning": parsed["reasoning"],
        "judge_score": LABEL_SCORE[parsed["label"]],
        "f1": f1_score(entry["prediction"], entry["reference"]),
        "em": exact_match(entry["prediction"], entry["reference"]),
    }


def parse_judge_output(text: str) -> dict:
    if not text:
        return {"label": "WRONG", "reasoning": "empty judge output"}
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {"label": "WRONG", "reasoning": f"unparseable: {text[:100]}"}
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"label": "WRONG", "reasoning": f"unparseable: {text[:100]}"}
    label = str(obj.get("label", "")).upper().strip()
    if label not in LABEL_SCORE:
        label = next((key for key in LABEL_SCORE if key in label), "WRONG")
    return {"label": label, "reasoning": str(obj.get("reasoning", ""))[:300]}


def fallback_grade(entry: dict, reason: str) -> dict:
    return {
        **entry,
        "judge_label": "WRONG",
        "judge_reasoning": reason,
        "judge_score": 0.0,
        "f1": f1_score(entry.get("prediction", ""), entry.get("reference", "")),
        "em": exact_match(entry.get("prediction", ""), entry.get("reference", "")),
    }


def aggregate(graded: list[dict], client: LLMClient, predictions_file: str) -> dict:
    by_category = defaultdict(lambda: {"n": 0, "score": 0.0, "f1": 0.0, "em": 0.0})
    for item in graded:
        bucket = by_category[item["category_name"]]
        bucket["n"] += 1
        bucket["score"] += item["judge_score"]
        bucket["f1"] += item["f1"]
        bucket["em"] += item["em"]
    for bucket in by_category.values():
        n = max(bucket["n"], 1)
        bucket["score"] = round(bucket["score"] / n, 4)
        bucket["f1"] = round(bucket["f1"] / n, 4)
        bucket["em"] = round(bucket["em"] / n, 4)
    n_total = max(len(graded), 1)
    return {
        "overall": {
            "n": len(graded),
            "score": round(sum(item["judge_score"] for item in graded) / n_total, 4),
            "f1": round(sum(item["f1"] for item in graded) / n_total, 4),
            "em": round(sum(item["em"] for item in graded) / n_total, 4),
            "avg_llm_calls_per_question": round(
                sum(item.get("llm_calls", 0) for item in graded) / n_total, 4
            ),
            "avg_end_to_end_latency_sec": round(
                sum(item.get("end_to_end_latency_sec", 0.0) for item in graded) / n_total, 4
            ),
        },
        "by_category": dict(by_category),
        "judge_model": client.model,
        "judge_client": client.snapshot(),
        "predictions_file": predictions_file,
        "graded": graded,
    }


def print_table(results: dict) -> None:
    print("\n===== Judge Results =====")
    print(f"{'category':<14}{'n':>5}{'score':>9}{'f1':>9}{'em':>9}")
    for name, item in sorted(results["by_category"].items()):
        print(f"{name:<14}{item['n']:>5}{item['score']:>9.3f}{item['f1']:>9.3f}{item['em']:>9.3f}")
    overall = results["overall"]
    print(f"{'overall':<14}{overall['n']:>5}{overall['score']:>9.3f}{overall['f1']:>9.3f}{overall['em']:>9.3f}")
    print(f"avg LLM calls/question: {overall['avg_llm_calls_per_question']}")
    print(f"avg end-to-end latency: {overall['avg_end_to_end_latency_sec']}s")


if __name__ == "__main__":
    main()
