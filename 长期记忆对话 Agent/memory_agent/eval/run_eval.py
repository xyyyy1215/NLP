import argparse
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

from memory_agent.eval.core.logging import append_jsonl, configure_logger


BASELINE_AGENTS = {
    "no_memory": "memory_agent.eval.baselines.no_memory:NoMemoryAgent",
    "full_context": "memory_agent.eval.baselines.full_context:FullContextAgent",
    "vanilla_rag": "memory_agent.eval.baselines.vanilla_rag:VanillaRAGAgent",
    "memory_agent": "memory_agent.agent.controller:MemoryAgent",
}


def load_agent_class(spec: str):
    if spec in BASELINE_AGENTS:
        spec = BASELINE_AGENTS[spec]
    if ":" not in spec:
        raise ValueError(f"agent must be 'module.path:ClassName' or one of {sorted(BASELINE_AGENTS)}")
    module_name, class_name = spec.split(":", 1)
    if "" not in sys.path:
        sys.path.insert(0, "")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generation for memory-agent baselines or system.")
    parser.add_argument("--eval_set", default="memory_agent/eval/data/eval_sets/eval_set_small.json")
    parser.add_argument("--agent", default="memory_agent", help="alias or module:ClassName")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit_conversations", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--llm_base_url", default=None, help="Override LLM_BASE_URL for generation.")
    parser.add_argument("--llm_model", default=None, help="Override LLM_MODEL for generation.")
    parser.add_argument("--llm_api_key", default=None, help="Override LLM_API_KEY for generation.")
    args = parser.parse_args()

    if args.llm_base_url:
        os.environ["LLM_BASE_URL"] = args.llm_base_url
    if args.llm_model:
        os.environ["LLM_MODEL"] = args.llm_model
    if args.llm_api_key:
        os.environ["LLM_API_KEY"] = args.llm_api_key

    run_name = args.run_name or args.agent.replace(":", "_").replace(".", "_")
    run_dir = Path("memory_agent/experiments/results") / run_name
    output = Path(args.output) if args.output else run_dir / "predictions.json"
    summary_path = run_dir / "summary.json"
    trace_path = run_dir / "trace.jsonl"
    logger = configure_logger(run_dir / "run.log")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(args.eval_set, encoding="utf-8") as f:
        eval_set = json.load(f)
    if args.limit_conversations:
        eval_set = eval_set[: args.limit_conversations]

    predictions = []
    done_ids = set()
    if args.resume and output.exists():
        with output.open(encoding="utf-8") as f:
            predictions = json.load(f)
        done_ids = {item["qa_id"] for item in predictions}
        logger.info("Resume enabled: loaded %d existing predictions.", len(done_ids))

    AgentCls = load_agent_class(args.agent)
    total_convs = len(eval_set)
    total_qas = sum(len(sample["qa_list"]) for sample in eval_set)
    logger.info("Start run=%s agent=%s eval_set=%s conversations=%d qas=%d",
                run_name, args.agent, args.eval_set, total_convs, total_qas)
    logger.info("Generation LLM base_url=%s model=%s",
                os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
                os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct-AWQ"))

    qa_done = len(done_ids)
    for idx, sample in enumerate(eval_set, start=1):
        sample_id = sample["sample_id"]
        remaining = [qa for qa in sample["qa_list"] if qa["qa_id"] not in done_ids]
        if not remaining:
            continue

        agent = AgentCls()
        logger.info("[%d/%d] sample=%s ingest sessions=%d qas=%d",
                    idx, total_convs, sample_id,
                    len(sample["conversation"].get("sessions", [])), len(remaining))
        calls_before_ingest = _llm_calls(agent)
        ingest_started = time.perf_counter()
        try:
            agent.ingest(sample["conversation"])
            ingest_error = None
        except Exception as exc:
            ingest_error = f"ingest_failed: {exc}"
            traceback.print_exc()
        ingest_time = time.perf_counter() - ingest_started
        ingest_share = ingest_time / max(len(remaining), 1)
        ingest_calls_share = max(_llm_calls(agent) - calls_before_ingest, 0) / max(len(remaining), 1)

        if ingest_error:
            for qa in remaining:
                entry = _prediction_entry(qa, "", ingest_error, 0.0, ingest_share, ingest_calls_share)
                predictions.append(entry)
                append_jsonl(trace_path, {"sample_id": sample_id, **entry})
                qa_done += 1
            _save_json(output, predictions)
            continue

        for qa in remaining:
            before_calls = _llm_calls(agent)
            answer_started = time.perf_counter()
            try:
                answer = agent.answer(qa["question"])
                error = None
            except Exception as exc:
                answer = ""
                error = f"answer_failed: {exc}"
                traceback.print_exc()
            answer_latency = time.perf_counter() - answer_started
            llm_calls = max(_llm_calls(agent) - before_calls, 0) + ingest_calls_share
            entry = _prediction_entry(qa, answer, error, answer_latency, ingest_share, llm_calls)
            predictions.append(entry)

            trace = agent.get_trace() if hasattr(agent, "get_trace") else {}
            append_jsonl(trace_path, {
                "sample_id": sample_id,
                **entry,
                "trace": trace,
            })
            qa_done += 1

        _save_json(output, predictions)
        _save_json(summary_path, _summarize(predictions, run_name, args.agent, args.eval_set))
        logger.info("Finished sample=%s ingest=%.3fs answered=%d progress=%d/%d",
                    sample_id, ingest_time, len(remaining), qa_done, total_qas)

    summary = _summarize(predictions, run_name, args.agent, args.eval_set)
    _save_json(summary_path, summary)
    logger.info("Done predictions=%s summary=%s avg_llm_calls=%.3f avg_e2e=%.3fs",
                output, summary_path,
                summary["overall"]["avg_llm_calls_per_question"],
                summary["overall"]["avg_end_to_end_latency_sec"])


def _prediction_entry(
    qa: dict,
    answer: str,
    error: str | None,
    answer_latency: float,
    ingest_share: float,
    llm_calls: float,
) -> dict:
    end_to_end = answer_latency + ingest_share
    return {
        "qa_id": qa["qa_id"],
        "question": qa["question"],
        "reference": qa["answer"],
        "category": qa["category"],
        "category_name": qa["category_name"],
        "prediction": str(answer).strip(),
        "error": error,
        "latency_sec": round(answer_latency, 3),
        "end_to_end_latency_sec": round(end_to_end, 3),
        "llm_calls": round(llm_calls, 4),
    }


def _llm_calls(agent) -> int:
    llm = getattr(agent, "llm", None)
    stats = getattr(llm, "stats", None)
    return int(getattr(stats, "calls", 0))


def _summarize(predictions: list[dict], run_name: str, agent: str, eval_set: str) -> dict:
    total = len(predictions)
    by_category: dict[str, dict] = {}
    for item in predictions:
        name = item["category_name"]
        bucket = by_category.setdefault(name, {"n": 0, "llm_calls": 0, "latency": 0.0, "e2e": 0.0})
        bucket["n"] += 1
        bucket["llm_calls"] += item.get("llm_calls", 0)
        bucket["latency"] += item.get("latency_sec", 0.0)
        bucket["e2e"] += item.get("end_to_end_latency_sec", 0.0)

    for bucket in by_category.values():
        n = max(bucket["n"], 1)
        bucket["avg_llm_calls_per_question"] = round(bucket.pop("llm_calls") / n, 4)
        bucket["avg_answer_latency_sec"] = round(bucket.pop("latency") / n, 4)
        bucket["avg_end_to_end_latency_sec"] = round(bucket.pop("e2e") / n, 4)

    n_total = max(total, 1)
    return {
        "run_name": run_name,
        "agent": agent,
        "eval_set": eval_set,
        "overall": {
            "n": total,
            "errors": sum(1 for item in predictions if item.get("error")),
            "avg_llm_calls_per_question": round(
                sum(item.get("llm_calls", 0) for item in predictions) / n_total, 4
            ),
            "avg_answer_latency_sec": round(
                sum(item.get("latency_sec", 0.0) for item in predictions) / n_total, 4
            ),
            "avg_end_to_end_latency_sec": round(
                sum(item.get("end_to_end_latency_sec", 0.0) for item in predictions) / n_total, 4
            ),
        },
        "by_category": by_category,
    }


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
