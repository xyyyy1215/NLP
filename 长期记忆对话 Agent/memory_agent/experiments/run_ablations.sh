#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
EVAL_SET="${EVAL_SET:-memory_agent/eval/data/eval_sets/eval_set_50_200.json}"
JUDGE_WORKERS="${JUDGE_WORKERS:-2}"
RUN_JUDGE="${RUN_JUDGE:-1}"

export LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8000/v1}"
export LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-3B-Instruct-AWQ}"
export EMBED_MODEL="${EMBED_MODEL:-/home/ubuntu22/nlp_memory_agent/models/bge-m3}"
export EMBED_DEVICE="${EMBED_DEVICE:-cpu}"

run_variant() {
  local run_name="$1"
  shift

  echo
  echo "===== Running ${run_name} ====="
  env "$@" "${PYTHON}" -m memory_agent.eval.run_eval \
    --eval_set "${EVAL_SET}" \
    --agent memory_agent \
    --run_name "${run_name}"

  if [[ "${RUN_JUDGE}" == "1" ]]; then
    echo
    echo "===== Judging ${run_name} ====="
    env "$@" "${PYTHON}" -m memory_agent.eval.run_judge \
      --predictions "memory_agent/experiments/results/${run_name}/predictions.json" \
      --output "memory_agent/experiments/results/${run_name}/results_judge.json" \
      --num_workers "${JUDGE_WORKERS}"
  fi
}

run_variant "ours_ablation_fast" \
  MEMORY_AGENT_USE_WRITER=0 \
  MEMORY_AGENT_USE_RAW_TURNS=1 \
  MEMORY_AGENT_USE_TEMPORAL_HINT=1 \
  MEMORY_AGENT_USE_LEXICAL=1 \
  MEMORY_AGENT_TOP_K=8

run_variant "ours_ablation_no_temporal" \
  MEMORY_AGENT_USE_WRITER=0 \
  MEMORY_AGENT_USE_RAW_TURNS=1 \
  MEMORY_AGENT_USE_TEMPORAL_HINT=0 \
  MEMORY_AGENT_USE_LEXICAL=1 \
  MEMORY_AGENT_TOP_K=8

run_variant "ours_ablation_embedding_only" \
  MEMORY_AGENT_USE_WRITER=0 \
  MEMORY_AGENT_USE_RAW_TURNS=1 \
  MEMORY_AGENT_USE_TEMPORAL_HINT=1 \
  MEMORY_AGENT_USE_LEXICAL=0 \
  MEMORY_AGENT_TOP_K=8

run_variant "ours_ablation_writer_only" \
  MEMORY_AGENT_USE_WRITER=1 \
  MEMORY_AGENT_USE_RAW_TURNS=0 \
  MEMORY_AGENT_USE_TEMPORAL_HINT=0 \
  MEMORY_AGENT_USE_LEXICAL=1 \
  MEMORY_AGENT_TOP_K=8
