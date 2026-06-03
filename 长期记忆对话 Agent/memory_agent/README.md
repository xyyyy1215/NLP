# Long-Term Memory Dialogue Agent

## Directory

```text
memory_agent/
├── memory/
│   ├── store.py
│   ├── writer.py
│   ├── retriever.py
│   └── updater.py
├── agent/
│   └── controller.py
├── eval/
│   ├── run_eval.py
│   ├── run_judge.py
│   ├── baselines/
│   ├── configs/
│   └── data/eval_sets/
├── experiments/
│   └── results/
└── README.md
```

## Environment

Install dependencies from the project root:

```bash
pip install -r memory_agent/eval/requirements.txt
```

Download models to the local `models/` directory before running:

```bash
python -m memory_agent.eval.download_models
```

If HuggingFace access is slow, add `--hf_endpoint https://hf-mirror.com`.

This creates:

```text
models/
├── Qwen2.5-3B-Instruct-AWQ/
└── bge-m3/
```

Start the local OpenAI-compatible generation service from the downloaded path:

```bash
OUTLINES_CACHE_DIR=/tmp/outlines vllm serve ./models/Qwen2.5-3B-Instruct-AWQ \
  --served-model-name Qwen/Qwen2.5-3B-Instruct-AWQ \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.75
```

If vLLM reports `ModuleNotFoundError: No module named 'pyairports'`, reinstall the dependency and keep `OUTLINES_CACHE_DIR` set to a writable directory.

Then configure generation:

```bash
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_API_KEY="EMPTY"
export LLM_MODEL="Qwen/Qwen2.5-3B-Instruct-AWQ"
export EMBED_MODEL="models/bge-m3"
```

For DeepSeek Judge, create a private config file:

```bash
cp memory_agent/eval/configs/deepseek.env.example memory_agent/eval/configs/deepseek.env
# edit memory_agent/eval/configs/deepseek.env and fill LLM_API_KEY
source memory_agent/eval/configs/deepseek.env
```

## Run Three Baselines

Use the small set first:

```bash
python -m memory_agent.eval.run_eval \
  --eval_set memory_agent/eval/data/eval_sets/eval_set_50_200.json \
  --agent no_memory \
  --run_name baseline_no_memory

python -m memory_agent.eval.run_eval \
  --eval_set memory_agent/eval/data/eval_sets/eval_set_50_200.json \
  --agent full_context \
  --run_name baseline_full_context

python -m memory_agent.eval.run_eval \
  --eval_set memory_agent/eval/data/eval_sets/eval_set_50_200.json \
  --agent vanilla_rag \
  --run_name baseline_vanilla_rag
```

Vanilla RAG uses the local `sentence-transformers` embedding model at `models/bge-m3`.

## Outputs

Each run writes files under `memory_agent/experiments/results/<run_name>/`:

- `predictions.json`: per-question prediction records.
- `summary.json`: aggregate runtime and cost metrics.
- `trace.jsonl`: per-question retrieved memories, prompts, and model trace.
- `run.log`: progress and error log.

The required cost and latency metrics are in `summary.json`:

- `overall.avg_llm_calls_per_question`
- `overall.avg_end_to_end_latency_sec`

## MemoryAgent Architecture

The custom system is a lightweight baseline inspired by Generative Agents, MemoryBank, and Mem0:

- `memory/writer.py`: uses an LLM prompt to extract concise long-term memory units from each dialogue session. Each unit stores text, subject, attribute, source, timestamp, and importance.
- `memory/store.py`: stores memory units and builds a local `BAAI/bge-m3` embedding index with `sentence-transformers`.
- `memory/retriever.py`: ranks memories with relevance, recency, and importance. The default weights are `0.70 / 0.15 / 0.15`.
- `memory/updater.py`: deduplicates identical memories and keeps the newer memory for the same subject/attribute slot.
- `agent/controller.py`: orchestrates write, update, store, retrieve, and answer generation. The trace records retrieved memories, prompt, and score components.

Run the custom system:

```bash
python -m memory_agent.eval.run_eval \
  --eval_set memory_agent/eval/data/eval_sets/eval_set_50_200.json \
  --agent memory_agent \
  --run_name ours_memory_agent
```

## Judge

After sourcing the DeepSeek config:

```bash
python -m memory_agent.eval.run_judge \
  --predictions memory_agent/experiments/results/baseline_no_memory_small/predictions.json \
  --num_workers "$JUDGE_NUM_WORKERS"
```
