cd "/home/ubuntu22/NLP/长期记忆对话 Agent"
source /home/ubuntu22/nlp_memory_agent/.venv/bin/activate

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export OUTLINES_CACHE_DIR=/tmp/outlines

python -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu22/nlp_memory_agent/models/Qwen2.5-3B-Instruct-AWQ \
  --served-model-name Qwen/Qwen2.5-3B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.65


另一个终端：
cd "/home/ubuntu22/NLP/长期记忆对话 Agent"
source /home/ubuntu22/nlp_memory_agent/.venv/bin/activate

export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_API_KEY="EMPTY"
export LLM_MODEL="Qwen/Qwen2.5-3B-Instruct-AWQ"
export EMBED_MODEL="/home/ubuntu22/nlp_memory_agent/models/bge-m3"
export EMBED_DEVICE="cpu"

base:
python -m memory_agent.eval.run_eval \
  --eval_set memory_agent/eval/data/eval_sets/eval_set_50_200.json \
  --agent vanilla_rag \
  --run_name baseline_vanilla_rag_vllm

自定义：
python -m memory_agent.eval.run_eval \
  --eval_set memory_agent/eval/data/eval_sets/eval_set_50_200.json \
  --agent memory_agent \
  --run_name ours_memory_agent_vllm

cat memory_agent/experiments/results/baseline_vanilla_rag_vllm/summary.json
cat memory_agent/experiments/results/ours_memory_agent_vllm/summary.json