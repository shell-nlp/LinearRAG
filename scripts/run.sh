#!/usr/bin/env bash

set -euo pipefail

uv run python run.py \
  --dataset_name "${DATASET_NAME:-2wikimultihop}" \
  --spacy_model "${SPACY_MODEL:-en_core_web_trf}" \
  --llm_model "${LLM_MODEL:-gpt-4o-mini}" \
  --embedding_model "${EMBEDDING_MODEL:-text-embedding-3-small}" \
  --max_workers "${MAX_WORKERS:-8}" \
  --max_iterations "${MAX_ITERATIONS:-3}" \
  --iteration_threshold "${ITERATION_THRESHOLD:-0.5}" \
  --passage_ratio "${PASSAGE_RATIO:-1.5}" \
  --top_k_sentence "${TOP_K_SENTENCE:-3}"
