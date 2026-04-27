from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _sanitize_index_name(dataset_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9_-]+", "-", dataset_name.lower())
    normalized = normalized.strip("-_")
    return f"linearrag-{normalized or 'default'}"


@dataclass(slots=True)
class LinearRAGConfig:
    dataset_name: str
    spacy_model: str = field(default_factory=lambda: os.getenv("SPACY_MODEL", "en_core_web_trf"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "dataset")))
    working_dir: Path = field(default_factory=lambda: Path(os.getenv("WORKING_DIR", "import")))
    results_dir: Path = field(default_factory=lambda: Path(os.getenv("RESULTS_DIR", "results")))

    llm_model_name: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    embedding_model_name: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://miyun.archermind.com/v1")
    )
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "123"))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 2000))
    request_timeout: float = field(default_factory=lambda: _env_float("REQUEST_TIMEOUT", 60.0))

    es_url: str = field(default_factory=lambda: os.getenv("ES_URL", "http://127.0.0.1:9200"))
    es_user: str = field(default_factory=lambda: os.getenv("ES_USER") or os.getenv("ES_URSR") or "elastic")
    es_password: str = field(default_factory=lambda: os.getenv("ES_PWD", "elastic@2024"))

    batch_size: int = field(default_factory=lambda: _env_int("BATCH_SIZE", 32))
    max_workers: int = field(default_factory=lambda: _env_int("MAX_WORKERS", 8))
    retrieval_top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 5))
    passage_search_candidate_k: int = field(default_factory=lambda: _env_int("PASSAGE_SEARCH_CANDIDATE_K", 50))
    max_iterations: int = field(default_factory=lambda: _env_int("MAX_ITERATIONS", 3))
    top_k_sentence: int = field(default_factory=lambda: _env_int("TOP_K_SENTENCE", 3))
    passage_ratio: float = field(default_factory=lambda: _env_float("PASSAGE_RATIO", 1.5))
    passage_node_weight: float = field(default_factory=lambda: _env_float("PASSAGE_NODE_WEIGHT", 0.05))
    damping: float = field(default_factory=lambda: _env_float("DAMPING", 0.5))
    iteration_threshold: float = field(default_factory=lambda: _env_float("ITERATION_THRESHOLD", 0.5))
    enable_hybrid_attribute_fallback: bool = field(
        default_factory=lambda: os.getenv("ENABLE_HYBRID_ATTRIBUTE_FALLBACK", "false").lower() == "true"
    )
    attribute_keyword_boost: float = field(default_factory=lambda: _env_float("ATTRIBUTE_KEYWORD_BOOST", 0.25))
    attribute_query_keywords: list[str] = field(
        default_factory=lambda: [
            "born",
            "birth",
            "where",
            "when",
            "located",
            "location",
            "founded",
            "founder",
            "died",
            "death",
            "nationality",
            "capital",
            "date",
            "year",
        ]
    )

    def __post_init__(self) -> None:
        self.passage_search_candidate_k = max(self.passage_search_candidate_k, self.retrieval_top_k)

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / self.dataset_name

    @property
    def cache_dir(self) -> Path:
        return self.working_dir / self.dataset_name

    @property
    def index_name(self) -> str:
        return _sanitize_index_name(self.dataset_name)

    @property
    def ner_results_path(self) -> Path:
        return self.cache_dir / "ner_results.json"

    @property
    def graph_path(self) -> Path:
        return self.cache_dir / "LinearRAG.graphml"
