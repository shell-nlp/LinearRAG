from __future__ import annotations

import argparse
import json
from datetime import datetime

from src.LinearRAG import LinearRAG
from src.config import LinearRAGConfig
from src.evaluate import Evaluator
from src.langchain_clients import LangChainEmbeddingModel, LangChainLLM
from src.utils import setup_logging


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="novel", help="Dataset directory under DATA_DIR")
    parser.add_argument("--spacy_model", type=str, default=None, help="spaCy model name")
    parser.add_argument("--llm_model", type=str, default=None, help="LLM model name")
    parser.add_argument("--embedding_model", type=str, default=None, help="Embedding model name")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum worker count")
    parser.add_argument("--max_iterations", type=int, default=None, help="Maximum graph expansion iterations")
    parser.add_argument("--iteration_threshold", type=float, default=None, help="Entity activation threshold")
    parser.add_argument("--passage_ratio", type=float, default=None, help="Weight for dense passage retrieval")
    parser.add_argument("--top_k_sentence", type=int, default=None, help="Top sentence count per entity expansion")
    parser.add_argument("--skip_eval", action="store_true", help="Skip answer evaluation")
    return parser.parse_args()


def load_dataset(config: LinearRAGConfig) -> tuple[list[dict[str, str]], list[str]]:
    questions_path = config.dataset_dir / "questions.json"
    chunks_path = config.dataset_dir / "chunks.json"

    with questions_path.open("r", encoding="utf-8") as file:
        questions = json.load(file)
    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    passages = [f"{idx}:{chunk}" for idx, chunk in enumerate(chunks)]
    return questions, passages


def build_config(args: argparse.Namespace) -> LinearRAGConfig:
    config = LinearRAGConfig(dataset_name=args.dataset_name)
    if args.spacy_model:
        config.spacy_model = args.spacy_model
    if args.llm_model:
        config.llm_model_name = args.llm_model
    if args.embedding_model:
        config.embedding_model_name = args.embedding_model
    if args.max_workers is not None:
        config.max_workers = args.max_workers
    if args.max_iterations is not None:
        config.max_iterations = args.max_iterations
    if args.iteration_threshold is not None:
        config.iteration_threshold = args.iteration_threshold
    if args.passage_ratio is not None:
        config.passage_ratio = args.passage_ratio
    if args.top_k_sentence is not None:
        config.top_k_sentence = args.top_k_sentence
    return config


def main() -> None:
    args = parse_arguments()
    config = build_config(args)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = config.results_dir / config.dataset_name / timestamp
    setup_logging(str(run_dir / "log.txt"))

    llm_model = LangChainLLM(config)
    embedding_model = LangChainEmbeddingModel(config)
    questions, passages = load_dataset(config)

    rag_model = LinearRAG(global_config=config, llm_model=llm_model, embedding_model=embedding_model)
    rag_model.index(passages)
    predictions = rag_model.qa(questions)

    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.json"
    with predictions_path.open("w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=2)

    if args.skip_eval:
        return

    evaluator = Evaluator(llm_model=llm_model, predictions_path=str(predictions_path))
    evaluator.evaluate(max_workers=config.max_workers)


if __name__ == "__main__":
    main()
