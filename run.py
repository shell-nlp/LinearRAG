from __future__ import annotations

import json
from datetime import datetime

from src.config import LinearRAGConfig
from src.langchain_clients import LangChainEmbeddingModel, LangChainLLM
from src.LinearRAG import LinearRAG
from src.utils import setup_logging


def build_sample_data() -> tuple[list[dict[str, str]], list[str]]:
    questions = [
        {"question": "Python 编程语言是由谁创建的？", "answer": "Guido van Rossum"},
        {"question": "日本的首都是哪座城市？", "answer": "东京"},
        {"question": "相对论是哪位科学家提出的？", "answer": "阿尔伯特·爱因斯坦"},
    ]
    passage_texts = [
        (
            "Python 是一种高级编程语言，由 Guido van Rossum 创建。"
            "它强调代码可读性，并拥有丰富的标准库。"
        ),
        (
            "东京是日本的首都，也是世界上规模最大的都市圈之一。"
            "它同时是日本的政治中心和经济中心。"
        ),
        (
            "阿尔伯特·爱因斯坦提出了相对论。"
            "这项理论改变了现代物理学，并解释了空间、时间与引力之间的关系。"
        ),
        (
            "格蕾丝·霍珀是美国计算机科学家，也是美国海军少将。"
            "她为早期编译器的发展作出了重要贡献。"
        ),
        ("太平洋是地球上面积最大、最深的大洋。它从北极附近海域一直延伸到南冰洋。"),
    ]
    passages = [f"{idx}:{text}" for idx, text in enumerate(passage_texts)]
    return questions, passages


def build_config() -> LinearRAGConfig:
    config = LinearRAGConfig(dataset_name="sample")
    config.llm_model_name = "gpt-4o"
    config.embedding_model_name = "qwen3-embedding"
    return config


def main() -> None:
    config = build_config()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = config.results_dir / config.dataset_name / timestamp
    setup_logging(str(run_dir / "log.txt"))

    llm_model = LangChainLLM(config)
    embedding_model = LangChainEmbeddingModel(config)
    questions, passages = build_sample_data()

    rag_model = LinearRAG(
        global_config=config,
        llm_model=llm_model,
        embedding_model=embedding_model,
    )
    rag_model.index(passages)
    predictions = rag_model.qa(questions)
    print(predictions)

    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.json"
    with predictions_path.open("w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
