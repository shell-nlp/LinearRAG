from __future__ import annotations

from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import LinearRAGConfig


def _to_langchain_message(message: dict[str, str]) -> BaseMessage:
    role = message["role"]
    content = message["content"]
    if role == "system":
        return SystemMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    return HumanMessage(content=content)


def _message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments: list[str] = []
        for item in content:
            if isinstance(item, str):
                fragments.append(item)
                continue
            if isinstance(item, dict) and item.get("type") == "text":
                fragments.append(str(item.get("text", "")))
        return "".join(fragments)
    return str(content)


class LangChainLLM:
    def __init__(self, config: LinearRAGConfig):
        self._client = ChatOpenAI(
            model=config.llm_model_name,
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            temperature=0,
            max_tokens=config.llm_max_tokens,
            timeout=config.request_timeout,
        )

    def infer(self, messages: Sequence[dict[str, str]]) -> str:
        response = self._client.invoke([_to_langchain_message(message) for message in messages])
        return _message_content_to_text(response.content)


class LangChainEmbeddingModel:
    def __init__(self, config: LinearRAGConfig):
        self._client = OpenAIEmbeddings(
            model=config.embedding_model_name,
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            chunk_size=config.batch_size,
            max_retries=2,
        )

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._client.embed_documents(list(texts))
