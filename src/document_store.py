from __future__ import annotations

import logging
import re
import socket
from dataclasses import dataclass, field
from typing import Any
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

from src.config import LinearRAGConfig
from src.elastic_utils import Elasticsearch
from src.utils import compute_mdhash_id


logger = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")


@dataclass(slots=True)
class IndexedDocument:
    doc_id: str
    doc_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        metadata = {"id": self.doc_id, **self.metadata}
        metadata["doc_type"] = self.doc_type
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "content": self.content,
            "metadata": metadata,
        }


def _keyword_units(text: str) -> list[str]:
    units: list[str] = []
    for token in TOKEN_PATTERN.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            units.extend(token)
            if len(token) > 1:
                units.extend(token[index : index + 2] for index in range(len(token) - 1))
        else:
            units.append(token)
    return units


def _lexical_score(query: str, content: str) -> float:
    query_units = _keyword_units(query)
    content_units = set(_keyword_units(content))
    if not query_units or not content_units:
        return 0.0
    overlap = sum(1 for unit in query_units if unit in content_units)
    return overlap / len(query_units)


class InMemoryDocumentStore:
    def __init__(self, config: LinearRAGConfig):
        self.config = config
        self.index_name = config.index_name
        self._documents: dict[str, dict[str, Any]] = {}

    def build_document(self, doc_type: str, content: str, metadata: dict[str, Any] | None = None) -> IndexedDocument:
        doc_id = compute_mdhash_id(content, prefix=f"{doc_type}-")
        return IndexedDocument(doc_id=doc_id, doc_type=doc_type, content=content, metadata=metadata or {})

    def upsert_documents(self, documents: Sequence[IndexedDocument]) -> list[str]:
        doc_ids: list[str] = []
        for document in documents:
            self._documents[document.doc_id] = {
                "id": document.doc_id,
                "content": document.content,
                "doc_type": document.doc_type,
                "metadata": dict(document.to_payload()["metadata"]),
            }
            doc_ids.append(document.doc_id)
        return doc_ids

    def index_exists(self) -> bool:
        return True

    def get(self, doc_id: str) -> dict[str, Any] | None:
        return self._documents.get(doc_id)

    def get_many(self, doc_ids: Iterable[str]) -> list[dict[str, Any]]:
        return [self._documents[doc_id] for doc_id in doc_ids if doc_id in self._documents]

    def update_metadata(self, doc_id: str, metadata: dict[str, Any]) -> bool:
        document = self._documents.get(doc_id)
        if document is None:
            return False
        document["metadata"] = dict(metadata)
        if metadata.get("doc_type"):
            document["doc_type"] = metadata["doc_type"]
        return True

    def get_documents(self, doc_type: str | None = None, ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
        if ids is not None:
            documents = self.get_many(ids)
        else:
            documents = list(self._documents.values())
        if doc_type is None:
            return documents
        return [document for document in documents if document.get("doc_type") == doc_type]

    def existing_ids(self, doc_ids: Iterable[str]) -> set[str]:
        return {doc_id for doc_id in doc_ids if doc_id in self._documents}

    def count(self, doc_type: str | None = None) -> int:
        return len(self.get_documents(doc_type=doc_type))

    def vector_search(
        self,
        query: str,
        doc_type: str,
        k: int,
        ids: Iterable[str] | None = None,
        min_similarity: float | None = None,
    ) -> list[dict[str, Any]]:
        candidates = self.get_documents(doc_type=doc_type, ids=ids)
        scored_documents = []
        for document in candidates:
            score = _lexical_score(query, str(document.get("content", "")))
            if min_similarity is not None and score < min_similarity:
                continue
            if score <= 0:
                continue
            scored_documents.append({**document, "score": score})
        scored_documents.sort(key=lambda item: item["score"], reverse=True)
        return scored_documents[:k]

    def keyword_search(self, query: str, doc_type: str, k: int) -> list[dict[str, Any]]:
        return self.vector_search(query=query, doc_type=doc_type, k=k)

    def hybrid_search(self, query: str, doc_type: str, k: int) -> list[dict[str, Any]]:
        return self.vector_search(query=query, doc_type=doc_type, k=k)


class ElasticsearchDocumentStore:
    def __init__(self, config: LinearRAGConfig, embedding_model):
        self.config = config
        self.index_name = config.index_name
        self.client = Elasticsearch(
            url=config.es_url,
            username=config.es_user,
            password=config.es_password,
            embedding_model=embedding_model,
        )
        self._memory_store: InMemoryDocumentStore | None = None
        try:
            parsed_url = urlparse(config.es_url)
            host = parsed_url.hostname or "127.0.0.1"
            port = parsed_url.port or 9200
            with socket.create_connection((host, port), timeout=0.5):
                pass
        except Exception as exc:
            logger.warning("Elasticsearch is unavailable (%s); using in-memory document store.", exc)
            self._memory_store = InMemoryDocumentStore(config)

    def build_document(self, doc_type: str, content: str, metadata: dict[str, Any] | None = None) -> IndexedDocument:
        if self._memory_store is not None:
            return self._memory_store.build_document(doc_type=doc_type, content=content, metadata=metadata)
        doc_id = compute_mdhash_id(content, prefix=f"{doc_type}-")
        return IndexedDocument(doc_id=doc_id, doc_type=doc_type, content=content, metadata=metadata or {})

    def upsert_documents(self, documents: Sequence[IndexedDocument]) -> list[str]:
        if self._memory_store is not None:
            return self._memory_store.upsert_documents(documents)
        if not documents:
            return []
        return self.client.add_batch([document.to_payload() for document in documents], index_name=self.index_name)

    def index_exists(self) -> bool:
        if self._memory_store is not None:
            return self._memory_store.index_exists()
        return self.client.index_exists(index_name=self.index_name)

    def get(self, doc_id: str) -> dict[str, Any] | None:
        if self._memory_store is not None:
            return self._memory_store.get(doc_id)
        return self.client.get(doc_id=doc_id, index_name=self.index_name)

    def get_many(self, doc_ids: Iterable[str]) -> list[dict[str, Any]]:
        if self._memory_store is not None:
            return self._memory_store.get_many(doc_ids)
        return self.client.get_many(doc_ids=doc_ids, index_name=self.index_name)

    def update_metadata(self, doc_id: str, metadata: dict[str, Any]) -> bool:
        if self._memory_store is not None:
            return self._memory_store.update_metadata(doc_id, metadata)
        return self.client.update(doc_id=doc_id, metadata=metadata, index_name=self.index_name)

    def get_documents(self, doc_type: str | None = None, ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
        if self._memory_store is not None:
            return self._memory_store.get_documents(doc_type=doc_type, ids=ids)
        if ids is not None:
            documents = self.get_many(doc_ids=ids)
            if doc_type is None:
                return documents
            return [document for document in documents if document.get("doc_type") == doc_type]

        filters = {"doc_type": doc_type} if doc_type else None
        return self.client.search_all(filter_conditions=filters, index_name=self.index_name)

    def existing_ids(self, doc_ids: Iterable[str]) -> set[str]:
        if self._memory_store is not None:
            return self._memory_store.existing_ids(doc_ids)
        return self.client.existing_ids(doc_ids=doc_ids, index_name=self.index_name)

    def count(self, doc_type: str | None = None) -> int:
        if self._memory_store is not None:
            return self._memory_store.count(doc_type=doc_type)
        filters = {"doc_type": doc_type} if doc_type else None
        return self.client.count(filter_conditions=filters, index_name=self.index_name)

    def vector_search(
        self,
        query: str,
        doc_type: str,
        k: int,
        ids: Iterable[str] | None = None,
        min_similarity: float | None = None,
    ) -> list[dict[str, Any]]:
        if self._memory_store is not None:
            return self._memory_store.vector_search(
                query=query,
                doc_type=doc_type,
                k=k,
                ids=ids,
                min_similarity=min_similarity,
            )
        filters = {"doc_type": doc_type}
        return self.client.vector_search(
            query=query,
            k=k,
            index_name=self.index_name,
            min_similarity=min_similarity,
            filter_conditions=filters,
            ids=list(ids) if ids is not None else None,
        )

    def keyword_search(self, query: str, doc_type: str, k: int) -> list[dict[str, Any]]:
        if self._memory_store is not None:
            return self._memory_store.keyword_search(query=query, doc_type=doc_type, k=k)
        return self.client.keyword_search(
            query=query,
            k=k,
            index_name=self.index_name,
            filter_conditions={"doc_type": doc_type},
        )

    def hybrid_search(self, query: str, doc_type: str, k: int) -> list[dict[str, Any]]:
        if self._memory_store is not None:
            return self._memory_store.hybrid_search(query=query, doc_type=doc_type, k=k)
        return self.client.retrieve(
            query=query,
            k=k,
            index_name=self.index_name,
            filter_conditions={"doc_type": doc_type},
        )
