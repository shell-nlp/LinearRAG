from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from src.config import LinearRAGConfig
from src.elastic_utils import Elasticsearch
from src.utils import compute_mdhash_id


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

    def build_document(self, doc_type: str, content: str, metadata: dict[str, Any] | None = None) -> IndexedDocument:
        doc_id = compute_mdhash_id(content, prefix=f"{doc_type}-")
        return IndexedDocument(doc_id=doc_id, doc_type=doc_type, content=content, metadata=metadata or {})

    def upsert_documents(self, documents: Sequence[IndexedDocument]) -> list[str]:
        if not documents:
            return []
        return self.client.add_batch([document.to_payload() for document in documents], index_name=self.index_name)

    def index_exists(self) -> bool:
        return self.client.index_exists(index_name=self.index_name)

    def get(self, doc_id: str) -> dict[str, Any] | None:
        return self.client.get(doc_id=doc_id, index_name=self.index_name)

    def get_many(self, doc_ids: Iterable[str]) -> list[dict[str, Any]]:
        return self.client.get_many(doc_ids=doc_ids, index_name=self.index_name)

    def update_metadata(self, doc_id: str, metadata: dict[str, Any]) -> bool:
        return self.client.update(doc_id=doc_id, metadata=metadata, index_name=self.index_name)

    def get_documents(self, doc_type: str | None = None, ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
        if ids is not None:
            documents = self.get_many(doc_ids=ids)
            if doc_type is None:
                return documents
            return [document for document in documents if document.get("doc_type") == doc_type]

        filters = {"doc_type": doc_type} if doc_type else None
        return self.client.search_all(filter_conditions=filters, index_name=self.index_name)

    def existing_ids(self, doc_ids: Iterable[str]) -> set[str]:
        return self.client.existing_ids(doc_ids=doc_ids, index_name=self.index_name)

    def count(self, doc_type: str | None = None) -> int:
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
        return self.client.keyword_search(
            query=query,
            k=k,
            index_name=self.index_name,
            filter_conditions={"doc_type": doc_type},
        )

    def hybrid_search(self, query: str, doc_type: str, k: int) -> list[dict[str, Any]]:
        return self.client.retrieve(
            query=query,
            k=k,
            index_name=self.index_name,
            filter_conditions={"doc_type": doc_type},
        )
