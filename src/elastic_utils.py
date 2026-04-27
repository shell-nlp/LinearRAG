from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from elasticsearch import Elasticsearch as ESClient
from elasticsearch.helpers import scan
from loguru import logger


class Elasticsearch:
    def __init__(
        self,
        url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        embedding_model=None,
    ):
        self._url = url
        self._username = username
        self._password = password
        self._embedding_model = embedding_model
        self._es_client: Optional[ESClient] = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            try:
                from langchain_api.utils import get_embedding_model
            except ImportError as exc:
                raise RuntimeError("embedding_model is required when langchain_api is unavailable") from exc
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    @property
    def es_client(self) -> ESClient:
        if self._es_client is None:
            self._es_client = ESClient(
                hosts=[self._url],
                basic_auth=(self._username, self._password) if self._username and self._password else None,
            )
        return self._es_client

    def ensure_index(
        self,
        index_name: str,
        vector_dims: int,
        recreate: bool = False,
        mappings: Optional[Dict[str, Any]] = None,
    ) -> None:
        if recreate and self.es_client.indices.exists(index=index_name):
            self.es_client.indices.delete(index=index_name)

        if self.es_client.indices.exists(index=index_name):
            return

        base_mappings: Dict[str, Any] = {
            "dynamic": True,
            "properties": {
                "content": {"type": "text"},
                "doc_type": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": vector_dims,
                    "index": True,
                    "similarity": "cosine",
                },
                "metadata": {"type": "object", "dynamic": True},
            },
        }
        if mappings:
            base_mappings["properties"].update(mappings.get("properties", {}))

        self.es_client.indices.create(index=index_name, mappings=base_mappings)

    def index_exists(self, index_name: Optional[str] = None) -> bool:
        if not index_name:
            raise ValueError("index_name is required for index_exists operations")
        return bool(self.es_client.indices.exists(index=index_name))

    def vector_search(
        self,
        query: str,
        k: int = 3,
        index_name: Optional[str] = None,
        min_similarity: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not index_name:
            raise ValueError("index_name is required for search operations")
        if not self.index_exists(index_name=index_name):
            return []
        return self._vector_search_raw(
            query=query,
            k=k,
            index_name=index_name,
            min_similarity=min_similarity,
            filter_conditions=filter_conditions,
            ids=ids,
        )

    def keyword_search(
        self,
        query: str,
        k: int = 3,
        index_name: Optional[str] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not index_name:
            raise ValueError("index_name is required for search operations")
        if not self.index_exists(index_name=index_name):
            return []

        bool_query: Dict[str, Any] = {
            "must": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^2", "title", "summary"],
                        "type": "best_fields",
                    }
                }
            ]
        }
        filters = self._build_filter_clauses(filter_conditions=filter_conditions)
        if filters:
            bool_query["filter"] = filters

        results = self.es_client.search(index=index_name, body={"query": {"bool": bool_query}}, size=k)
        return [self._hit_to_result(hit) for hit in results["hits"]["hits"]]

    def retrieve(
        self,
        query: str,
        k: int = 3,
        index_name: Optional[str] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not index_name:
            raise ValueError("index_name is required for retrieve operations")
        if not self.index_exists(index_name=index_name):
            return []

        vector_results = self.vector_search(query, k, index_name, filter_conditions=filter_conditions)
        keyword_results = self.keyword_search(query, k, index_name, filter_conditions=filter_conditions)

        merged_results: List[Dict[str, Any]] = []
        seen_ids = set()
        for doc in vector_results + keyword_results:
            doc_id = doc["id"]
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            merged_results.append(doc)

        logger.info(
            "ES retrieve completed: vector_results={}, keyword_results={}, merged_results={}",
            len(vector_results),
            len(keyword_results),
            len(merged_results),
        )
        return merged_results[:k]

    def vector_graph_retrieve(
        self,
        query: str,
        k: int = 6,
        index_name: Optional[str] = None,
        entity_index_name: Optional[str] = None,
        relation_index_name: Optional[str] = None,
        entity_top_k: int = 5,
        relation_top_k: int = 8,
        expansion_degree: int = 1,
        relation_limit: int = 30,
        min_similarity: Optional[float] = None,
        query_entities: Optional[List[str]] = None,
        return_debug: bool = False,
    ) -> List[Dict[str, Any]] | Dict[str, Any]:
        if not index_name:
            raise ValueError("index_name is required for vector_graph_retrieve operations")

        entity_index_name = entity_index_name or index_name
        relation_index_name = relation_index_name or index_name
        query_entities = query_entities or self._simple_extract_entities(query)

        seed_entities = self._search_graph_items(
            texts=query_entities,
            index_name=entity_index_name,
            k=entity_top_k,
            min_similarity=min_similarity,
        )
        seed_relations = self._search_graph_items(
            texts=[query],
            index_name=relation_index_name,
            k=relation_top_k,
            min_similarity=min_similarity,
        )

        entity_ids = self._ids_from_hits(seed_entities)
        relation_ids = self._ids_from_hits(seed_relations)
        expanded_entity_ids, expanded_relation_ids, expansion_steps = self._expand_es_graph(
            entity_ids=entity_ids,
            relation_ids=relation_ids,
            entity_index_name=entity_index_name,
            relation_index_name=relation_index_name,
            degree=expansion_degree,
        )
        kept_relations, eviction = self._evict_relations_by_vector(
            query=query,
            relation_ids=expanded_relation_ids,
            relation_index_name=relation_index_name,
            limit=relation_limit,
        )
        passages = self._search_passages_by_graph(
            query=query,
            index_name=index_name,
            relation_ids=kept_relations,
            entity_ids=expanded_entity_ids,
            k=k,
        )

        if not passages:
            passages = self.retrieve(query=query, k=k, index_name=index_name)

        if not return_debug:
            return passages[:k]

        return {
            "query": query,
            "query_entities": query_entities,
            "passages": passages[:k],
            "seed_entity_ids": entity_ids,
            "seed_relation_ids": relation_ids,
            "expanded_entity_ids": expanded_entity_ids,
            "expanded_relation_ids": expanded_relation_ids,
            "kept_relation_ids": kept_relations,
            "eviction": eviction,
            "expansion_steps": expansion_steps,
        }

    def add(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        index_name: Optional[str] = None,
    ) -> str:
        if not index_name:
            raise ValueError("index_name is required for add operations")

        metadata = metadata or {}
        doc_type = metadata.get("doc_type")
        embedding = self.embedding_model.embed_query(content)
        self.ensure_index(index_name=index_name, vector_dims=len(embedding))

        doc_body = {
            "content": content,
            "doc_type": doc_type,
            "embedding": embedding,
            "metadata": metadata,
        }
        result = self.es_client.index(index=index_name, id=doc_id, document=doc_body, refresh=True)
        logger.info("Document indexed: id={}, index={}", result["_id"], index_name)
        return result["_id"]

    def add_batch(
        self,
        documents: List[Dict[str, Any]],
        index_name: Optional[str] = None,
    ) -> List[str]:
        if not index_name:
            raise ValueError("index_name is required for add_batch operations")
        if not documents:
            return []

        texts_to_embed: List[str] = []
        embedding_positions: List[int] = []
        prepared_docs: List[Dict[str, Any]] = []

        for document in documents:
            content = document.get("content", "")
            embedding = document.get("embedding")
            prepared_docs.append(document)
            if embedding is None:
                texts_to_embed.append(content)
                embedding_positions.append(len(prepared_docs) - 1)

        generated_embeddings = self.embedding_model.embed_documents(texts_to_embed) if texts_to_embed else []
        for position, embedding in zip(embedding_positions, generated_embeddings):
            prepared_docs[position]["embedding"] = embedding

        first_embedding = prepared_docs[0].get("embedding")
        if not first_embedding:
            raise ValueError("At least one embedding is required to create the index")
        self.ensure_index(index_name=index_name, vector_dims=len(first_embedding))

        operations: List[Dict[str, Any]] = []
        for document in prepared_docs:
            doc_id = document.get("doc_id")
            operation: Dict[str, Any] = {"index": {"_index": index_name}}
            if doc_id is not None:
                operation["index"]["_id"] = doc_id

            metadata = document.get("metadata", {}) or {}
            doc_type = document.get("doc_type") or metadata.get("doc_type")
            payload = {
                "content": document.get("content", ""),
                "doc_type": doc_type,
                "embedding": document["embedding"],
                "metadata": metadata,
            }
            for key, value in document.items():
                if key in {"content", "doc_id", "doc_type", "embedding", "metadata"}:
                    continue
                payload[key] = value

            operations.append(operation)
            operations.append(payload)

        result = self.es_client.bulk(operations=operations, refresh=True)
        ids = [item["index"]["_id"] for item in result["items"]]
        logger.info("Bulk index completed: count={}, index={}", len(ids), index_name)
        return ids

    def update(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None,
    ) -> bool:
        if not index_name:
            raise ValueError("index_name is required for update operations")

        update_body: Dict[str, Any] = {}
        if content is not None:
            update_body["content"] = content
            update_body["embedding"] = self.embedding_model.embed_query(content)
        if metadata is not None:
            update_body["metadata"] = metadata
            if "doc_type" in metadata:
                update_body["doc_type"] = metadata["doc_type"]

        if not update_body:
            logger.warning("Document update skipped because no fields were provided: id={}", doc_id)
            return False

        result = self.es_client.update(index=index_name, id=doc_id, doc=update_body, refresh=True)
        return result["result"] in {"updated", "noop"}

    def delete(self, doc_id: str, index_name: Optional[str] = None) -> bool:
        if not index_name:
            raise ValueError("index_name is required for delete operations")
        result = self.es_client.delete(index=index_name, id=doc_id, refresh=True)
        return result["result"] == "deleted"

    def delete_batch(self, doc_ids: List[str], index_name: Optional[str] = None) -> List[bool]:
        if not index_name:
            raise ValueError("index_name is required for delete_batch operations")
        if not doc_ids:
            return []

        operations = [{"delete": {"_index": index_name, "_id": doc_id}} for doc_id in doc_ids]
        result = self.es_client.bulk(operations=operations, refresh=True)
        return [item["delete"]["result"] == "deleted" for item in result["items"]]

    def get(self, doc_id: str, index_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not index_name:
            raise ValueError("index_name is required for get operations")
        if not self.index_exists(index_name=index_name):
            return None
        try:
            result = self.es_client.get(index=index_name, id=doc_id)
        except Exception as exc:
            logger.error("Get document failed: id={}, error={}", doc_id, exc)
            return None

        source = result["_source"]
        return {
            "id": doc_id,
            "content": source.get("content", ""),
            "doc_type": source.get("doc_type"),
            "metadata": source.get("metadata", {}),
        }

    def search(
        self,
        query: Optional[str] = None,
        k: int = 3,
        filter_conditions: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not index_name:
            raise ValueError("index_name is required for search operations")
        if not self.index_exists(index_name=index_name):
            return []

        filters = self._build_filter_clauses(filter_conditions=filter_conditions)
        if query:
            body: Dict[str, Any] = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["content^2", "title", "summary"],
                                    "type": "best_fields",
                                }
                            }
                        ]
                    }
                }
            }
            if filters:
                body["query"]["bool"]["filter"] = filters
        elif filters:
            body = {"query": {"bool": {"filter": filters}}}
        else:
            body = {"query": {"match_all": {}}}

        results = self.es_client.search(index=index_name, body=body, size=k)
        return [self._hit_to_result(hit) for hit in results["hits"]["hits"]]

    def search_all(
        self,
        query: Optional[str] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not index_name:
            raise ValueError("index_name is required for search_all operations")
        if not self.index_exists(index_name=index_name):
            return []

        filters = self._build_filter_clauses(filter_conditions=filter_conditions)
        if query:
            body: Dict[str, Any] = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["content^2", "title", "summary"],
                                    "type": "best_fields",
                                }
                            }
                        ]
                    }
                }
            }
            if filters:
                body["query"]["bool"]["filter"] = filters
        elif filters:
            body = {"query": {"bool": {"filter": filters}}}
        else:
            body = {"query": {"match_all": {}}}

        return [
            self._hit_to_result(hit)
            for hit in scan(
                client=self.es_client,
                index=index_name,
                query=body,
                preserve_order=False,
            )
        ]

    def exists(self, doc_id: str, index_name: Optional[str] = None) -> bool:
        if not index_name:
            raise ValueError("index_name is required for exists operations")
        if not self.index_exists(index_name=index_name):
            return False
        return self.es_client.exists(index=index_name, id=doc_id)

    def existing_ids(self, doc_ids: Iterable[str], index_name: Optional[str] = None) -> set[str]:
        if not index_name:
            raise ValueError("index_name is required for existing_ids operations")
        if not self.index_exists(index_name=index_name):
            return set()

        ids = [doc_id for doc_id in doc_ids if doc_id]
        if not ids:
            return set()

        results = self.es_client.mget(index=index_name, ids=ids)
        return {doc["_id"] for doc in results.get("docs", []) if doc.get("found")}

    def get_many(self, doc_ids: Iterable[str], index_name: Optional[str] = None) -> List[Dict[str, Any]]:
        if not index_name:
            raise ValueError("index_name is required for get_many operations")
        if not self.index_exists(index_name=index_name):
            return []
        return self._get_docs_by_ids(index_name=index_name, doc_ids=doc_ids)

    def count(
        self,
        filter_conditions: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None,
    ) -> int:
        if not index_name:
            raise ValueError("index_name is required for count operations")
        if not self.index_exists(index_name=index_name):
            return 0

        filters = self._build_filter_clauses(filter_conditions=filter_conditions)
        body = {"query": {"bool": {"filter": filters}}} if filters else {"query": {"match_all": {}}}
        result = self.es_client.count(index=index_name, body=body)
        return result["count"]

    def _search_graph_items(
        self,
        texts: List[str],
        index_name: str,
        k: int,
        min_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        hits: List[Dict[str, Any]] = []
        seen_ids = set()
        for text in texts:
            if not text.strip():
                continue
            for item in self._vector_search_raw(text, k, index_name, min_similarity=min_similarity):
                if item["id"] in seen_ids:
                    continue
                seen_ids.add(item["id"])
                hits.append(item)
        return hits

    def _vector_search_raw(
        self,
        query: str,
        k: int,
        index_name: str,
        min_similarity: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        query_vector = self.embedding_model.embed_query(query)
        self.ensure_index(index_name=index_name, vector_dims=len(query_vector))

        filters = self._build_filter_clauses(filter_conditions=filter_conditions, ids=ids)
        knn_query: Dict[str, Any] = {
            "field": "embedding",
            "query_vector": query_vector,
            "num_candidates": max(k * 4, 20),
        }
        if filters:
            knn_query["filter"] = {"bool": {"must": filters}}

        results = self.es_client.search(index=index_name, body={"query": {"knn": knn_query}}, size=k)

        hits = []
        for hit in results["hits"]["hits"]:
            score = hit.get("_score", 0.0)
            if min_similarity is not None and score < min_similarity:
                continue
            hits.append(self._hit_to_result(hit))
        return hits

    def _expand_es_graph(
        self,
        entity_ids: List[str],
        relation_ids: List[str],
        entity_index_name: str,
        relation_index_name: str,
        degree: int,
    ) -> tuple[List[str], List[str], List[Dict[str, Any]]]:
        all_entity_ids = set(entity_ids)
        all_relation_ids = set(relation_ids)
        steps: List[Dict[str, Any]] = []

        relation_ids_from_seed_entities = self._relations_by_entities(
            entity_ids=list(all_entity_ids),
            entity_index_name=entity_index_name,
            relation_index_name=relation_index_name,
        )
        new_relation_ids = relation_ids_from_seed_entities - all_relation_ids
        all_relation_ids.update(new_relation_ids)
        steps.append(
            {
                "step": 0,
                "operation": "entity_to_relation",
                "new_entity_ids": [],
                "new_relation_ids": sorted(new_relation_ids),
            }
        )

        for step in range(1, degree + 1):
            found_entity_ids = self._entities_by_relations(
                relation_ids=list(all_relation_ids),
                relation_index_name=relation_index_name,
            )
            new_entity_ids = found_entity_ids - all_entity_ids
            all_entity_ids.update(new_entity_ids)

            found_relation_ids = self._relations_by_entities(
                entity_ids=list(new_entity_ids),
                entity_index_name=entity_index_name,
                relation_index_name=relation_index_name,
            )
            new_relation_ids = found_relation_ids - all_relation_ids
            all_relation_ids.update(new_relation_ids)

            steps.append(
                {
                    "step": step,
                    "operation": "relation_to_entity_to_relation",
                    "new_entity_ids": sorted(new_entity_ids),
                    "new_relation_ids": sorted(new_relation_ids),
                }
            )
            if not new_entity_ids and not new_relation_ids:
                break

        return sorted(all_entity_ids), sorted(all_relation_ids), steps

    def _relations_by_entities(
        self,
        entity_ids: List[str],
        entity_index_name: str,
        relation_index_name: str,
    ) -> set[str]:
        relation_ids: set[str] = set()
        for entity in self._get_docs_by_ids(entity_index_name, entity_ids):
            relation_ids.update(self._metadata_list(entity, "relation_ids"))

        if relation_ids:
            return relation_ids

        for relation in self._search_by_terms(
            index_name=relation_index_name,
            field="metadata.entity_ids",
            values=entity_ids,
            size=max(len(entity_ids) * 20, 50),
        ):
            relation_ids.add(relation["id"])
        return relation_ids

    def _entities_by_relations(self, relation_ids: List[str], relation_index_name: str) -> set[str]:
        entity_ids: set[str] = set()
        for relation in self._get_docs_by_ids(relation_index_name, relation_ids):
            entity_ids.update(self._metadata_list(relation, "entity_ids"))
        return entity_ids

    def _evict_relations_by_vector(
        self,
        query: str,
        relation_ids: List[str],
        relation_index_name: str,
        limit: int,
    ) -> tuple[List[str], Dict[str, Any]]:
        before_count = len(relation_ids)
        if before_count <= limit:
            return sorted(relation_ids), {
                "occurred": False,
                "before_count": before_count,
                "after_count": before_count,
            }

        kept = [
            hit["id"]
            for hit in self._vector_search_raw(
                query=query,
                k=limit,
                index_name=relation_index_name,
                ids=relation_ids,
            )
        ]
        return kept, {
            "occurred": True,
            "before_count": before_count,
            "after_count": len(kept),
        }

    def _search_passages_by_graph(
        self,
        query: str,
        index_name: str,
        relation_ids: List[str],
        entity_ids: List[str],
        k: int,
    ) -> List[Dict[str, Any]]:
        should_clauses = []
        if relation_ids:
            should_clauses.append({"terms": {"metadata.relation_ids": relation_ids}})
        if entity_ids:
            should_clauses.append({"terms": {"metadata.entity_ids": entity_ids}})
        if query:
            should_clauses.append(
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^2", "title", "summary"],
                        "type": "best_fields",
                    }
                }
            )
        if not should_clauses:
            return []

        results = self.es_client.search(
            index=index_name,
            body={"query": {"bool": {"should": should_clauses, "minimum_should_match": 1}}},
            size=k,
        )
        return [self._hit_to_result(hit) for hit in results["hits"]["hits"]]

    def _get_docs_by_ids(self, index_name: str, doc_ids: Iterable[str]) -> List[Dict[str, Any]]:
        ids = [doc_id for doc_id in doc_ids if doc_id]
        if not ids:
            return []

        results = self.es_client.mget(index=index_name, ids=ids)
        docs = []
        found_ids = set()
        for doc in results.get("docs", []):
            if not doc.get("found"):
                continue
            source = doc.get("_source", {})
            metadata = source.get("metadata", {})
            found_ids.add(doc["_id"])
            if metadata.get("id") is not None:
                found_ids.add(str(metadata["id"]))
            docs.append(
                {
                    "id": doc["_id"],
                    "content": source.get("content", ""),
                    "doc_type": source.get("doc_type"),
                    "metadata": metadata,
                }
            )

        missed_ids = [doc_id for doc_id in ids if doc_id not in found_ids]
        docs.extend(self._search_by_terms(index_name=index_name, field="metadata.id", values=missed_ids, size=len(missed_ids)))
        return docs

    def _search_by_terms(self, index_name: str, field: str, values: List[str], size: int) -> List[Dict[str, Any]]:
        if not values:
            return []
        results = self.es_client.search(index=index_name, body={"query": {"terms": {field: values}}}, size=size)
        return [self._hit_to_result(hit) for hit in results["hits"]["hits"]]

    def _hit_to_result(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        source = hit.get("_source", {})
        metadata = source.get("metadata", {})
        result_id = str(metadata.get("id") or hit.get("_id"))
        return {
            "id": result_id,
            "es_id": hit.get("_id"),
            "content": source.get("content", ""),
            "doc_type": source.get("doc_type"),
            "metadata": metadata,
            "score": hit.get("_score"),
        }

    def _ids_from_hits(self, hits: List[Dict[str, Any]]) -> List[str]:
        return list(dict.fromkeys(str(hit["id"]) for hit in hits))

    def _metadata_list(self, doc: Dict[str, Any], key: str) -> List[str]:
        value = doc.get("metadata", {}).get(key, [])
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]

    def _simple_extract_entities(self, query: str) -> List[str]:
        cleaned_words = []
        for raw_word in query.split():
            word = raw_word.strip("'\".,;:!?()[]{}<>")
            if len(word) >= 2:
                cleaned_words.append(word)
        return list(dict.fromkeys(cleaned_words))[:8]

    def _build_filter_clauses(
        self,
        filter_conditions: Optional[Dict[str, Any]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        clauses: List[Dict[str, Any]] = []
        if ids:
            clauses.append({"ids": {"values": ids}})

        if not filter_conditions:
            return clauses

        for field, value in filter_conditions.items():
            if isinstance(value, (list, tuple, set)):
                clauses.append({"terms": {field: list(value)}})
            else:
                clauses.append({"term": {field: value}})
        return clauses
