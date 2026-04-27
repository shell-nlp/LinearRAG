from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from src.config import LinearRAGConfig
from src.document_store import ElasticsearchDocumentStore
from src.knowledge_graph import IgraphKnowledgeGraph, KnowledgeGraph
from src.ner import SpacyNER
from src.utils import compute_mdhash_id, min_max_normalize

logger = logging.getLogger(__name__)


class LinearRAG:
    def __init__(
        self,
        global_config: LinearRAGConfig,
        llm_model,
        embedding_model,
        graph: KnowledgeGraph | None = None,
    ):
        self.config = global_config
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.spacy_ner = SpacyNER(self.config.spacy_model)
        self.graph = graph or IgraphKnowledgeGraph()
        self.document_store = ElasticsearchDocumentStore(self.config, embedding_model=self.embedding_model)
        self._reset_runtime_state()

    def _reset_runtime_state(self) -> None:
        """重置当前进程内的运行态。ES 才是持久化数据源，这里只保留本次检索需要的内存索引。"""
        self.graph.clear()
        self.ordered_passage_ids: list[str] = []
        self.passage_hash_id_to_text: dict[str, str] = {}
        self.entity_hash_id_to_text: dict[str, str] = {}
        self.sentence_hash_id_to_text: dict[str, str] = {}
        self.passage_hash_id_to_entity_hash_ids: dict[str, list[str]] = {}
        self.passage_hash_id_to_sentence_hash_ids: dict[str, list[str]] = {}
        self.entity_hash_id_to_sentence_hash_ids: dict[str, list[str]] = {}
        self.sentence_hash_id_to_entity_hash_ids: dict[str, list[str]] = {}

    def qa(self, questions: list[dict[str, str]]) -> list[dict[str, object]]:
        retrieval_results = self.retrieve(questions)
        system_prompt = (
            "As an advanced reading comprehension assistant, your task is to analyze text passages "
            "and corresponding questions meticulously. Your response start after \"Thought: \", where "
            "you will methodically break down the reasoning process, illustrating how you arrive at "
            "conclusions. Conclude with \"Answer: \" to present a concise, definitive response, "
            "devoid of additional elaborations."
        )

        all_messages = []
        for retrieval_result in retrieval_results:
            passage_context = "\n".join(retrieval_result["sorted_passage"])
            question = retrieval_result["question"]
            all_messages.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{passage_context}\nQuestion: {question}\n Thought: "},
                ]
            )

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            all_qa_results = list(
                tqdm(
                    executor.map(self.llm_model.infer, all_messages),
                    total=len(all_messages),
                    desc="QA Reading",
                )
            )

        for qa_result, question_info in zip(all_qa_results, retrieval_results):
            try:
                pred_answer = qa_result.split("Answer:", 1)[1].strip()
            except Exception:
                pred_answer = qa_result
            question_info["pred_answer"] = pred_answer
        return retrieval_results

    def retrieve(self, questions: list[dict[str, str]]) -> list[dict[str, object]]:
        retrieval_results = []
        for question_info in tqdm(questions, desc="Retrieving"):
            question = question_info["question"]
            seed_entities = self.get_seed_entities(question)

            if seed_entities:
                ranked_passages = self.graph_search_with_seed_entities(question, seed_entities)
                final_hits = ranked_passages[: self.config.retrieval_top_k]
            else:
                final_hits = self.dense_passage_retrieval(question, self.config.retrieval_top_k)

            if not final_hits:
                final_hits = self.dense_passage_retrieval(question, self.config.retrieval_top_k)

            if not final_hits:
                final_hits = self.document_store.keyword_search(
                    query=question,
                    doc_type="passage",
                    k=self.config.retrieval_top_k,
                )

            retrieval_results.append(
                {
                    "question": question,
                    "sorted_passage": [hit["content"] for hit in final_hits],
                    "sorted_passage_scores": [hit.get("score") for hit in final_hits],
                    "gold_answer": question_info.get("answer", ""),
                }
            )
        return retrieval_results

    def get_seed_entities(self, question: str) -> list[dict[str, object]]:
        question_entities = sorted(self.spacy_ner.question_ner(question))
        if not question_entities:
            return []

        seeds = []
        seen_entity_ids = set()
        for entity_text in question_entities:
            hits = self.document_store.vector_search(query=entity_text, doc_type="entity", k=1)
            if not hits:
                continue
            hit = hits[0]
            entity_hash_id = hit["id"]
            if entity_hash_id in seen_entity_ids:
                continue
            seen_entity_ids.add(entity_hash_id)
            seeds.append(
                {
                    "entity_hash_id": entity_hash_id,
                    "entity_text": hit["content"],
                    "score": float(hit.get("score") or 0.0),
                }
            )
        return seeds

    def graph_search_with_seed_entities(
        self,
        question: str,
        seed_entities: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        entity_weights, active_entities = self.calculate_entity_scores(question, seed_entities)
        passage_weights = self.calculate_passage_scores(question, active_entities)
        reset_weights = self._merge_weights(entity_weights, passage_weights)
        ranked_passages = self.graph.personalized_pagerank(
            reset_weights=reset_weights,
            damping=self.config.damping,
            node_type="passage",
        )

        results = []
        for passage_hash_id, score in ranked_passages:
            passage_text = self.passage_hash_id_to_text.get(passage_hash_id)
            if passage_text is None:
                continue
            results.append({"id": passage_hash_id, "content": passage_text, "score": score})
        return results

    def calculate_entity_scores(
        self,
        question: str,
        seed_entities: list[dict[str, object]],
    ) -> tuple[dict[str, float], dict[str, tuple[float, int]]]:
        """论文中的第一阶段：从 seed entity 出发，经由 sentence 节点做逐层扩展。"""
        entity_weights: dict[str, float] = defaultdict(float)
        active_entities: dict[str, tuple[float, int]] = {}
        current_entities: dict[str, tuple[float, int]] = {}
        used_sentence_hash_ids: set[str] = set()

        for seed in seed_entities:
            entity_hash_id = str(seed["entity_hash_id"])
            score = float(seed["score"])
            entity_weights[entity_hash_id] += score
            active_entities[entity_hash_id] = (score, 1)
            current_entities[entity_hash_id] = (score, 1)

        for iteration in range(1, self.config.max_iterations):
            new_entities: dict[str, tuple[float, int]] = {}
            for entity_hash_id, (entity_score, _) in current_entities.items():
                if entity_score < self.config.iteration_threshold:
                    continue

                sentence_hash_ids = [
                    sentence_hash_id
                    for sentence_hash_id in self.entity_hash_id_to_sentence_hash_ids.get(entity_hash_id, [])
                    if sentence_hash_id not in used_sentence_hash_ids
                ]
                if not sentence_hash_ids:
                    continue

                sentence_hits = self.document_store.vector_search(
                    query=question,
                    doc_type="sentence",
                    k=min(self.config.top_k_sentence, len(sentence_hash_ids)),
                    ids=sentence_hash_ids,
                )
                for sentence_hit in sentence_hits:
                    sentence_hash_id = sentence_hit["id"]
                    sentence_score = float(sentence_hit.get("score") or 0.0)
                    used_sentence_hash_ids.add(sentence_hash_id)
                    for next_entity_hash_id in sentence_hit.get("metadata", {}).get("entity_hash_ids", []):
                        next_entity_score = entity_score * sentence_score
                        if next_entity_score < self.config.iteration_threshold:
                            continue
                        entity_weights[next_entity_hash_id] += next_entity_score
                        previous_score, _ = new_entities.get(next_entity_hash_id, (0.0, iteration + 1))
                        new_entities[next_entity_hash_id] = (previous_score + next_entity_score, iteration + 1)

            if not new_entities:
                break

            active_entities.update(new_entities)
            current_entities = new_entities
        return dict(entity_weights), active_entities

    def calculate_passage_scores(
        self,
        question: str,
        active_entities: dict[str, tuple[float, int]],
    ) -> dict[str, float]:
        """论文中的第二阶段：用 dense passage score + entity bonus 构造 PPR reset 权重。"""
        candidate_hits = self.dense_passage_retrieval(question, self.config.passage_search_candidate_k)
        if not candidate_hits:
            return {}

        dense_scores = min_max_normalize([float(hit.get("score") or 0.0) for hit in candidate_hits])
        question_lower = question.lower()
        apply_attribute_boost = (
            self.config.enable_hybrid_attribute_fallback and self._is_attribute_query(question)
        )

        passage_weights: dict[str, float] = {}
        for index, hit in enumerate(candidate_hits):
            passage_hash_id = hit["id"]
            passage_text_lower = hit["content"].lower()
            dense_score = float(dense_scores[index])
            total_entity_bonus = 0.0

            for entity_hash_id, (entity_score, tier) in active_entities.items():
                entity_text = self.entity_hash_id_to_text.get(entity_hash_id)
                if not entity_text:
                    continue
                entity_occurrences = passage_text_lower.count(entity_text.lower())
                if entity_occurrences <= 0:
                    continue
                total_entity_bonus += entity_score * math.log(1 + entity_occurrences) / max(tier, 1)

            passage_score = self.config.passage_ratio * dense_score + math.log(1 + total_entity_bonus)
            if apply_attribute_boost:
                overlap = self._attribute_keyword_overlap(question_lower, passage_text_lower)
                if overlap > 0:
                    passage_score += self.config.attribute_keyword_boost * math.log(1 + overlap)

            passage_weights[passage_hash_id] = passage_score * self.config.passage_node_weight
        return passage_weights

    def dense_passage_retrieval(self, question: str, top_k: int) -> list[dict[str, object]]:
        hits = self.document_store.vector_search(query=question, doc_type="passage", k=top_k)
        if hits:
            return hits
        return self.document_store.keyword_search(query=question, doc_type="passage", k=top_k)

    def _merge_weights(self, *weight_maps: dict[str, float]) -> dict[str, float]:
        merged: dict[str, float] = defaultdict(float)
        for weight_map in weight_maps:
            for node_id, score in weight_map.items():
                if score > 0:
                    merged[node_id] += score
        return dict(merged)

    def _is_attribute_query(self, question: str) -> bool:
        tokens = set(re.findall(r"\w+", question.lower()))
        return any(keyword in tokens for keyword in self.config.attribute_query_keywords)

    def _attribute_keyword_overlap(self, question_lower: str, passage_text_lower: str) -> int:
        overlap = 0
        for keyword in self.config.attribute_query_keywords:
            if keyword in question_lower and keyword in passage_text_lower:
                overlap += 1
        return overlap

    def index(self, passages: list[str]) -> None:
        """
        以 ES 为唯一持久化存储构建索引。

        处理顺序：
        1. 先根据当前输入生成 passage id。
        2. 去 ES 判断哪些 passage 已存在，只对新增 passage 做 NER。
        3. 把新增 passage 产生的 passage/entity/sentence 文档写回 ES。
        4. 再从 ES 回读当前数据集状态，重建运行时图结构。
        """
        logger.info("Indexing dataset=%s into Elasticsearch index=%s", self.config.dataset_name, self.config.index_name)
        self._reset_runtime_state()
        self._prepare_passages(passages)

        existing_passage_ids, new_passage_ids = self._split_existing_and_new_passages()
        logger.info(
            "ES passage state: existing=%s, new=%s",
            len(existing_passage_ids),
            len(new_passage_ids),
        )

        self._sync_existing_passage_metadata(existing_passage_ids)

        if new_passage_ids:
            (
                new_passage_hash_id_to_entities,
                new_sentence_to_entities,
                new_passage_hash_id_to_sentences,
            ) = self._extract_new_passage_state(new_passage_ids)
            self._persist_new_state_to_es(
                new_passage_hash_id_to_entities=new_passage_hash_id_to_entities,
                new_sentence_to_entities=new_sentence_to_entities,
                new_passage_hash_id_to_sentences=new_passage_hash_id_to_sentences,
            )

        # 最终运行态全部从 ES 回读，避免内存和 ES 状态不一致。
        self._load_runtime_state_from_es()
        self._build_graph_from_runtime_state()

    def _prepare_passages(self, passages: list[str]) -> None:
        """把本次输入规范化为稳定 id，后续所有 ES 检查都基于这些 id。"""
        self.passage_hash_id_to_text = {
            compute_mdhash_id(passage, prefix="passage-"): passage for passage in passages
        }
        self.ordered_passage_ids = list(self.passage_hash_id_to_text.keys())

    def _split_existing_and_new_passages(self) -> tuple[list[str], list[str]]:
        existing_ids = self.document_store.existing_ids(self.ordered_passage_ids)
        existing_passage_ids = [passage_id for passage_id in self.ordered_passage_ids if passage_id in existing_ids]
        new_passage_ids = [passage_id for passage_id in self.ordered_passage_ids if passage_id not in existing_ids]
        return existing_passage_ids, new_passage_ids

    def _sync_existing_passage_metadata(self, existing_passage_ids: list[str]) -> None:
        """已有 passage 不需要重新做 NER，但要把本次顺序信息同步回 ES。"""
        if not existing_passage_ids:
            return

        sequence_index_by_passage_id = {
            passage_id: index for index, passage_id in enumerate(self.ordered_passage_ids)
        }
        for passage_doc in self.document_store.get_documents(doc_type="passage", ids=existing_passage_ids):
            metadata = dict(passage_doc.get("metadata", {}))
            metadata["dataset_name"] = self.config.dataset_name
            metadata["sequence_index"] = sequence_index_by_passage_id[passage_doc["id"]]
            self.document_store.update_metadata(doc_id=passage_doc["id"], metadata=metadata)

    def _extract_new_passage_state(
        self,
        new_passage_ids: list[str],
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
        new_hash_id_to_passage = {
            passage_hash_id: self.passage_hash_id_to_text[passage_hash_id]
            for passage_hash_id in new_passage_ids
        }
        return self.spacy_ner.batch_ner(new_hash_id_to_passage, self.config.max_workers)

    def _persist_new_state_to_es(
        self,
        new_passage_hash_id_to_entities: dict[str, list[str]],
        new_sentence_to_entities: dict[str, list[str]],
        new_passage_hash_id_to_sentences: dict[str, list[str]],
    ) -> None:
        """
        仅把新增 passage 产生的数据写入 ES。

        entity/sentence 文档可能已存在，因此这里先读取 ES 里的旧关系，再做集合合并后回写。
        """
        existing_entity_docs = self.document_store.get_documents(doc_type="entity")
        existing_sentence_docs = self.document_store.get_documents(doc_type="sentence")

        entity_hash_id_to_text = {doc["id"]: doc["content"] for doc in existing_entity_docs}
        entity_hash_id_to_sentence_hash_ids = {
            doc["id"]: set(self._metadata_ids(doc, "sentence_hash_ids")) for doc in existing_entity_docs
        }
        sentence_hash_id_to_text = {doc["id"]: doc["content"] for doc in existing_sentence_docs}
        sentence_hash_id_to_entity_hash_ids = {
            doc["id"]: set(self._metadata_ids(doc, "entity_hash_ids")) for doc in existing_sentence_docs
        }
        sentence_hash_id_to_passage_hash_ids = {
            doc["id"]: set(self._metadata_ids(doc, "passage_hash_ids")) for doc in existing_sentence_docs
        }

        passage_hash_id_to_entity_hash_ids: dict[str, list[str]] = {}
        passage_hash_id_to_sentence_hash_ids: dict[str, list[str]] = {}

        for sentence_text, entity_texts in new_sentence_to_entities.items():
            sentence_hash_id = compute_mdhash_id(sentence_text, prefix="sentence-")
            sentence_hash_id_to_text[sentence_hash_id] = sentence_text
            for entity_text in entity_texts:
                entity_hash_id = compute_mdhash_id(entity_text, prefix="entity-")
                entity_hash_id_to_text[entity_hash_id] = entity_text
                entity_hash_id_to_sentence_hash_ids.setdefault(entity_hash_id, set()).add(sentence_hash_id)
                sentence_hash_id_to_entity_hash_ids.setdefault(sentence_hash_id, set()).add(entity_hash_id)

        for passage_hash_id, sentence_texts in new_passage_hash_id_to_sentences.items():
            sentence_hash_ids = [
                compute_mdhash_id(sentence_text, prefix="sentence-")
                for sentence_text in sentence_texts
            ]
            passage_hash_id_to_sentence_hash_ids[passage_hash_id] = sentence_hash_ids
            for sentence_hash_id in sentence_hash_ids:
                sentence_hash_id_to_passage_hash_ids.setdefault(sentence_hash_id, set()).add(passage_hash_id)

        for passage_hash_id, entity_texts in new_passage_hash_id_to_entities.items():
            entity_hash_ids = [
                compute_mdhash_id(entity_text, prefix="entity-")
                for entity_text in entity_texts
            ]
            passage_hash_id_to_entity_hash_ids[passage_hash_id] = sorted(entity_hash_ids)

        sequence_index_by_passage_id = {
            passage_id: index for index, passage_id in enumerate(self.ordered_passage_ids)
        }

        passage_documents = []
        for passage_hash_id, entity_texts in new_passage_hash_id_to_entities.items():
            passage_text = self.passage_hash_id_to_text[passage_hash_id]
            passage_documents.append(
                self.document_store.build_document(
                    doc_type="passage",
                    content=passage_text,
                    metadata={
                        "dataset_name": self.config.dataset_name,
                        "sequence_index": sequence_index_by_passage_id[passage_hash_id],
                        "entity_hash_ids": passage_hash_id_to_entity_hash_ids.get(passage_hash_id, []),
                        "entity_texts": entity_texts,
                        "sentence_hash_ids": passage_hash_id_to_sentence_hash_ids.get(passage_hash_id, []),
                    },
                )
            )

        entity_documents = []
        updated_entity_ids = {
            compute_mdhash_id(entity_text, prefix="entity-")
            for entity_texts in new_passage_hash_id_to_entities.values()
            for entity_text in entity_texts
        }
        for entity_hash_id in sorted(updated_entity_ids):
            entity_documents.append(
                self.document_store.build_document(
                    doc_type="entity",
                    content=entity_hash_id_to_text[entity_hash_id],
                    metadata={
                        "dataset_name": self.config.dataset_name,
                        "sentence_hash_ids": sorted(entity_hash_id_to_sentence_hash_ids.get(entity_hash_id, set())),
                    },
                )
            )

        sentence_documents = []
        updated_sentence_ids = {
            compute_mdhash_id(sentence_text, prefix="sentence-")
            for sentence_text in new_sentence_to_entities.keys()
        }
        for sentence_hash_id in sorted(updated_sentence_ids):
            sentence_documents.append(
                self.document_store.build_document(
                    doc_type="sentence",
                    content=sentence_hash_id_to_text[sentence_hash_id],
                    metadata={
                        "dataset_name": self.config.dataset_name,
                        "entity_hash_ids": sorted(sentence_hash_id_to_entity_hash_ids.get(sentence_hash_id, set())),
                        "passage_hash_ids": sorted(sentence_hash_id_to_passage_hash_ids.get(sentence_hash_id, set())),
                    },
                )
            )

        self.document_store.upsert_documents(passage_documents)
        self.document_store.upsert_documents(entity_documents)
        self.document_store.upsert_documents(sentence_documents)

    def _load_runtime_state_from_es(self) -> None:
        """
        从 ES 回读当前数据集对应的 passage/entity/sentence 文档。

        注意：
        - 这里只回读本次输入 passage 对应的文档，不把 ES 中其他无关 passage 混进来。
        - entity 和 sentence 则按 passage 元数据里的引用继续向外展开。
        """
        passage_docs = {
            doc["id"]: doc
            for doc in self.document_store.get_documents(doc_type="passage", ids=self.ordered_passage_ids)
        }

        self.ordered_passage_ids = [
            passage_id for passage_id in self.ordered_passage_ids if passage_id in passage_docs
        ]
        self.passage_hash_id_to_text = {
            passage_id: passage_docs[passage_id]["content"] for passage_id in self.ordered_passage_ids
        }
        self.passage_hash_id_to_entity_hash_ids = {
            passage_id: self._metadata_ids(passage_docs[passage_id], "entity_hash_ids")
            for passage_id in self.ordered_passage_ids
        }
        self.passage_hash_id_to_sentence_hash_ids = {
            passage_id: self._metadata_ids(passage_docs[passage_id], "sentence_hash_ids")
            for passage_id in self.ordered_passage_ids
        }

        entity_ids = sorted(
            {
                entity_hash_id
                for entity_hash_ids in self.passage_hash_id_to_entity_hash_ids.values()
                for entity_hash_id in entity_hash_ids
            }
        )
        entity_docs = self.document_store.get_documents(doc_type="entity", ids=entity_ids)
        self.entity_hash_id_to_text = {doc["id"]: doc["content"] for doc in entity_docs}
        self.entity_hash_id_to_sentence_hash_ids = {
            doc["id"]: self._metadata_ids(doc, "sentence_hash_ids")
            for doc in entity_docs
        }

        sentence_ids = sorted(
            {
                sentence_hash_id
                for sentence_hash_ids in self.entity_hash_id_to_sentence_hash_ids.values()
                for sentence_hash_id in sentence_hash_ids
            }
            | {
                sentence_hash_id
                for sentence_hash_ids in self.passage_hash_id_to_sentence_hash_ids.values()
                for sentence_hash_id in sentence_hash_ids
            }
        )
        sentence_docs = self.document_store.get_documents(doc_type="sentence", ids=sentence_ids)
        self.sentence_hash_id_to_text = {doc["id"]: doc["content"] for doc in sentence_docs}
        self.sentence_hash_id_to_entity_hash_ids = {
            doc["id"]: self._metadata_ids(doc, "entity_hash_ids")
            for doc in sentence_docs
        }

        # 某些旧数据可能没有 sentence -> entity 元数据，这里用 entity 的反向引用补齐。
        for entity_hash_id, sentence_hash_ids in self.entity_hash_id_to_sentence_hash_ids.items():
            for sentence_hash_id in sentence_hash_ids:
                self.sentence_hash_id_to_entity_hash_ids.setdefault(sentence_hash_id, [])
                if entity_hash_id not in self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id]:
                    self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id].append(entity_hash_id)

        for sentence_hash_id in list(self.sentence_hash_id_to_entity_hash_ids.keys()):
            self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id] = sorted(
                set(self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id])
            )

        logger.info(
            "Loaded runtime state from ES: passages=%s, entities=%s, sentences=%s",
            len(self.passage_hash_id_to_text),
            len(self.entity_hash_id_to_text),
            len(self.sentence_hash_id_to_text),
        )

    def _build_graph_from_runtime_state(self) -> None:
        """图只在内存里重建，不再写本地 graphml。"""
        self.graph.clear()
        for passage_hash_id, passage_text in self.passage_hash_id_to_text.items():
            self.graph.add_node(passage_hash_id, passage_text, "passage")
        for entity_hash_id, entity_text in self.entity_hash_id_to_text.items():
            self.graph.add_node(entity_hash_id, entity_text, "entity")

        self._add_entity_to_passage_edges()
        self._add_adjacent_passage_edges()

    def _add_adjacent_passage_edges(self) -> None:
        for current_passage_id, next_passage_id in zip(self.ordered_passage_ids, self.ordered_passage_ids[1:]):
            self.graph.add_edge(current_passage_id, next_passage_id, weight=1.0)

    def _add_entity_to_passage_edges(self) -> None:
        for passage_hash_id, entity_hash_ids in self.passage_hash_id_to_entity_hash_ids.items():
            passage_text = self.passage_hash_id_to_text[passage_hash_id]
            entity_counts = {}
            total_mentions = 0
            for entity_hash_id in entity_hash_ids:
                entity_text = self.entity_hash_id_to_text.get(entity_hash_id)
                if not entity_text:
                    continue
                count = passage_text.count(entity_text)
                if count <= 0:
                    continue
                entity_counts[entity_hash_id] = count
                total_mentions += count

            if total_mentions == 0:
                continue

            for entity_hash_id, count in entity_counts.items():
                self.graph.add_edge(
                    passage_hash_id,
                    entity_hash_id,
                    weight=count / total_mentions,
                )

    def _metadata_ids(self, document: dict[str, object], key: str) -> list[str]:
        value = document.get("metadata", {}).get(key, [])
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]
