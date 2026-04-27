from __future__ import annotations

import json
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
        self.graph.clear()
        self.ordered_passage_ids: list[str] = []
        self.passage_hash_id_to_text: dict[str, str] = {}
        self.entity_hash_id_to_text: dict[str, str] = {}
        self.sentence_hash_id_to_text: dict[str, str] = {}
        self.passage_hash_id_to_entity_hash_ids: dict[str, list[str]] = {}
        self.entity_hash_id_to_sentence_hash_ids: dict[str, list[str]] = {}
        self.sentence_hash_id_to_entity_hash_ids: dict[str, list[str]] = {}

    def load_existing_data(self, passage_hash_ids: list[str]) -> tuple[dict[str, list[str]], dict[str, list[str]], set[str]]:
        if self.config.ner_results_path.exists():
            with self.config.ner_results_path.open("r", encoding="utf-8") as file:
                existing_ner_results = json.load(file)
            existing_passage_hash_id_to_entities = existing_ner_results["passage_hash_id_to_entities"]
            existing_sentence_to_entities = existing_ner_results["sentence_to_entities"]
            existing_passage_hash_ids = set(existing_passage_hash_id_to_entities.keys())
            new_passage_hash_ids = set(passage_hash_ids) - existing_passage_hash_ids
            return existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids
        return {}, {}, set(passage_hash_ids)

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
                        combined_score = previous_score + next_entity_score
                        new_entities[next_entity_hash_id] = (combined_score, iteration + 1)

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
        logger.info("Indexing dataset=%s into Elasticsearch index=%s", self.config.dataset_name, self.config.index_name)
        self._reset_runtime_state()

        self.passage_hash_id_to_text = {
            compute_mdhash_id(passage, prefix="passage-"): passage for passage in passages
        }
        self.ordered_passage_ids = list(self.passage_hash_id_to_text.keys())

        (
            existing_passage_hash_id_to_entities,
            existing_sentence_to_entities,
            new_passage_hash_ids,
        ) = self.load_existing_data(self.ordered_passage_ids)

        if new_passage_hash_ids:
            new_hash_id_to_passage = {
                passage_hash_id: self.passage_hash_id_to_text[passage_hash_id]
                for passage_hash_id in new_passage_hash_ids
            }
            new_passage_hash_id_to_entities, new_sentence_to_entities = self.spacy_ner.batch_ner(
                new_hash_id_to_passage,
                self.config.max_workers,
            )
            self.merge_ner_results(
                existing_passage_hash_id_to_entities,
                existing_sentence_to_entities,
                new_passage_hash_id_to_entities,
                new_sentence_to_entities,
            )

        self.save_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        (
            entity_nodes,
            sentence_nodes,
            passage_hash_id_to_entities,
            entity_to_sentence,
            sentence_to_entity,
        ) = self.extract_nodes_and_edges(existing_passage_hash_id_to_entities, existing_sentence_to_entities)

        self.entity_hash_id_to_text = {
            compute_mdhash_id(entity_text, prefix="entity-"): entity_text for entity_text in sorted(entity_nodes)
        }
        self.sentence_hash_id_to_text = {
            compute_mdhash_id(sentence_text, prefix="sentence-"): sentence_text for sentence_text in sorted(sentence_nodes)
        }

        entity_text_to_hash_id = {text: hash_id for hash_id, text in self.entity_hash_id_to_text.items()}
        sentence_text_to_hash_id = {text: hash_id for hash_id, text in self.sentence_hash_id_to_text.items()}

        self.passage_hash_id_to_entity_hash_ids = {
            passage_hash_id: sorted(entity_text_to_hash_id[entity_text] for entity_text in entity_texts)
            for passage_hash_id, entity_texts in passage_hash_id_to_entities.items()
        }
        self.entity_hash_id_to_sentence_hash_ids = {
            entity_text_to_hash_id[entity_text]: sorted(sentence_text_to_hash_id[sentence_text] for sentence_text in sentence_texts)
            for entity_text, sentence_texts in entity_to_sentence.items()
        }
        self.sentence_hash_id_to_entity_hash_ids = {
            sentence_text_to_hash_id[sentence_text]: sorted(entity_text_to_hash_id[entity_text] for entity_text in entity_texts)
            for sentence_text, entity_texts in sentence_to_entity.items()
        }

        self._index_documents()
        self._build_graph()
        self.graph.save(self.config.graph_path)

    def _index_documents(self) -> None:
        passage_documents = []
        for sequence_index, (passage_hash_id, passage_text) in enumerate(self.passage_hash_id_to_text.items()):
            passage_documents.append(
                self.document_store.build_document(
                    doc_type="passage",
                    content=passage_text,
                    metadata={
                        "dataset_name": self.config.dataset_name,
                        "sequence_index": sequence_index,
                        "entity_hash_ids": self.passage_hash_id_to_entity_hash_ids.get(passage_hash_id, []),
                    },
                )
            )

        entity_documents = []
        for entity_hash_id, entity_text in self.entity_hash_id_to_text.items():
            entity_documents.append(
                self.document_store.build_document(
                    doc_type="entity",
                    content=entity_text,
                    metadata={
                        "dataset_name": self.config.dataset_name,
                        "sentence_hash_ids": self.entity_hash_id_to_sentence_hash_ids.get(entity_hash_id, []),
                    },
                )
            )

        sentence_documents = []
        for sentence_hash_id, sentence_text in self.sentence_hash_id_to_text.items():
            sentence_documents.append(
                self.document_store.build_document(
                    doc_type="sentence",
                    content=sentence_text,
                    metadata={
                        "dataset_name": self.config.dataset_name,
                        "entity_hash_ids": self.sentence_hash_id_to_entity_hash_ids.get(sentence_hash_id, []),
                    },
                )
            )

        self.document_store.upsert_documents(passage_documents)
        self.document_store.upsert_documents(entity_documents)
        self.document_store.upsert_documents(sentence_documents)

    def _build_graph(self) -> None:
        for passage_hash_id, passage_text in self.passage_hash_id_to_text.items():
            self.graph.add_node(passage_hash_id, passage_text, "passage")
        for entity_hash_id, entity_text in self.entity_hash_id_to_text.items():
            self.graph.add_node(entity_hash_id, entity_text, "entity")

        self.add_entity_to_passage_edges()
        self.add_adjacent_passage_edges()

    def add_adjacent_passage_edges(self) -> None:
        for current_passage_id, next_passage_id in zip(self.ordered_passage_ids, self.ordered_passage_ids[1:]):
            self.graph.add_edge(current_passage_id, next_passage_id, weight=1.0)

    def add_entity_to_passage_edges(self) -> None:
        for passage_hash_id, entity_hash_ids in self.passage_hash_id_to_entity_hash_ids.items():
            passage_text = self.passage_hash_id_to_text[passage_hash_id]
            entity_counts = {}
            total_mentions = 0
            for entity_hash_id in entity_hash_ids:
                entity_text = self.entity_hash_id_to_text[entity_hash_id]
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

    def extract_nodes_and_edges(
        self,
        existing_passage_hash_id_to_entities: dict[str, list[str]],
        existing_sentence_to_entities: dict[str, list[str]],
    ) -> tuple[
        set[str],
        set[str],
        dict[str, set[str]],
        dict[str, set[str]],
        dict[str, set[str]],
    ]:
        entity_nodes = set()
        sentence_nodes = set()
        passage_hash_id_to_entities: dict[str, set[str]] = defaultdict(set)
        entity_to_sentence: dict[str, set[str]] = defaultdict(set)
        sentence_to_entity: dict[str, set[str]] = defaultdict(set)

        for passage_hash_id, entities in existing_passage_hash_id_to_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_hash_id_to_entities[passage_hash_id].add(entity)

        for sentence, entities in existing_sentence_to_entities.items():
            sentence_nodes.add(sentence)
            for entity in entities:
                entity_nodes.add(entity)
                entity_to_sentence[entity].add(sentence)
                sentence_to_entity[sentence].add(entity)

        return (
            entity_nodes,
            sentence_nodes,
            passage_hash_id_to_entities,
            entity_to_sentence,
            sentence_to_entity,
        )

    def merge_ner_results(
        self,
        existing_passage_hash_id_to_entities: dict[str, list[str]],
        existing_sentence_to_entities: dict[str, list[str]],
        new_passage_hash_id_to_entities: dict[str, list[str]],
        new_sentence_to_entities: dict[str, list[str]],
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        existing_passage_hash_id_to_entities.update(new_passage_hash_id_to_entities)
        for sentence, entities in new_sentence_to_entities.items():
            merged_entities = set(existing_sentence_to_entities.get(sentence, []))
            merged_entities.update(entities)
            existing_sentence_to_entities[sentence] = sorted(merged_entities)
        return existing_passage_hash_id_to_entities, existing_sentence_to_entities

    def save_ner_results(
        self,
        existing_passage_hash_id_to_entities: dict[str, list[str]],
        existing_sentence_to_entities: dict[str, list[str]],
    ) -> None:
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        with self.config.ner_results_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "passage_hash_id_to_entities": existing_passage_hash_id_to_entities,
                    "sentence_to_entities": existing_sentence_to_entities,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
