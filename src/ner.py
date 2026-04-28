from __future__ import annotations

import logging
from collections import defaultdict

import spacy


logger = logging.getLogger(__name__)


class SpacyNER:
    def __init__(self, spacy_model: str):
        self.has_trained_ner = True
        try:
            self.spacy_model = spacy.load(spacy_model)
            if not any(name in self.spacy_model.pipe_names for name in {"parser", "senter", "sentencizer"}):
                self.spacy_model.add_pipe("sentencizer")
        except OSError:
            logger.warning(
                "spaCy model '%s' is unavailable; falling back to a blank Chinese pipeline without NER.",
                spacy_model,
            )
            self.spacy_model = spacy.blank("zh")
            if "sentencizer" not in self.spacy_model.pipe_names:
                self.spacy_model.add_pipe("sentencizer")
            self.has_trained_ner = False

    def batch_ner(
        self,
        hash_id_to_passage: dict[str, str],
        max_workers: int,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
        passage_items = list(hash_id_to_passage.items())
        if not passage_items:
            return {}, {}, {}

        batch_size = max(len(passage_items) // max(max_workers, 1), 1)
        docs = self.spacy_model.pipe((text for _, text in passage_items), batch_size=batch_size)

        passage_hash_id_to_entities: dict[str, list[str]] = {}
        sentence_to_entities: dict[str, list[str]] = defaultdict(list)
        passage_hash_id_to_sentences: dict[str, list[str]] = {}
        for (passage_hash_id, _), doc in zip(passage_items, docs):
            (
                single_passage_entities,
                single_sentence_entities,
                single_passage_sentences,
            ) = self.extract_entities_sentences(doc, passage_hash_id)
            passage_hash_id_to_entities.update(single_passage_entities)
            passage_hash_id_to_sentences.update(single_passage_sentences)
            for sentence, entities in single_sentence_entities.items():
                for entity in entities:
                    if entity not in sentence_to_entities[sentence]:
                        sentence_to_entities[sentence].append(entity)
        return passage_hash_id_to_entities, dict(sentence_to_entities), passage_hash_id_to_sentences

    def extract_entities_sentences(
        self,
        doc,
        passage_hash_id: str,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
        sentence_to_entities: dict[str, list[str]] = defaultdict(list)
        unique_entities = set()
        ordered_sentences = [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]
        if not ordered_sentences:
            whole_text = doc.text.strip()
            ordered_sentences = [whole_text] if whole_text else []
        for entity in doc.ents:
            if entity.label_ in {"ORDINAL", "CARDINAL"}:
                continue
            sentence_text = entity.sent.text.strip()
            entity_text = entity.text
            if entity_text not in sentence_to_entities[sentence_text]:
                sentence_to_entities[sentence_text].append(entity_text)
            if sentence_text not in ordered_sentences:
                ordered_sentences.append(sentence_text)
            unique_entities.add(entity_text)

        return (
            {passage_hash_id: sorted(unique_entities)},
            dict(sentence_to_entities),
            {passage_hash_id: ordered_sentences},
        )

    def question_ner(self, question: str) -> set[str]:
        doc = self.spacy_model(question)
        if not self.has_trained_ner:
            return set()
        question_entities = set()
        for entity in doc.ents:
            if entity.label_ in {"ORDINAL", "CARDINAL"}:
                continue
            question_entities.add(entity.text.lower())
        return question_entities
