from __future__ import annotations

import logging
import os
import re
import string
from hashlib import md5

import numpy as np


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode("utf-8")).hexdigest()


def normalize_answer(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(value.lower())))


def setup_logging(log_file: str) -> None:
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)


def min_max_normalize(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    min_val = np.min(array)
    max_val = np.max(array)
    value_range = max_val - min_val
    if value_range == 0:
        return np.ones_like(array)
    return (array - min_val) / value_range
