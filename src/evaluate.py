from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.utils import normalize_answer

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, llm_model, predictions_path: str):
        self.llm_model = llm_model
        self.predictions_path = predictions_path
        self.prediction_results = self.load_predictions()

    def load_predictions(self):
        with open(self.predictions_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def calculate_llm_accuracy(self, pred_answer: str, gold_answer: str) -> float:
        system_prompt = "You are an expert evaluator."
        user_prompt = f"""Please evaluate if the generated answer is correct by comparing it with the gold answer.
Generated answer: {pred_answer}
Gold answer: {gold_answer}

The generated answer should be considered correct if it:
1. Contains the key information from the gold answer
2. Is factually accurate and consistent with the gold answer
3. Does not contain any contradicting information

Respond with ONLY 'correct' or 'incorrect'.
Response:
"""
        response = self.llm_model.infer(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return 1.0 if response.strip().lower() == "correct" else 0.0

    def calculate_contain(self, pred_answer: str, gold_answer: str) -> int:
        if pred_answer is None or str(pred_answer).strip() == "":
            return 0
        if gold_answer is None or str(gold_answer).strip() == "":
            return 0

        normalized_prediction = normalize_answer(pred_answer)
        normalized_gold = normalize_answer(gold_answer)
        return 1 if normalized_gold in normalized_prediction else 0

    def evaluate_sig_sample(self, idx: int, prediction: dict) -> tuple[int, float, int]:
        pred_answer = prediction["pred_answer"]
        gold_answer = prediction["gold_answer"]
        llm_acc = self.calculate_llm_accuracy(pred_answer, gold_answer)
        contain_acc = self.calculate_contain(pred_answer, gold_answer)
        return idx, llm_acc, contain_acc

    def evaluate(self, max_workers: int) -> tuple[float, float]:
        llm_scores = [0.0] * len(self.prediction_results)
        contain_scores = [0.0] * len(self.prediction_results)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.evaluate_sig_sample, idx, pred): idx
                for idx, pred in enumerate(self.prediction_results)
            }

            completed = 0
            total_llm_score = 0.0
            total_contain_score = 0.0
            progress_bar = tqdm(total=len(futures), desc="Evaluating samples", unit="sample")
            for future in as_completed(futures):
                idx, llm_acc, contain_acc = future.result()
                llm_scores[idx] = llm_acc
                contain_scores[idx] = contain_acc
                self.prediction_results[idx]["llm_accuracy"] = llm_acc
                self.prediction_results[idx]["contain_accuracy"] = contain_acc
                total_llm_score += llm_acc
                total_contain_score += contain_acc
                completed += 1
                progress_bar.set_postfix(
                    {
                        "LLM_Acc": f"{total_llm_score / completed:.3f}",
                        "Contain_Acc": f"{total_contain_score / completed:.3f}",
                    }
                )
                progress_bar.update(1)
            progress_bar.close()

        llm_accuracy = sum(llm_scores) / len(llm_scores)
        contain_accuracy = sum(contain_scores) / len(contain_scores)

        logger.info("Evaluation Results:")
        logger.info("  LLM Accuracy: %.4f (%s/%s)", llm_accuracy, sum(llm_scores), len(llm_scores))
        logger.info(
            "  Contain Accuracy: %.4f (%s/%s)",
            contain_accuracy,
            sum(contain_scores),
            len(contain_scores),
        )

        with open(self.predictions_path, "w", encoding="utf-8") as file:
            json.dump(self.prediction_results, file, ensure_ascii=False, indent=2)

        with open(os.path.join(os.path.dirname(self.predictions_path), "evaluation_results.json"), "w", encoding="utf-8") as file:
            json.dump(
                {"llm_accuracy": llm_accuracy, "contain_accuracy": contain_accuracy},
                file,
                ensure_ascii=False,
                indent=2,
            )
        return llm_accuracy, contain_accuracy
