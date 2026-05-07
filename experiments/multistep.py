"""
experiments/multistep.py — 다단계 추론 실험.

케이스별 3가지 독립 실험 변형:
  Step 1 (Zero-shot Baseline):    Q → 진단
  Step 2 (Observation-Guided):    소견 Q → 진단 Q
  Step 3 (Chain-of-Thought):      소견 Q → 감별진단 Q → 진단 Q

Faithfulness는 중간 응답과 최종 진단 응답의
평균 Jaccard 유사도로 측정한다.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 결과 데이터클래스
# ---------------------------------------------------------------------------


@dataclass
class MultiStepResult:
    case_id: str
    image_path: str          # 정면 이미지 (judge용)
    ground_truth: str
    step: int
    question: str
    response: str
    is_correct: bool = False          # GPT-4o judge가 나중에 채점
    judge_reason: str = ""            # judge가 판단한 근거
    faithfulness_score: float = 0.0   # 0.0–1.0; step=1이면 trivially 1.0

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "image_path": self.image_path,
            "ground_truth": self.ground_truth,
            "step": self.step,
            "question": self.question,
            "response": self.response,
            "is_correct": self.is_correct,
            "judge_reason": self.judge_reason,
            "faithfulness_score": self.faithfulness_score,
        }


# ---------------------------------------------------------------------------
# Faithfulness 계산기
# ---------------------------------------------------------------------------


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """두 문자열 간 단어 수준 Jaccard 유사도를 반환한다."""
    def tokens(t: str) -> set:
        return set(re.findall(r"\w+", t.lower()))

    a, b = tokens(text_a), tokens(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _compute_faithfulness(responses: List[str]) -> float:
    """
    중간 응답이 최종 응답에 얼마나 반영됐는지 확인한다.

    Step 1 (응답 1개): trivially faithful → 1.0.
    Step 2+: 각 중간 응답과 최종 응답의 Jaccard 유사도 평균.
    """
    if len(responses) <= 1:
        return 1.0

    final = responses[-1]
    intermediate_scores = [_keyword_overlap(r, final) for r in responses[:-1]]
    return sum(intermediate_scores) / len(intermediate_scores)


# ---------------------------------------------------------------------------
# 주요 실험 함수
# ---------------------------------------------------------------------------


def run_multistep_experiment(model, dataset, config) -> List[MultiStepResult]:
    """
    다단계 추론 실험을 실행한다.

    케이스별 3가지 독립 실험 변형:
      Step 1 (Zero-shot Baseline):
        Q:  "What is the diagnosis for this X-ray?"
        → A: 진단

      Step 2 (Observation-Guided):
        Q1: "Please describe the key findings in this X-ray."  → A1: 소견
        Q2: "What is the diagnosis for this X-ray?"           → A2: 진단

      Step 3 (Chain-of-Thought):
        Q1: "Please describe the key findings in this X-ray."     → A1
        Q2: "Please provide 3~5 possible differential diagnoses." → A2
        Q3: "What is the diagnosis for this X-ray?"               → A3

    각 step은 독립 실험으로 새 대화를 시작한다.
    정확도와 faithfulness는 최종(진단) 응답을 기준으로 평가한다.
    """
    results: List[MultiStepResult] = []
    total = len(dataset) * len(config.reasoning_steps)

    with tqdm(total=total, desc="Multi-step experiment") as pbar:
        for case in dataset:
            for step in config.reasoning_steps:
                questions = case.question_variants.get(step, case.question_variants[1])
                ground_truth = case.ground_truth_by_step.get(step, case.ground_truth)

                logger.info(
                    "MultiStep | case=%s step=%d (%d turns)",
                    case.case_id, step, len(questions),
                )

                # 각 step마다 독립 대화 시작 — step 간 히스토리 공유 없음
                chain_history: List[dict] = []
                responses: List[str] = []

                for q in questions:
                    response = model.generate(
                        image_paths=case.image_paths,
                        prompt=q,
                        history=chain_history if chain_history else None,
                        case_id=f"{case.case_id}_step{step}",
                    )
                    chain_history = chain_history + [
                        {"role": "user",      "content": q},
                        {"role": "assistant", "content": response},
                    ]
                    responses.append(response)

                faithfulness = _compute_faithfulness(responses)

                results.append(
                    MultiStepResult(
                        case_id=case.case_id,
                        image_path=case.image_paths[0],
                        ground_truth=ground_truth,
                        step=step,
                        question=questions[-1],   # 최종 (진단) 질문
                        response=responses[-1],   # 최종 (진단) 응답
                        faithfulness_score=faithfulness,
                    )
                )
                pbar.update(1)

    logger.info("Multi-step experiment complete. %d results.", len(results))
    return results
