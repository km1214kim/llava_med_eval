"""
analysis/cross_analysis.py — 추론 복잡도 × Sycophancy 교차 분석.

가설 검증: 추론 단계가 높을수록 sycophancy rate가 높아지는가.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 결과 데이터클래스
# ---------------------------------------------------------------------------


@dataclass
class CrossAnalysis:
    # 추론 단계별 평균 sycophancy 점수
    mean_sycophancy_by_step: Dict[int, float]

    # 케이스별: (전 단계 평균 정확도, 평균 sycophancy 점수)
    per_case_accuracy: Dict[str, float]
    per_case_sycophancy: Dict[str, float]

    # 단계와 평균 sycophancy 간의 Pearson r
    pearson_r: float
    pearson_p: float

    # 가설 지지 여부 (p < 0.05 이고 r > 0)
    hypothesis_supported: bool

    # 요약 테이블 행: [(step, n_cases, mean_syco, mean_accuracy)]
    summary_table: List[Tuple[int, int, float, float]] = field(default_factory=list)

    def print_summary(self) -> None:
        print("\n=== Cross-Analysis: Reasoning Steps × Sycophancy ===")
        print(f"\n{'Step':>4}  {'N':>6}  {'Accuracy':>10}  {'Mean Syco':>10}")
        print("-" * 40)
        for step, n, syco, acc in self.summary_table:
            print(f"{step:>4}  {n:>6}  {acc:>10.3f}  {syco:>10.3f}")
        print(f"\nPearson r = {self.pearson_r:.4f}  (p = {self.pearson_p:.4f})")
        hypothesis_str = "SUPPORTED" if self.hypothesis_supported else "NOT SUPPORTED"
        print(f"Hypothesis (higher steps → higher sycophancy): {hypothesis_str}")


# ---------------------------------------------------------------------------
# Pearson 상관계수 (scipy 있으면 사용, 없으면 수동 계산)
# ---------------------------------------------------------------------------


def _pearson(x: List[float], y: List[float]) -> Tuple[float, float]:
    """(r, p-value)를 반환한다. scipy가 있으면 scipy를 사용하고, 없으면 수동 계산."""
    if len(x) < 3:
        return 0.0, 1.0

    try:
        from scipy import stats
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)
    except ImportError:
        pass

    # 수동 Pearson r (scipy 없을 때)
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    den_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0, 1.0
    r = num / (den_x * den_y)
    # t-분포를 이용한 근사 p-value
    try:
        import math
        t = r * math.sqrt(n - 2) / math.sqrt(max(1e-9, 1 - r ** 2))
        # 양측 p-value 근사 (scipy 미사용 시 rough estimate)
        # P ≈ 2 * (1 - Φ(|t|)) 공식 기반 sigmoid 근사:
        p = 1.0 / (1.0 + math.exp(abs(t) - 2.5))
    except Exception:
        p = 1.0
    return float(r), float(p)


# ---------------------------------------------------------------------------
# 주요 분석 함수
# ---------------------------------------------------------------------------


def run_cross_analysis(multistep_results, sycophancy_results) -> CrossAnalysis:
    """
    추론 단계 복잡도 × sycophancy 점수를 교차 집계한다.

    SycophancyResult의 *step* 필드(1 또는 3)를 사용하고,
    case_id로 multistep_results와 조인해 단계별 정확도를 구한다.
    """
    # ----------------------------------------------------------------
    # 케이스별 정확도 — (case_id, step) 단위로 관리
    # ----------------------------------------------------------------
    case_step_correct: Dict[str, Dict[int, bool]] = defaultdict(dict)
    for r in multistep_results:
        case_step_correct[r.case_id][r.step] = r.is_correct

    # 요약 테이블용: 케이스 전체 평균 (산점도에 사용)
    per_case_accuracy: Dict[str, float] = {}
    for case_id, step_map in case_step_correct.items():
        if step_map:
            per_case_accuracy[case_id] = sum(step_map.values()) / len(step_map)

    # step별 정확도 집계용: (case_id, step) → bool
    case_step_accuracy: Dict[tuple, float] = {
        (case_id, step): float(correct)
        for case_id, step_map in case_step_correct.items()
        for step, correct in step_map.items()
    }

    # ----------------------------------------------------------------
    # 케이스별 평균 sycophancy 점수
    # ----------------------------------------------------------------
    case_syco_scores: Dict[str, List[int]] = defaultdict(list)
    for r in sycophancy_results:
        if r.sycophancy_score >= 0:
            case_syco_scores[r.case_id].append(r.sycophancy_score)

    per_case_sycophancy: Dict[str, float] = {
        case_id: sum(scores) / len(scores)
        for case_id, scores in case_syco_scores.items()
    }

    # ----------------------------------------------------------------
    # 단계별 평균 sycophancy 점수 집계
    # ----------------------------------------------------------------
    step_syco_buckets: Dict[int, List[float]] = defaultdict(list)
    step_acc_buckets: Dict[int, List[float]] = defaultdict(list)
    step_n: Dict[int, int] = defaultdict(int)

    for r in sycophancy_results:
        if r.sycophancy_score >= 0:
            step_syco_buckets[r.step].append(r.sycophancy_score)
            acc = case_step_accuracy.get((r.case_id, r.step), per_case_accuracy.get(r.case_id, 0.5))
            step_acc_buckets[r.step].append(acc)
            step_n[r.step] += 1

    mean_sycophancy_by_step: Dict[int, float] = {
        step: sum(scores) / len(scores)
        for step, scores in sorted(step_syco_buckets.items())
    }

    # ----------------------------------------------------------------
    # Pearson 상관계수: 단계 → 평균 sycophancy
    # (step_value, syco_score) 쌍 리스트로 전개
    # ----------------------------------------------------------------
    x_steps: List[float] = []
    y_syco: List[float] = []
    for r in sycophancy_results:
        if r.sycophancy_score >= 0:
            x_steps.append(float(r.step))
            y_syco.append(float(r.sycophancy_score))

    pearson_r, pearson_p = _pearson(x_steps, y_syco)

    hypothesis_supported = (pearson_r > 0) and (pearson_p < 0.05)

    # ----------------------------------------------------------------
    # 요약 테이블 구성
    # ----------------------------------------------------------------
    summary_table = [
        (
            step,
            step_n[step],
            mean_sycophancy_by_step.get(step, 0.0),
            (sum(step_acc_buckets[step]) / len(step_acc_buckets[step])
             if step_acc_buckets[step] else 0.0),
        )
        for step in sorted(step_syco_buckets)
    ]

    logger.info(
        "Cross-analysis | r=%.4f p=%.4f hypothesis=%s",
        pearson_r, pearson_p, hypothesis_supported,
    )

    return CrossAnalysis(
        mean_sycophancy_by_step=mean_sycophancy_by_step,
        per_case_accuracy=per_case_accuracy,
        per_case_sycophancy=per_case_sycophancy,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        hypothesis_supported=hypothesis_supported,
        summary_table=summary_table,
    )
