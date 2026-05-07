"""
main.py — Entry point for the LLaVA-MED 1.5 evaluation pipeline.

Runs:
  1. Multi-step reasoning experiment
  2. Sycophancy injection experiment
  3. GPT-4o judge scoring
  4. Metric computation
  5. Cross-analysis
  6. Visualization & export
"""

from __future__ import annotations

# huggingface_hub import 전에 HF 캐시 경로를 재지정한다.
# 공용 서버의 HF_HOME=/models/hf-cache 는 일반 사용자에게 읽기 전용이므로 홈 디렉토리로 대체.
import os as _os
_user_hf = _os.path.expanduser("~/.cache/huggingface")
_user_hub = _os.path.join(_user_hf, "hub")
_os.makedirs(_user_hub, exist_ok=True)
for _k, _v in [
    ("HF_HOME",               _user_hf),
    ("HF_HUB_CACHE",          _user_hub),
    ("HUGGINGFACE_HUB_CACHE", _user_hub),
    ("HF_ASSETS_CACHE",       _os.path.join(_user_hf, "assets")),
]:
    _os.environ[_k] = _v

import argparse
import logging
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 로깅 설정 (다른 import보다 먼저 실행해야 함)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# 임포트
# ---------------------------------------------------------------------------

from config import load_config
from data.loader import load_dataset
from model.llava_med import LLaVAMedModel
from experiments.multistep import run_multistep_experiment
from experiments.sycophancy import run_sycophancy_experiment
from evaluation.judge import GPT4oJudge
from evaluation.metrics import (
    compute_step_accuracy,
    compute_step_dropoff,
    compute_faithfulness,
    compute_sycophancy_rate,
    compute_severity_score,
)
from analysis.cross_analysis import run_cross_analysis
from results.visualize import visualize_all


# ---------------------------------------------------------------------------
# 시드 설정 헬퍼
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 결과 요약 출력
# ---------------------------------------------------------------------------

def print_summary(
    step_acc: dict,
    dropoff: dict,
    faithfulness: dict,
    sy_rate: dict,
    severity: float,
    cross,
) -> None:
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)

    print("\n[Multi-step Accuracy]")
    for step, acc in sorted(step_acc.items()):
        bar = "█" * int(acc * 20)
        print(f"  Step {step}: {acc:.3f}  {bar}")

    print("\n[Step Drop-off]")
    for (s1, s2), drop in sorted(dropoff.items()):
        sign = "▼" if drop > 0 else "▲"
        print(f"  {s1}→{s2}: {sign} {abs(drop):.3f}")

    print("\n[Faithfulness by Step]")
    for step, score in sorted(faithfulness.items()):
        print(f"  Step {step}: {score:.3f}")

    print("\n[Sycophancy Rates]")
    for t, rate in sorted(sy_rate.items()):
        print(f"  {t:<20}: {rate:.3f}")

    print(f"\n[Severity Score]: {severity:.4f}")

    print("\n[Cross-Analysis]")
    cross.print_summary()

    print("=" * 60)


# ---------------------------------------------------------------------------
# 인자 파싱
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLaVA-MED 1.5 multi-step reasoning & sycophancy evaluation"
    )
    parser.add_argument("--model-path", default=None, help="Override model path")
    parser.add_argument(
        "--dataset",
        choices=["iu_xray", "slake"],
        default=None,
        help="Dataset to use: iu_xray or slake",
    )
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Force mock mode (no real model or OpenAI calls)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # 로그 레벨 적용
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # 설정 빌드
    overrides = {}
    if args.model_path:
        overrides["model_path"] = args.model_path
    if args.dataset:
        overrides["dataset"] = args.dataset
    if args.n_samples:
        overrides["n_samples"] = args.n_samples
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.mock:
        overrides["use_mock_dataset"] = True

    config = load_config(**overrides)
    _set_seeds(config.seed)

    # 출력 및 캐시 디렉토리 생성
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Config: %s", config)

    # ----------------------------------------------------------------
    # 데이터셋 로드
    # ----------------------------------------------------------------
    logger.info("Loading dataset …")
    dataset = load_dataset(config)
    logger.info("Dataset size: %d cases", len(dataset))

    # ----------------------------------------------------------------
    # 모델 로드
    # ----------------------------------------------------------------
    logger.info("Initialising model …")
    model = LLaVAMedModel(config)

    # ----------------------------------------------------------------
    # 판정 모델 초기화
    # ----------------------------------------------------------------
    logger.info("Initialising judge …")
    judge = GPT4oJudge(config)

    # ----------------------------------------------------------------
    # 실험 실행
    # ----------------------------------------------------------------
    logger.info("Running multi-step experiment …")
    ms_results = run_multistep_experiment(model, dataset, config)

    logger.info("Running sycophancy experiment …")
    sy_results = run_sycophancy_experiment(model, dataset, config, judge)

    # ----------------------------------------------------------------
    # Judge 채점
    # ----------------------------------------------------------------
    logger.info("Scoring multi-step results with judge …")
    for r in ms_results:
        r.is_correct, r.judge_reason = judge.score_correctness(
            r.response, r.ground_truth, r.image_path
        )

    logger.info("Scoring sycophancy results with judge …")
    for r in sy_results:
        r.sycophancy_score, r.sycophancy_reason = judge.score_sycophancy(
            r.baseline_response, r.post_injection_response, r.wrong_label
        )

    # ----------------------------------------------------------------
    # 지표 계산
    # ----------------------------------------------------------------
    logger.info("Computing metrics …")
    step_acc = compute_step_accuracy(ms_results)
    dropoff = compute_step_dropoff(step_acc)
    faithfulness = compute_faithfulness(ms_results)
    sy_rate = compute_sycophancy_rate(sy_results)
    severity = compute_severity_score(sy_results)

    # ----------------------------------------------------------------
    # 교차 분석
    # ----------------------------------------------------------------
    logger.info("Running cross-analysis …")
    cross = run_cross_analysis(ms_results, sy_results)

    # ----------------------------------------------------------------
    # 시각화 및 결과 저장
    # ----------------------------------------------------------------
    logger.info("Generating visualizations …")
    visualize_all(
        step_acc=step_acc,
        sycophancy_rates=sy_rate,
        cross_analysis=cross,
        config=config,
        multistep_results=ms_results,
        sycophancy_results=sy_results,
        severity=severity,
    )

    # ----------------------------------------------------------------
    # 요약 출력
    # ----------------------------------------------------------------
    print_summary(step_acc, dropoff, faithfulness, sy_rate, severity, cross)

    logger.info(
        "Pipeline complete. Results in: %s", Path(config.output_dir).resolve()
    )


if __name__ == "__main__":
    main()
