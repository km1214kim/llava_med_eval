"""
results/visualize.py — 플롯 생성 및 JSON 내보내기.

생성 결과물:
  1. 선 그래프: 단계별 정확도 (오차 막대 포함)
  2. 막대 그래프: 주입 유형별 sycophancy rate
  3. 히트맵: Step × Sycophancy type → 평균 sycophancy 점수
  4. 산점도: 케이스별 추론 정확도 vs sycophancy 점수
  5. 전체 결과를 results.json으로 내보내기
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Matplotlib / seaborn 가드
# ---------------------------------------------------------------------------

def _import_plot_libs():
    try:
        import matplotlib
        matplotlib.use("Agg")  # 비대화형 백엔드
        import matplotlib.pyplot as plt
        import seaborn as sns
        return plt, sns
    except ImportError as exc:
        logger.error("Plotting libraries unavailable: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# 1. 선 그래프 — 단계별 정확도
# ---------------------------------------------------------------------------


def _plot_step_accuracy(
    accuracy_by_step: Dict[int, float],
    multistep_results,
    output_dir: Path,
    plt,
    sns,
) -> None:
    from collections import defaultdict
    import numpy as np

    steps = sorted(accuracy_by_step)
    means = [accuracy_by_step[s] for s in steps]

    # 단계별 표준편차 계산 (오차 막대용)
    step_correct: Dict[int, List[float]] = defaultdict(list)
    for r in multistep_results:
        step_correct[r.step].append(float(r.is_correct))
    stds = [
        float(np.std(step_correct[s])) if step_correct[s] else 0.0
        for s in steps
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.set_style("whitegrid")
    ax.errorbar(steps, means, yerr=stds, marker="o", linewidth=2, capsize=4,
                color="#2196F3", ecolor="#90CAF9", label="Accuracy ± std")
    ax.set_xlabel("Reasoning Steps", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Step Accuracy (1 → 3)", fontsize=14)
    ax.set_xticks(steps)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "step_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# 2. 막대 그래프 — 주입 유형별 sycophancy rate
# ---------------------------------------------------------------------------


def _plot_sycophancy_rate(
    sycophancy_rates: Dict[str, float],
    output_dir: Path,
    plt,
    sns,
) -> None:
    types = [k for k in sycophancy_rates if k != "overall"]
    rates = [sycophancy_rates[k] for k in types]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.set_style("whitegrid")
    palette = sns.color_palette("Set2", len(types))
    bars = ax.bar(types, rates, color=palette, edgecolor="white", linewidth=1.2)

    # 전체 평균선 추가
    overall = sycophancy_rates.get("overall", 0)
    ax.axhline(overall, linestyle="--", color="#E53935", linewidth=1.5,
               label=f"Overall ({overall:.2f})")

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rate:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Sycophancy Type", fontsize=12)
    ax.set_ylabel("Sycophancy Rate", fontsize=12)
    ax.set_title("Sycophancy Rate by Injection Type", fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "sycophancy_rate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# 3. 히트맵 — step × sycophancy type
# ---------------------------------------------------------------------------


def _plot_heatmap(
    sycophancy_results,
    sycophancy_types: List[str],
    output_dir: Path,
    plt,
    sns,
) -> None:
    import numpy as np
    import pandas as pd
    from collections import defaultdict

    steps = sorted({r.step for r in sycophancy_results})
    buckets: Dict[Tuple, List[float]] = defaultdict(list)
    for r in sycophancy_results:
        if r.sycophancy_score >= 0:
            buckets[(r.step, r.sycophancy_type)].append(float(r.sycophancy_score))

    data = {
        t: [
            sum(buckets[(s, t)]) / len(buckets[(s, t)]) if buckets[(s, t)] else 0.0
            for s in steps
        ]
        for t in sycophancy_types
    }
    df = pd.DataFrame(data, index=steps)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=4, linewidths=0.5, ax=ax)
    ax.set_title("Mean Sycophancy Score: Step × Injection Type", fontsize=14)
    ax.set_xlabel("Injection Type", fontsize=12)
    ax.set_ylabel("Reasoning Step", fontsize=12)
    fig.tight_layout()
    path = output_dir / "sycophancy_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# 4. 산점도 — 케이스별 정확도 vs sycophancy
# ---------------------------------------------------------------------------


def _plot_scatter(
    cross_analysis,
    output_dir: Path,
    plt,
    sns,
) -> None:
    import numpy as np

    cases = sorted(
        set(cross_analysis.per_case_accuracy) & set(cross_analysis.per_case_sycophancy)
    )
    if not cases:
        logger.warning("No overlapping cases for scatter plot.")
        return

    x = [cross_analysis.per_case_accuracy[c] for c in cases]
    y = [cross_analysis.per_case_sycophancy[c] for c in cases]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, alpha=0.5, color="#7E57C2", edgecolors="white", s=50)

    # 회귀선
    if len(x) >= 2 and len(set(x)) > 1:
        try:
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(min(x), max(x), 100)
            ax.plot(x_line, m * x_line + b, "--", color="#E53935", linewidth=1.5,
                    label=f"y = {m:.2f}x + {b:.2f}")
        except np.linalg.LinAlgError:
            logger.warning("Regression line could not be computed (degenerate data).")

    r = cross_analysis.pearson_r
    p = cross_analysis.pearson_p
    ax.set_title(
        f"Per-case Accuracy vs Sycophancy\n(r = {r:.3f}, p = {p:.3f})", fontsize=13
    )
    ax.set_xlabel("Mean Reasoning Accuracy", fontsize=12)
    ax.set_ylabel("Mean Sycophancy Score (0–4)", fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 4.1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = output_dir / "accuracy_vs_sycophancy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# JSON 내보내기
# ---------------------------------------------------------------------------


def _export_json(
    multistep_results,
    sycophancy_results,
    step_acc: Dict,
    sycophancy_rates: Dict,
    severity: float,
    cross_analysis,
    output_dir: Path,
) -> None:
    payload = {
        "step_accuracy": {str(k): v for k, v in step_acc.items()},
        "sycophancy_rates": sycophancy_rates,
        "severity_score": severity,
        "pearson_r": cross_analysis.pearson_r,
        "pearson_p": cross_analysis.pearson_p,
        "hypothesis_supported": cross_analysis.hypothesis_supported,
        "multistep_results": [r.to_dict() for r in multistep_results],
        "sycophancy_results": [r.to_dict() for r in sycophancy_results],
    }
    path = output_dir / "results.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Results exported to %s", path)


# ---------------------------------------------------------------------------
# 공개 진입점
# ---------------------------------------------------------------------------


def visualize_all(
    step_acc: Dict[int, float],
    sycophancy_rates: Dict[str, float],
    cross_analysis,
    config,
    multistep_results=None,
    sycophancy_results=None,
    severity: float = 0.0,
) -> None:
    """모든 플롯을 생성하고 JSON으로 내보낸다."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt, sns = _import_plot_libs()
    if plt is None:
        logger.error("Skipping visualization (matplotlib unavailable).")
        return

    if multistep_results:
        _plot_step_accuracy(step_acc, multistep_results, output_dir, plt, sns)

    _plot_sycophancy_rate(sycophancy_rates, output_dir, plt, sns)

    if sycophancy_results:
        _plot_heatmap(
            sycophancy_results, config.sycophancy_types, output_dir, plt, sns
        )

    _plot_scatter(cross_analysis, output_dir, plt, sns)

    _export_json(
        multistep_results or [],
        sycophancy_results or [],
        step_acc,
        sycophancy_rates,
        severity,
        cross_analysis,
        output_dir,
    )
