"""
config.py — LLaVA-MED 평가 파이프라인의 모든 하이퍼파라미터 및 설정.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # 모델
    model_path: str = "JZPeterPan/MedVLM-R1" # 변경부

    # 데이터셋 ("iu_xray" | "slake")
    dataset: str = "iu_xray"

    # 샘플링
    n_samples: int = 150
    seed: int = 42

    # 평가할 추론 단계
    reasoning_steps: List[int] = field(default_factory=lambda: [1, 2, 3])

    # Sycophancy 주입 유형
    sycophancy_types: List[str] = field(
        default_factory=lambda: [
            "authority",
            "negation",
            "false_certainty",
            "repetition",
        ]
    )

    # 판정 모델 — OpenRouter 모델 ID 형식: "provider/model-name"
    judge_model: str = "google/gemini-2.0-flash-001"

    # 결과 저장 디렉토리
    output_dir: str = "./results"

    # 추론 설정
    temperature: float = 0.2
    max_new_tokens: int = 512

    # 모델 응답 캐시 디렉토리
    cache_dir: str = "./cache"

    # OpenRouter API 키 — 미설정 시 OPENROUTER_API_KEY 환경변수에서 읽음
    openrouter_api_key: str = ""

    # 실제 데이터 미사용 시 mock dataset으로 대체
    use_mock_dataset: bool = False


def load_config(**overrides) -> Config:
    """Config를 생성하고 키워드 오버라이드를 적용해 반환한다."""
    cfg = Config()
    for key, val in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config key: {key!r}")
        setattr(cfg, key, val)
    return cfg
