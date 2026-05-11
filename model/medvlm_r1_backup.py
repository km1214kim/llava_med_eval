"""
model/medvlm_r1.py — MedVLM-R1 (Qwen2-VL-2B 기반) inference wrapper.

기존 LLaVA-MED wrapper와 동일한 public API를 유지하므로
experiments/, evaluation/ 코드는 수정 없이 그대로 사용 가능.

Supports:
- Single-turn generation
- Multi-turn history for sycophancy experiments
- Disk caching keyed by (case_id, prompt_hash)
- Mock mode when model weights are unavailable

MedVLM-R1 특징:
- Qwen2-VL-2B 기반, GRPO 강화학습으로 파인튜닝
- 출력 형식: <think>...</think><answer>...</answer>
- parse_output()으로 reasoning과 answer를 분리해 반환
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock 응답 (GPU 없이 파이프라인 테스트용)
# ---------------------------------------------------------------------------

_MOCK_RESPONSES = [
    "<think>The image shows consolidation in the right lower lobe with air bronchograms present.</think><answer>Pneumonia</answer>",
    "<think>There is blunting of the left costophrenic angle with homogeneous opacity at the base.</think><answer>Pleural effusion</answer>",
    "<think>The cardiothoracic ratio exceeds 0.50 with bilateral hilar prominence.</think><answer>Cardiomegaly</answer>",
    "<think>A visible pleural line is seen with absent lung markings peripheral to it.</think><answer>Pneumothorax</answer>",
    "<think>An ill-defined nodular opacity in the right upper lobe raises concern for malignancy.</think><answer>Pulmonary nodule</answer>",
]

_MOCK_SYCOPHANTIC = [
    "<think>Reconsidering the specialist's input, the findings could be reinterpreted.</think><answer>I may have been mistaken earlier.</answer>",
    "<think>Taking the provided opinion into account, re-evaluation seems warranted.</think><answer>You may be correct.</answer>",
    "<think>On further reflection, both interpretations are plausible.</think><answer>Additional review is needed.</answer>",
]

# MedVLM-R1 프롬프트 템플릿 (HuggingFace 공식)
_QUESTION_TEMPLATE = (
    "{question}\n"
    "Your task:\n"
    "1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags.\n"
    "2. Then provide your answer inside <answer>...</answer> tags.\n"
    "3. No extra information or text outside of these tags."
)


# ---------------------------------------------------------------------------
# 캐시 헬퍼 (기존과 동일)
# ---------------------------------------------------------------------------


class _ResponseCache:
    def __init__(self, cache_dir: str):
        self._path = Path(cache_dir) / "model_responses.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._store: Dict[str, str] = {}
        if self._path.exists():
            try:
                self._store = json.loads(self._path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Cache file corrupted; starting fresh.")

    def _key(self, case_id: str, prompt: str) -> str:
        digest = hashlib.md5(prompt.encode()).hexdigest()[:8]
        return f"{case_id}_{digest}"

    def get(self, case_id: str, prompt: str) -> Optional[str]:
        return self._store.get(self._key(case_id, prompt))

    def set(self, case_id: str, prompt: str, response: str) -> None:
        self._store[self._key(case_id, prompt)] = response
        self._path.write_text(
            json.dumps(self._store, ensure_ascii=False, indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# 출력 파싱 유틸
# ---------------------------------------------------------------------------


def parse_output(raw: str) -> Tuple[str, str]:
    """
    MedVLM-R1 출력에서 reasoning과 answer를 분리한다.

    Returns:
        (thinking, answer) 튜플.
        태그가 없을 경우 thinking은 빈 문자열, answer는 전체 raw를 반환.
    """
    thinking = ""
    answer = raw.strip()

    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()

    answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()

    return thinking, answer


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class LLaVAMedModel:
    """
    MedVLM-R1 (Qwen2-VL-2B) inference wrapper.

    기존 LLaVAMedModel과 동일한 public API를 제공하므로
    main.py의 import 경로만 바꾸면 나머지 코드는 수정 불필요.

    generate()가 반환하는 문자열은 <think>...</think><answer>...</answer> 형식이며,
    judge 및 metrics 모듈에서는 parse_output()으로 answer 부분만 추출해 사용.
    """

    def __init__(self, config):
        self.config = config
        self._cache = _ResponseCache(getattr(config, "cache_dir", "./cache"))
        self._rng = random.Random(config.seed)
        self._model = None
        self._processor = None
        self._mock_mode: bool = getattr(config, "use_mock_dataset", True)

        if not self._mock_mode:
            self._load_model()

    # ------------------------------------------------------------------
    # Public API — 기존과 동일한 시그니처 유지
    # ------------------------------------------------------------------

    def generate(
        self,
        image_paths: List[str],
        prompt: str,
        history: Optional[List[Dict]] = None,
        case_id: str = "unknown",
    ) -> str:
        """
        MedVLM-R1로 응답을 생성한다.

        Args:
            image_paths: 의료 이미지 경로 리스트.
            prompt: 질문 문자열.
            history: 멀티턴 대화 히스토리.
                     각 원소: {"role": "user"|"assistant", "content": str}
            case_id: 캐시 키 생성에 사용.

        Returns:
            <think>...</think><answer>...</answer> 형식의 문자열.
        """
        cache_key_prompt = (
            json.dumps(history or [], ensure_ascii=False) + "\n" + prompt
        )
        cached = self._cache.get(case_id, cache_key_prompt)
        if cached is not None:
            logger.debug("Cache hit for case_id=%s", case_id)
            return cached

        if self._mock_mode:
            response = self._mock_generate(prompt, history)
        else:
            response = self._real_generate(image_paths, prompt, history)

        self._cache.set(case_id, cache_key_prompt, response)
        logger.debug("Generated for case_id=%s (len=%d)", case_id, len(response))
        return response

    # ------------------------------------------------------------------
    # 실제 모델 추론
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig

            logger.info("Loading MedVLM-R1 from %s …", self.config.model_path)

            cfg = AutoConfig.from_pretrained(self.config.model_path)
            if hasattr(cfg, 'use_cache') and cfg.use_cache is None:
                cfg.use_cache = True
            if hasattr(cfg, 'text_config') and hasattr(cfg.text_config, 'use_cache') and cfg.text_config.use_cache is None:
                cfg.text_config.use_cache = True

            self._model = Qwen2VLForConditionalGeneration.from_pretrained(

    def _build_messages(
        self,
        image_paths: List[str],
        prompt: str,
        history: Optional[List[Dict]],
    ) -> List[Dict]:
        """
        Qwen2-VL chat template에 맞는 messages 리스트를 구성한다.

        - 첫 번째 user 메시지에 이미지를 포함시킨다.
        - history가 있으면 이전 대화를 그대로 붙이고 새 prompt를 마지막에 추가한다.
        - MedVLM-R1 프롬프트 템플릿을 적용한다.
        """
        messages: List[Dict] = []
        formatted_prompt = _QUESTION_TEMPLATE.format(question=prompt)

        if history:
            # history의 첫 user 턴에 이미지 삽입
            for i, turn in enumerate(history):
                if turn["role"] == "user" and i == 0:
                    content = [
                        *[{"type": "image", "image": f"file://{p}"} for p in image_paths],
                        {"type": "text", "text": turn["content"]},
                    ]
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({
                        "role": turn["role"],
                        "content": [{"type": "text", "text": turn["content"]}],
                    })
            # 새 prompt 추가
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": formatted_prompt}],
            })
        else:
            # 단일 턴: 이미지 + 질문
            content = [
                *[{"type": "image", "image": f"file://{p}"} for p in image_paths],
                {"type": "text", "text": formatted_prompt},
            ]
            messages.append({"role": "user", "content": content})

        return messages

    def _real_generate(
        self,
        image_paths: List[str],
        prompt: str,
        history: Optional[List[Dict]],
    ) -> str:
        import torch
        from transformers import GenerationConfig
        from qwen_vl_utils import process_vision_info

        messages = self._build_messages(image_paths, prompt, history)

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        gen_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.temperature > 0,
            temperature=self.config.temperature if self.config.temperature > 0 else 1.0,
            num_return_sequences=1,
            pad_token_id=151643,  # Qwen2 pad token
        )

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                use_cache=True,
                generation_config=gen_config,
            )

        # 입력 토큰 제거 후 디코딩
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()

    # ------------------------------------------------------------------
    # Mock 추론
    # ------------------------------------------------------------------

    def _mock_generate(
        self,
        prompt: str,
        history: Optional[List[Dict]],
    ) -> str:
        time.sleep(0.01)

        if history:
            injection_keywords = ["incorrect", "specialist", "clearly", "instead"]
            history_text = " ".join(h.get("content", "") for h in history)
            if any(kw in history_text for kw in injection_keywords):
                roll = self._rng.random()
                if roll < 0.3:
                    return self._rng.choice(_MOCK_SYCOPHANTIC)
                elif roll < 0.6:
                    base = self._rng.choice(_MOCK_RESPONSES)
                    _, ans = parse_output(base)
                    hedge = " However, the possibility raised cannot be entirely excluded."
                    return f"<think>Re-evaluating given the new information.{hedge}</think><answer>{ans}</answer>"

        idx = (len(prompt) // 20) % len(_MOCK_RESPONSES)
        return _MOCK_RESPONSES[idx]