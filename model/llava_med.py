"""
model/llava_med.py — LLaVA-MED 1.5 (7B) inference wrapper.

Supports:
- Single-turn generation
- Multi-turn history for sycophancy experiments
- Disk caching keyed by (case_id, step, sycophancy_type)
- Mock mode when model weights are unavailable
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 모델 가중치 미사용 시 Mock 응답
# ---------------------------------------------------------------------------

_MOCK_RESPONSES = [
    "Consolidation is observed in the right lower lobe of this chest X-ray, consistent with pneumonia.",
    "There is homogeneous opacification at the left lung base with blunting of the costophrenic angle, consistent with pleural effusion.",
    "The cardiothoracic ratio is approximately 0.55 with bilateral hilar vascular prominence, consistent with cardiomegaly.",
    "The right lung is collapsed with a visible pleural line, consistent with pneumothorax.",
    "An ill-defined nodular opacity is seen in the right upper lobe. Malignancy cannot be excluded.",
]

_MOCK_SYCOPHANTIC = [
    "On further review, that could be the case.",
    "Taking the specialist's opinion into consideration, re-evaluation may be warranted.",
    "You may have a point. Additional review is needed.",
]


# ---------------------------------------------------------------------------
# 캐시 헬퍼
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
# Model wrapper
# ---------------------------------------------------------------------------


class LLaVAMedModel:
    """
    Wraps LLaVA-MED 1.5 (7B) for inference.

    When the model weights are unavailable (or *mock_mode=True*), falls back
    to deterministic mock responses so the rest of the pipeline can be tested
    end-to-end without GPU resources.
    """

    def __init__(self, config):
        self.config = config
        self._cache = _ResponseCache(getattr(config, "cache_dir", "./cache"))
        self._rng = random.Random(config.seed)
        self._model = None
        self._mock_mode: bool = getattr(config, "use_mock_dataset", True)

        if not self._mock_mode:
            self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        image_paths: List[str],
        prompt: str,
        history: Optional[List[Dict]] = None,
        case_id: str = "unknown",
    ) -> str:
        """
        Generate a response for *image_paths* given *prompt*.

        Args:
            image_paths: Paths to the medical images (frontal + lateral).
            prompt: The question / instruction string.
            history: Optional list of prior turns for multi-turn conversations.
                     Each element: {"role": "user"|"assistant", "content": str}
            case_id: Used as part of the cache key.

        Returns:
            Model response as a string.
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
        logger.debug("Generated response for case_id=%s (len=%d)", case_id, len(response))
        return response

    # ------------------------------------------------------------------
    # Real model inference
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            import torch
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path

            logger.info("Loading LLaVA-MED model from %s …", self.config.model_path)
            model_name = get_model_name_from_path(self.config.model_path)
            self._tokenizer, self._model, self._image_processor, self._context_len = (
                load_pretrained_model(
                    model_path=self.config.model_path,
                    model_base=None,
                    model_name=model_name,
                    device_map="auto",
                )
            )
            self._model.eval()
            logger.info("Model loaded successfully (context_len=%d).", self._context_len)
        except Exception as exc:
            import traceback
            logger.warning(
                "Could not load model. Switching to mock mode.\n%s",
                traceback.format_exc()
            )
            self._mock_mode = True

    def _real_generate(
        self,
        image_paths: List[str],
        prompt: str,
        history: Optional[List[Dict]],
    ) -> str:
        import torch
        from PIL import Image
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )
        from llava.conversation import conv_templates

        images = [Image.open(p).convert("RGB") for p in image_paths]
        image_tensor = process_images(
            images, self._image_processor, self._model.config
        ).to(self._model.device, dtype=torch.float16)

        # 이미지 수만큼 토큰 생성 (frontal + lateral)
        use_im_tags = getattr(self._model.config, "mm_use_im_start_end", False)
        if use_im_tags:
            single_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            single_token = DEFAULT_IMAGE_TOKEN
        img_tokens = "\n".join(single_token for _ in image_paths)

        conv = conv_templates["mistral_instruct"].copy()

        if history:
            for i, turn in enumerate(history):
                role_idx = 0 if turn["role"] == "user" else 1
                content = turn["content"]
                if role_idx == 0 and i == 0:
                    content = img_tokens + "\n" + content
                conv.append_message(conv.roles[role_idx], content)

            # If history ends with a user turn, merge prompt into it to avoid
            # consecutive user turns which break the Mistral chat template.
            if conv.messages and conv.messages[-1][0] == conv.roles[0]:
                conv.messages[-1] = (
                    conv.roles[0],
                    conv.messages[-1][1] + "\n" + prompt,
                )
            else:
                conv.append_message(conv.roles[0], prompt)
        else:
            conv.append_message(conv.roles[0], img_tokens + "\n" + prompt)

        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_text, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=image_tensor,
                do_sample=self.config.temperature > 0,
                temperature=self.config.temperature,
                max_new_tokens=self.config.max_new_tokens,
                min_new_tokens=1,
                use_cache=True,
            )

        # LLaVA processes inputs via inputs_embeds internally, so output_ids
        # contains only the generated tokens (not the echoed input).
        # Decode the full output directly.
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Mock inference
    # ------------------------------------------------------------------

    def _mock_generate(
        self,
        prompt: str,
        history: Optional[List[Dict]],
    ) -> str:
        """Return a plausible-looking mock response."""
        time.sleep(0.01)  # Simulate latency

        # If history contains a false-premise injection, simulate partial sycophancy
        if history:
            injection_keywords = ["incorrect", "specialist", "clearly", "instead"]
            history_text = " ".join(h.get("content", "") for h in history)
            if any(kw in history_text for kw in injection_keywords):
                sycophancy_roll = self._rng.random()
                if sycophancy_roll < 0.3:
                    return self._rng.choice(_MOCK_SYCOPHANTIC)
                elif sycophancy_roll < 0.6:
                    base = self._rng.choice(_MOCK_RESPONSES)
                    hedge = " However, the possibility you raised cannot be entirely excluded."
                    return base + hedge

        # Default: pick a base response varied by prompt length (proxy for step count)
        idx = (len(prompt) // 20) % len(_MOCK_RESPONSES)
        return _MOCK_RESPONSES[idx]
