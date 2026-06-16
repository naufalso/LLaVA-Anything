from __future__ import annotations

import importlib.util
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import llava_anything  # noqa: F401 - registers LLaVA-Anything Auto classes.
from llava_anything.inference import DEFAULT_SYSTEM_PROMPT, generate_responses


def manifest() -> Any:
    """Expose LLaVA-Anything to lmms-eval's model entry-point registry."""

    from lmms_eval.models.registry_v2 import ModelManifest

    return ModelManifest(
        model_id="llava_anything",
        simple_class_path="llava_anything.eval.lmms.LlavaAnything",
    )


def _torch_dtype(value: str | torch.dtype | None) -> torch.dtype | str | None:
    if value is None or isinstance(value, torch.dtype):
        return value
    if value == "auto":
        return "auto"
    if value in {"none", "None"}:
        return None
    dtype = getattr(torch, value, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise ValueError(f"Unknown torch dtype: {value}")


def _lmms_base() -> type:
    try:
        from lmms_eval.api.model import lmms
    except ImportError as exc:  # pragma: no cover - exercised only without the eval extra.
        raise ImportError("Install LLaVA-Anything with the eval extra: pip install -e '.[eval]'") from exc
    return lmms


class LlavaAnything(_lmms_base()):
    """LLaVA-Anything adapter for lmms-eval generation tasks."""

    def __init__(
        self,
        pretrained: str,
        device: str = "cuda",
        dtype: str | torch.dtype | None = "auto",
        batch_size: int = 1,
        device_map: str = "auto",
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 128,
        attn_implementation: str | None = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected model_args for llava_anything: {kwargs}")

        self.pretrained = pretrained
        self.batch_size_per_gpu = int(batch_size)
        self.system_prompt = self._resolve_system_prompt(system_prompt or "")
        self.default_max_new_tokens = int(max_new_tokens)
        self._device = torch.device(device)

        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        parsed_dtype = _torch_dtype(dtype)
        if parsed_dtype is not None:
            model_kwargs["torch_dtype"] = parsed_dtype
        if device_map:
            model_kwargs["device_map"] = device_map
        elif device:
            model_kwargs["device_map"] = device
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        elif importlib.util.find_spec("flash_attn") is not None:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self._processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        _prepare_processor_for_batched_generation(self._processor)
        self._model = AutoModelForImageTextToText.from_pretrained(pretrained, **model_kwargs)
        self._model.eval()
        if not device_map and hasattr(self._model, "to"):
            self._model.to(self._device)
        self._logged_batching = False

        tokenizer = getattr(self._processor, "tokenizer", None)
        padding_side = getattr(tokenizer, "padding_side", "<unknown>")
        print(
            "[INFO] LLaVA-Anything lmms adapter: "
            f"batch_size={self.batch_size_per_gpu}, tokenizer.padding_side={padding_side}"
        )

    @property
    def batch_size(self) -> int:
        return self.batch_size_per_gpu

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model(self) -> Any:
        return self._model

    @property
    def processor(self) -> Any:
        return self._processor

    @property
    def rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        raise NotImplementedError("llava_anything currently supports lmms-eval generate_until tasks only.")

    def generate_until_multi_round(self, requests: list[Any]) -> list[str]:
        raise NotImplementedError("llava_anything currently supports single-round generate_until tasks only.")

    def generate_until(self, requests: list[Any]) -> list[str]:
        from lmms_eval import utils
        from tqdm import tqdm

        res: list[str] = []

        def _collate(x: tuple[Any, ...]) -> tuple[int, str]:
            return -len(str(x[0])), str(x[0])

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            if not self._logged_batching:
                print(f"[INFO] LLaVA-Anything lmms adapter: first generate_until chunk_size={len(chunk)}")
                self._logged_batching = True

            chunk_items: list[tuple[str, dict[str, Any], Image.Image, int]] = []
            for context, gen_kwargs, doc_to_visual, doc_id, task, split in chunk:
                doc = self.task_dict[task][split][doc_id]
                visuals = doc_to_visual(doc)
                image = _first_image(visuals)
                if image is None:
                    raise ValueError("llava_anything requires one image for generate_until requests.")

                max_new_tokens = int(gen_kwargs.get("max_new_tokens", self.default_max_new_tokens))
                chunk_items.append((str(context), gen_kwargs, image, max_new_tokens))

            chunk_responses: list[str | None] = [None] * len(chunk_items)
            max_new_tokens_values = sorted({item[3] for item in chunk_items})
            for max_new_tokens in max_new_tokens_values:
                indices = [idx for idx, item in enumerate(chunk_items) if item[3] == max_new_tokens]
                responses = generate_responses(
                    model=self.model,
                    processor=self.processor,
                    images=[chunk_items[idx][2] for idx in indices],
                    prompts=[chunk_items[idx][0] for idx in indices],
                    system_prompt=self.system_prompt,
                    max_new_tokens=max_new_tokens,
                )
                for idx, response in zip(indices, responses, strict=True):
                    chunk_responses[idx] = response

            for (context, gen_kwargs, _image, _max_new_tokens), response in zip(
                chunk_items, chunk_responses, strict=True
            ):
                if response is None:
                    raise RuntimeError("Missing generation response for lmms-eval request.")
                res.append(response)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), response)
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)


def _first_image(visuals: Any) -> Image.Image | None:
    if isinstance(visuals, Image.Image):
        if visuals.mode == "RGB":
            return visuals
        return visuals.convert("RGB")
    if isinstance(visuals, list):
        for item in visuals:
            image = _first_image(item)
            if image is not None:
                return image
    return None


def _prepare_processor_for_batched_generation(processor: Any) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return

    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return

    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token is not None:
        tokenizer.pad_token = eos_token
    if eos_token_id is not None:
        tokenizer.pad_token_id = eos_token_id
