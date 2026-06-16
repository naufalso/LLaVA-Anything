"""Training utilities for LLaVa-Anything."""

from __future__ import annotations

import argparse
from collections import deque
import faulthandler
from fnmatch import fnmatch
import os
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from .builder import config_from_yaml_dict, load_yaml, model_from_yaml_dict, processor_from_yaml_dict
from .modeling_llava_anything import LlavaAnythingForConditionalGeneration
from .processing_llava_anything import LlavaAnythingProcessor
from .dataset import (
    IGNORE_INDEX,
    LlavaPretrainDataCollator,
    LlavaPretrainDataset,
    _conversation_text,
    _is_main_process,
    _launcher_world_size,
    _load_json_records,
    _preview_record,
    _process_rank,
    _render_prefix,
    _resolve_model_max_length,
    _role_name,
    _tokenize_text,
    log_preview_samples,
)

import torch.distributed as dist


class LlavaAnythingTrainer(Trainer):
    """Trainer with checks for invalid supervised-loss batches."""

    def __init__(
        self,
        *args: Any,
        skip_nonfinite_loss: bool = False,
        skip_nonfinite_gradients: bool | None = None,
        max_consecutive_nonfinite_losses: int = 8,
        finite_parameter_check_steps: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.skip_nonfinite_loss = skip_nonfinite_loss
        self.skip_nonfinite_gradients = (
            skip_nonfinite_loss if skip_nonfinite_gradients is None else bool(skip_nonfinite_gradients)
        )
        self.nonfinite_loss_batches = 0
        self.nonfinite_gradient_steps = 0
        self.max_consecutive_nonfinite_losses = max(1, int(max_consecutive_nonfinite_losses))
        self._consecutive_nonfinite_losses = 0
        self._consecutive_nonfinite_loss_windows = 0
        self._skip_next_optimizer_step = False
        self._optimizer_step_skip_reason = "this accumulation window was marked unsafe"
        self._optimizer_step_skip_marked_by_guard = False
        self._wrapped_optimizer: Any | None = None
        self._wrapped_deepspeed_backward: Any | None = None
        self._last_batch_metadata: Any | None = None
        self._recent_batch_metadata = deque(maxlen=5)
        self.finite_parameter_check_steps = max(0, int(finite_parameter_check_steps))
        self._optimizer_step_attempts = 0
        self._first_training_step_seen = False
        self._visible_zero3_nonfinite_parameter_warning_count = 0
        self._start_startup_watchdog()

    @staticmethod
    def _rank_log(message: str) -> None:
        print(f"[RANK {_process_rank()}] {message}", flush=True)

    def _start_startup_watchdog(self) -> None:
        """Dump Python stacks if distributed setup hangs before the first batch."""

        raw_seconds = os.environ.get("LLAVA_TRAINING_STARTUP_WATCHDOG_SECONDS")
        if not raw_seconds:
            return
        try:
            seconds = int(raw_seconds)
        except ValueError:
            warnings.warn(
                f"Ignoring invalid LLAVA_TRAINING_STARTUP_WATCHDOG_SECONDS={raw_seconds!r}.",
                UserWarning,
                stacklevel=2,
            )
            return
        if seconds <= 0:
            return
        faulthandler.enable(file=sys.stderr, all_threads=True)

        def watchdog() -> None:
            while not self._first_training_step_seen:
                time.sleep(seconds)
                if self._first_training_step_seen:
                    return
                print(
                    f"[RANK {_process_rank()}] No training step reached after another "
                    f"{seconds}s; dumping Python stacks for startup-hang diagnosis.",
                    file=sys.stderr,
                    flush=True,
                )
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

        threading.Thread(target=watchdog, name="llava-startup-watchdog", daemon=True).start()

    @staticmethod
    def _any_rank_should_skip_loss(loss: torch.Tensor) -> bool:
        """Return whether any distributed rank saw a non-finite loss."""

        local_should_skip = 0 if torch.isfinite(loss.detach()).all() else 1
        should_skip = torch.tensor(local_should_skip, device=loss.device, dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(should_skip, op=dist.ReduceOp.MAX)
        return bool(should_skip.item())

    def _any_rank_should_skip_gradient_norm(self, grad_norm: Any) -> bool:
        """Return whether any distributed rank saw a non-finite gradient norm."""

        if torch.is_tensor(grad_norm):
            local_should_skip = 0 if torch.isfinite(grad_norm.detach()).all() else 1
            device = grad_norm.device
        else:
            try:
                local_should_skip = 0 if torch.isfinite(torch.tensor(float(grad_norm))).item() else 1
            except (TypeError, ValueError, OverflowError):
                local_should_skip = 1
            device = self.args.device
        should_skip = torch.tensor(local_should_skip, device=device, dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(should_skip, op=dist.ReduceOp.MAX)
        return bool(should_skip.item())

    @staticmethod
    def _skipped_loss_like(loss: torch.Tensor) -> torch.Tensor:
        """Return a scalar finite loss accepted by DeepSpeed backward."""

        zero_source = torch.zeros((), device=loss.device, dtype=torch.float32, requires_grad=True)
        return zero_source * 0.0

    def _mark_optimizer_step_skipped(self) -> None:
        """Make Accelerate report that the current optimizer step was skipped."""

        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None and hasattr(optimizer, "_is_overflow"):
            optimizer._is_overflow = True
            self._optimizer_step_skip_marked_by_guard = True

    def _request_optimizer_step_skip(self, reason: str) -> None:
        self._skip_next_optimizer_step = True
        self._optimizer_step_skip_reason = reason
        self._mark_optimizer_step_skipped()

    def _consume_optimizer_step_skip(self) -> str:
        reason = getattr(self, "_optimizer_step_skip_reason", "this accumulation window was marked unsafe")
        self._skip_next_optimizer_step = False
        self._optimizer_step_skip_reason = "this accumulation window was marked unsafe"
        return reason

    @staticmethod
    def _is_nonfinite_loss_skip_reason(reason: str) -> bool:
        return "non-finite loss" in reason

    def _record_skipped_optimizer_window(self, reason: str) -> bool:
        if self._is_nonfinite_loss_skip_reason(reason):
            self._consecutive_nonfinite_loss_windows += 1
            return self._consecutive_nonfinite_loss_windows >= self.max_consecutive_nonfinite_losses
        return False

    def _persistent_nonfinite_loss_error(self) -> RuntimeError:
        state_bad, state_message = self._any_rank_has_nonfinite_training_state()
        state_status = (
            f"non-finite training state detected: {state_message}"
            if state_bad
            else f"no non-finite training state detected locally: {state_message}"
        )
        return RuntimeError(
            "Persistent non-finite training loss detected after "
            f"{self._consecutive_nonfinite_loss_windows} consecutive skipped optimizer window(s). "
            "The model state may be non-finite or this shard is repeatedly producing invalid losses. "
            f"state_check={state_status}. "
            f"recent_batches={self._format_recent_batch_metadata()}"
        )

    @staticmethod
    def _format_batch_metadata(metadata: Any, limit: int = 4) -> str:
        """Format compact batch metadata for warnings without flooding Slurm logs."""

        if not metadata:
            return ""
        items = list(metadata) if isinstance(metadata, (list, tuple)) else [metadata]
        formatted: list[str] = []
        for item in items[:limit]:
            if isinstance(item, dict):
                fields = []
                if "record_index" in item:
                    fields.append(f"idx={item['record_index']}")
                if "record_id" in item:
                    fields.append(f"id={item['record_id']!r}")
                if "image_path" in item:
                    fields.append(f"image={item['image_path']!r}")
                formatted.append("{" + ", ".join(fields) + "}" if fields else repr(item))
            else:
                formatted.append(repr(item))
        remaining = len(items) - limit
        if remaining > 0:
            formatted.append(f"... +{remaining} more")
        return " batch_metadata=[" + "; ".join(formatted) + "]"

    def _format_recent_batch_metadata(self) -> str:
        """Format the current batch plus recent predecessors for first-NaN debugging."""

        if not self._recent_batch_metadata:
            return ""
        formatted_batches = []
        recent = list(self._recent_batch_metadata)
        for offset, metadata in enumerate(recent):
            distance = len(recent) - offset - 1
            label = "current" if distance == 0 else f"prev_{distance}"
            formatted = self._format_batch_metadata(metadata, limit=2).strip()
            formatted_batches.append(f"{label}: {formatted or 'batch_metadata=[]'}")
        return " recent_batches=[" + " | ".join(formatted_batches) + "]"

    def _wrap_optimizer_step_if_needed(self) -> None:
        """Guard optimizer.step so skipped non-finite windows cannot update weights."""

        optimizer = getattr(self, "optimizer", None)
        if optimizer is None:
            return
        if not hasattr(optimizer, "step"):
            return
        if optimizer is self._wrapped_optimizer and getattr(
            optimizer.step, "_llava_anything_nonfinite_guard", False
        ):
            return

        original_step = optimizer.step

        def guarded_step(*args: Any, **kwargs: Any) -> Any:
            if self._skip_next_optimizer_step:
                reason = self._consume_optimizer_step_skip()
                self._mark_optimizer_step_skipped()
                should_abort = self._record_skipped_optimizer_window(reason)
                warnings.warn(
                    f"[rank {_process_rank()}] Skipping optimizer step because {reason}.",
                    UserWarning,
                    stacklevel=2,
                )
                if should_abort:
                    raise self._persistent_nonfinite_loss_error()
                return None
            if self._optimizer_step_skip_marked_by_guard and hasattr(optimizer, "_is_overflow"):
                optimizer._is_overflow = False
                self._optimizer_step_skip_marked_by_guard = False
            result = original_step(*args, **kwargs)
            if not self._deepspeed_engine():
                self._after_actual_optimizer_step()
            return result

        guarded_step._llava_anything_nonfinite_guard = True
        if getattr(original_step, "_wrapped_by_lr_sched", False):
            guarded_step._wrapped_by_lr_sched = True
        optimizer.step = guarded_step
        self._wrapped_optimizer = optimizer

    def create_optimizer(self, model: torch.nn.Module | None = None) -> torch.optim.Optimizer:
        self._rank_log("Entering Trainer.create_optimizer().")
        optimizer = super().create_optimizer(model=model)
        self._rank_log(f"Finished Trainer.create_optimizer(): {type(optimizer).__name__}.")
        self._wrap_optimizer_step_if_needed()
        return optimizer

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        self._rank_log("Building train dataloader.")
        dataloader = super().get_train_dataloader()
        try:
            length = len(dataloader)
        except TypeError:
            length = "unknown"
        self._rank_log(f"Built train dataloader with length={length}.")
        return dataloader

    def _prepare_for_training(
        self,
        max_steps: int,
        train_dataloader: torch.utils.data.DataLoader,
        resume_from_checkpoint: str | bool | None,
    ) -> tuple[torch.nn.Module, torch.utils.data.DataLoader]:
        self._rank_log("Entering Trainer._prepare_for_training().")
        result = super()._prepare_for_training(max_steps, train_dataloader, resume_from_checkpoint)
        self._rank_log("Finished Trainer._prepare_for_training().")
        return result

    def _deepspeed_engine(self) -> Any | None:
        accelerator = getattr(self, "accelerator", None)
        wrapped = getattr(accelerator, "deepspeed_engine_wrapped", None)
        return getattr(wrapped, "engine", None)

    def _deepspeed_zero_optimizer(self) -> Any | None:
        engine = self._deepspeed_engine()
        return getattr(engine, "optimizer", None)

    def _after_actual_optimizer_step(self) -> None:
        self._optimizer_step_attempts = getattr(self, "_optimizer_step_attempts", 0) + 1
        self._consecutive_nonfinite_losses = 0
        self._consecutive_nonfinite_loss_windows = 0
        if (
            getattr(self, "finite_parameter_check_steps", 0)
            and self._optimizer_step_attempts <= self.finite_parameter_check_steps
        ):
            self._raise_if_any_rank_has_nonfinite_training_state()

    @staticmethod
    def _tensor_nonfinite_summary(label: str, tensor: torch.Tensor) -> str | None:
        if not (torch.is_floating_point(tensor) or torch.is_complex(tensor)):
            return None
        if tensor.numel() == 0:
            return None
        finite = torch.isfinite(tensor)
        if bool(finite.all()):
            return None
        nonfinite = int((~finite).sum().item())
        return f"{label}: shape={tuple(tensor.shape)} dtype={tensor.dtype} nonfinite={nonfinite}"

    def _append_nonfinite_tensor_tree_summaries(
        self,
        summaries: list[str],
        label: str,
        value: Any,
        limit: int,
    ) -> None:
        if len(summaries) >= limit:
            return
        if torch.is_tensor(value):
            summary = self._tensor_nonfinite_summary(label, value.detach())
            if summary is not None:
                summaries.append(summary)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                self._append_nonfinite_tensor_tree_summaries(summaries, f"{label}[{key!r}]", item, limit)
                if len(summaries) >= limit:
                    return
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self._append_nonfinite_tensor_tree_summaries(summaries, f"{label}[{index}]", item, limit)
                if len(summaries) >= limit:
                    return

    @staticmethod
    def _zero3_parameter_is_available(parameter: torch.nn.Parameter) -> bool:
        """Return whether a ZeRO-3 parameter's visible tensor data is real."""

        status = getattr(parameter, "ds_status", None)
        if status is None:
            return True
        return getattr(status, "name", None) == "AVAILABLE"

    def _local_nonfinite_gradient_summaries(self, limit: int = 8) -> list[str]:
        summaries: list[str] = []
        model = getattr(self, "model", None)
        if model is not None:
            for name, parameter in model.named_parameters():
                grad = getattr(parameter, "grad", None)
                if grad is None:
                    continue
                summary = self._tensor_nonfinite_summary(f"{name}.grad", grad.detach())
                if summary is not None:
                    summaries.append(summary)
                    if len(summaries) >= limit:
                        return summaries

        zero_optimizer = self._deepspeed_zero_optimizer()
        if zero_optimizer is None:
            return summaries

        norm_for_param_grads = getattr(zero_optimizer, "norm_for_param_grads", {})
        if isinstance(norm_for_param_grads, dict):
            for param_id, norm in norm_for_param_grads.items():
                if torch.is_tensor(norm):
                    summary = self._tensor_nonfinite_summary(
                        f"deepspeed.norm_for_param_grads[{param_id}]",
                        norm.detach(),
                    )
                    if summary is not None:
                        summaries.append(summary)
                else:
                    try:
                        if not torch.isfinite(torch.tensor(float(norm))).item():
                            summaries.append(f"deepspeed.norm_for_param_grads[{param_id}]: value={norm!r}")
                    except (TypeError, ValueError, OverflowError):
                        summaries.append(f"deepspeed.norm_for_param_grads[{param_id}]: value={norm!r}")
                if len(summaries) >= limit:
                    return summaries

        grad_partitions_flat_buffer = getattr(zero_optimizer, "grad_partitions_flat_buffer", None)
        self._append_nonfinite_tensor_tree_summaries(
            summaries,
            "deepspeed.grad_partitions_flat_buffer",
            grad_partitions_flat_buffer,
            limit,
        )
        if len(summaries) >= limit:
            return summaries

        averaged_gradients = getattr(zero_optimizer, "averaged_gradients", None)
        self._append_nonfinite_tensor_tree_summaries(
            summaries,
            "deepspeed.averaged_gradients",
            averaged_gradients,
            limit,
        )
        if len(summaries) >= limit:
            return summaries

        fp32_groups = getattr(zero_optimizer, "fp32_partitioned_groups_flat", [])
        for group_index, flat_param in enumerate(fp32_groups):
            grad = getattr(flat_param, "grad", None)
            if grad is None:
                continue
            summary = self._tensor_nonfinite_summary(
                f"deepspeed.fp32_partitioned_groups_flat[{group_index}].grad",
                grad.detach(),
            )
            if summary is not None:
                summaries.append(summary)
                if len(summaries) >= limit:
                    return summaries

        return summaries

    def _any_rank_has_nonfinite_gradient_state(self) -> tuple[bool, str]:
        summaries = self._local_nonfinite_gradient_summaries()
        local_bad = 1 if summaries else 0
        bad = torch.tensor(local_bad, device=self.args.device, dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        local_message = "; ".join(summaries) if summaries else "no local non-finite gradient state"
        return bool(bad.item()), local_message

    def _clear_deepspeed_gradient_state(self) -> None:
        engine = self._deepspeed_engine()
        zero_optimizer = self._deepspeed_zero_optimizer()
        if zero_optimizer is not None:
            if hasattr(zero_optimizer, "zero_grad"):
                zero_optimizer.zero_grad(set_to_none=True)
            if hasattr(zero_optimizer, "reset_cpu_buffers"):
                zero_optimizer.reset_cpu_buffers()
            if hasattr(zero_optimizer, "averaged_gradients"):
                zero_optimizer.averaged_gradients = {}
            grad_partitions_flat_buffer = getattr(zero_optimizer, "grad_partitions_flat_buffer", None)
            if torch.is_tensor(grad_partitions_flat_buffer):
                grad_partitions_flat_buffer.zero_()
            fp32_groups = getattr(zero_optimizer, "fp32_partitioned_groups_flat", [])
            for flat_param in fp32_groups:
                grad = getattr(flat_param, "grad", None)
                if torch.is_tensor(grad):
                    grad.zero_()
        if engine is not None and hasattr(engine, "zero_grad"):
            engine.zero_grad()

    def _run_deepspeed_overflow_cleanup_step(self, engine: Any) -> None:
        """Run DeepSpeed's native skipped-step cleanup without updating parameters."""

        zero_optimizer = self._deepspeed_zero_optimizer()
        if zero_optimizer is not None and hasattr(zero_optimizer, "overflow"):
            zero_optimizer.overflow = True
        engine.step()
        if zero_optimizer is not None and hasattr(zero_optimizer, "overflow"):
            zero_optimizer.overflow = False

    def _wrap_deepspeed_backward_if_needed(self) -> None:
        accelerator = getattr(self, "accelerator", None)
        wrapped = getattr(accelerator, "deepspeed_engine_wrapped", None)
        if wrapped is None:
            return
        if wrapped is self._wrapped_deepspeed_backward and getattr(
            wrapped.backward, "_llava_anything_nonfinite_guard", False
        ):
            return

        original_backward = wrapped.backward

        def guarded_backward(loss: torch.Tensor, sync_gradients: bool = True, **kwargs: Any) -> Any:
            engine = wrapped.engine
            engine.set_gradient_accumulation_boundary(is_boundary=sync_gradients)
            result = engine.backward(loss, **kwargs)
            if not sync_gradients:
                return result

            if self._skip_next_optimizer_step:
                reason = self._consume_optimizer_step_skip()
                self._mark_optimizer_step_skipped()
                should_abort = self._record_skipped_optimizer_window(reason)
                if self._is_nonfinite_loss_skip_reason(reason):
                    self._run_deepspeed_overflow_cleanup_step(engine)
                else:
                    self._clear_deepspeed_gradient_state()
                warnings.warn(
                    f"[rank {_process_rank()}] Skipping DeepSpeed optimizer step because "
                    f"{reason}.",
                    UserWarning,
                    stacklevel=2,
                )
                if should_abort:
                    raise self._persistent_nonfinite_loss_error()
                return result

            should_skip, gradient_message = self._any_rank_has_nonfinite_gradient_state()
            if self.skip_nonfinite_gradients and should_skip:
                self.nonfinite_gradient_steps += 1
                self._request_optimizer_step_skip(f"DeepSpeed gradient state is non-finite: {gradient_message}")
                reason = self._consume_optimizer_step_skip()
                self._clear_deepspeed_gradient_state()
                warnings.warn(
                    f"[rank {_process_rank()}] Skipping DeepSpeed optimizer step because "
                    f"{reason}. "
                    f"recent_batches={self._format_recent_batch_metadata()}",
                    UserWarning,
                    stacklevel=2,
                )
                return result

            engine.step()
            self._after_actual_optimizer_step()
            return result

        guarded_backward._llava_anything_nonfinite_guard = True
        guarded_backward._llava_anything_original_backward = original_backward
        wrapped.backward = guarded_backward
        self._wrapped_deepspeed_backward = wrapped

    def _local_nonfinite_parameter_summaries(self, limit: int = 8) -> list[str]:
        summaries: list[str] = []
        zero_optimizer = self._deepspeed_zero_optimizer()

        # ZeRO-3 swaps/gathers/repartitions model parameters around engine.step().
        # The visible bf16 Parameter tensors can be transient views, so use
        # DeepSpeed's fp32 partitioned tensors as the authoritative local state.
        if zero_optimizer is None:
            for name, parameter in self.model.named_parameters():
                if not self._zero3_parameter_is_available(parameter):
                    continue
                data = parameter.detach()
                if not (torch.is_floating_point(data) or torch.is_complex(data)):
                    continue
                if data.numel() == 0:
                    continue
                finite = torch.isfinite(data)
                if not bool(finite.all()):
                    nonfinite = int((~finite).sum().item())
                    summaries.append(
                        f"{name}: shape={tuple(data.shape)} dtype={data.dtype} nonfinite={nonfinite}"
                    )
                    if len(summaries) >= limit:
                        break

        if len(summaries) >= limit:
            return summaries

        if zero_optimizer is not None:
            fp32_groups = getattr(zero_optimizer, "fp32_partitioned_groups_flat", [])
            for group_index, flat_param in enumerate(fp32_groups):
                if not torch.is_tensor(flat_param):
                    continue
                data = flat_param.detach()
                if not torch.is_floating_point(data):
                    continue
                if data.numel() == 0:
                    continue
                finite = torch.isfinite(data)
                if not bool(finite.all()):
                    nonfinite = int((~finite).sum().item())
                    summaries.append(
                        "deepspeed.fp32_partitioned_groups_flat"
                        f"[{group_index}]: shape={tuple(data.shape)} dtype={data.dtype} nonfinite={nonfinite}"
                    )
                    if len(summaries) >= limit:
                        break
        return summaries

    def _local_visible_zero3_nonfinite_parameter_summaries(self, limit: int = 8) -> list[str]:
        summaries: list[str] = []
        if self._deepspeed_zero_optimizer() is None:
            return summaries
        model = getattr(self, "model", None)
        if model is None:
            return summaries
        for name, parameter in model.named_parameters():
            data = parameter.detach()
            if not (torch.is_floating_point(data) or torch.is_complex(data)):
                continue
            if data.numel() == 0:
                continue
            finite = torch.isfinite(data)
            if bool(finite.all()):
                continue
            nonfinite = int((~finite).sum().item())
            summaries.append(f"{name}: shape={tuple(data.shape)} dtype={data.dtype} nonfinite={nonfinite}")
            if len(summaries) >= limit:
                break
        return summaries

    def _warn_if_visible_zero3_parameters_nonfinite(self) -> None:
        summaries = self._local_visible_zero3_nonfinite_parameter_summaries()
        if not summaries:
            return
        self._visible_zero3_nonfinite_parameter_warning_count = getattr(
            self,
            "_visible_zero3_nonfinite_parameter_warning_count",
            0,
        ) + 1
        if self._visible_zero3_nonfinite_parameter_warning_count > 3:
            return
        warnings.warn(
            f"[rank {_process_rank()}] Detected non-finite visible ZeRO-3 model parameter views "
            "while DeepSpeed fp32 master parameters remain the authoritative training state. "
            "Treating these visible views as diagnostic only. "
            f"visible_summaries={'; '.join(summaries)}",
            UserWarning,
            stacklevel=2,
        )

    def _local_nonfinite_buffer_summaries(self, limit: int = 8) -> list[str]:
        summaries: list[str] = []
        model = getattr(self, "model", None)
        if model is None:
            return summaries
        for name, buffer in model.named_buffers():
            summary = self._tensor_nonfinite_summary(name, buffer.detach())
            if summary is not None:
                summaries.append(summary)
                if len(summaries) >= limit:
                    break
        return summaries

    def _local_nonfinite_optimizer_state_summaries(self, limit: int = 8) -> list[str]:
        summaries: list[str] = []
        candidates: list[tuple[str, Any]] = []
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            candidates.append(("trainer.optimizer", optimizer))
        zero_optimizer = self._deepspeed_zero_optimizer()
        if zero_optimizer is not None:
            candidates.append(("deepspeed.optimizer", zero_optimizer))
            inner_optimizer = getattr(zero_optimizer, "optimizer", None)
            if inner_optimizer is not None and inner_optimizer is not zero_optimizer:
                candidates.append(("deepspeed.optimizer.optimizer", inner_optimizer))

        seen: set[int] = set()
        for optimizer_label, optimizer_obj in candidates:
            if id(optimizer_obj) in seen:
                continue
            seen.add(id(optimizer_obj))
            state = getattr(optimizer_obj, "state", None)
            if not isinstance(state, dict):
                continue
            for state_index, state_value in enumerate(state.values()):
                if torch.is_tensor(state_value):
                    summary = self._tensor_nonfinite_summary(
                        f"{optimizer_label}.state[{state_index}]",
                        state_value.detach(),
                    )
                    if summary is not None:
                        summaries.append(summary)
                elif isinstance(state_value, dict):
                    for item_key, item_value in state_value.items():
                        if not torch.is_tensor(item_value):
                            continue
                        summary = self._tensor_nonfinite_summary(
                            f"{optimizer_label}.state[{state_index}][{item_key!r}]",
                            item_value.detach(),
                        )
                        if summary is not None:
                            summaries.append(summary)
                        if len(summaries) >= limit:
                            return summaries
                if len(summaries) >= limit:
                    return summaries
        return summaries

    def _local_nonfinite_training_state_summaries(self, limit: int = 8) -> list[str]:
        summaries = self._local_nonfinite_parameter_summaries(limit=limit)
        if len(summaries) >= limit:
            return summaries
        summaries.extend(self._local_nonfinite_buffer_summaries(limit=limit - len(summaries)))
        if len(summaries) >= limit:
            return summaries
        summaries.extend(self._local_nonfinite_optimizer_state_summaries(limit=limit - len(summaries)))
        return summaries

    def _any_rank_has_nonfinite_training_state(self) -> tuple[bool, str]:
        summaries = self._local_nonfinite_training_state_summaries()
        local_bad = 1 if summaries else 0
        bad = torch.tensor(local_bad, device=self.args.device, dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        local_message = "; ".join(summaries) if summaries else "no local non-finite training state"
        return bool(bad.item()), local_message

    def _raise_if_any_rank_has_nonfinite_parameter(self) -> None:
        summaries = self._local_nonfinite_parameter_summaries()
        local_bad = 1 if summaries else 0
        device = self.args.device
        bad = torch.tensor(local_bad, device=device, dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if bool(bad.item()):
            local_message = "; ".join(summaries) if summaries else "no local non-finite parameter shard"
            raise RuntimeError(
                "Non-finite model parameter detected after optimizer step "
                f"{self._optimizer_step_attempts} on rank {_process_rank()}: {local_message}. "
                f"recent_batches={self._format_recent_batch_metadata()}"
            )

    def _raise_if_any_rank_has_nonfinite_training_state(self) -> None:
        self._warn_if_visible_zero3_parameters_nonfinite()
        bad, message = self._any_rank_has_nonfinite_training_state()
        if bad:
            raise RuntimeError(
                "Non-finite training state detected after optimizer step "
                f"{self._optimizer_step_attempts} on rank {_process_rank()}: {message}. "
                f"recent_batches={self._format_recent_batch_metadata()}"
            )

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        if not self._first_training_step_seen:
            self._first_training_step_seen = True
            self._rank_log("First training_step reached.")
        self._wrap_optimizer_step_if_needed()
        self._wrap_deepspeed_backward_if_needed()
        return super().training_step(model, inputs, num_items_in_batch)

    def _get_grad_norm(self, model: torch.nn.Module, grad_norm: Any = None) -> Any:
        """Return gradient norm and mark unsafe optimizer steps before they corrupt weights."""

        grad_norm = super()._get_grad_norm(model, grad_norm=grad_norm)
        if self.skip_nonfinite_gradients and self._any_rank_should_skip_gradient_norm(grad_norm):
            self.nonfinite_gradient_steps += 1
            self._skip_next_optimizer_step = True
            warnings.warn(
                f"[rank {_process_rank()}] Skipping optimizer step because gradient norm is non-finite: "
                f"{grad_norm!r}. recent_batches={self._format_recent_batch_metadata()}",
                UserWarning,
                stacklevel=2,
            )
        return grad_norm

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        batch_metadata = inputs.pop("_metadata", None)
        self._last_batch_metadata = batch_metadata
        self._recent_batch_metadata.append(batch_metadata)
        labels = inputs.get("labels")
        if torch.is_tensor(labels) and not torch.any(labels != IGNORE_INDEX):
            raise ValueError("Refusing to train on a batch with no supervised target tokens.")

        result = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        loss = result[0] if return_outputs else result
        should_handle_loss = torch.is_tensor(loss) and (
            self._any_rank_should_skip_loss(loss)
            if self.skip_nonfinite_loss
            else not torch.isfinite(loss.detach()).all()
        )
        if should_handle_loss:
            if not self.skip_nonfinite_loss:
                raise FloatingPointError(f"Non-finite training loss detected: {loss.detach().item()}")

            self.nonfinite_loss_batches += 1
            self._consecutive_nonfinite_losses += 1
            self._request_optimizer_step_skip("this accumulation window had a non-finite loss")
            if self.nonfinite_loss_batches == 1:
                state_bad, state_message = self._any_rank_has_nonfinite_training_state()
                if state_bad:
                    raise RuntimeError(
                        "Non-finite training state detected at first non-finite loss "
                        f"after optimizer step {self._optimizer_step_attempts} "
                        f"on rank {_process_rank()}: {state_message}. "
                        f"recent_batches={self._format_recent_batch_metadata()}"
                    )
                warnings.warn(
                    f"[rank {_process_rank()}] First non-finite loss did not coincide with "
                    f"non-finite local parameter, buffer, ZeRO master parameter, or optimizer state. "
                    f"state_check={state_message}.",
                    UserWarning,
                    stacklevel=2,
                )
            finite_on_this_rank = torch.isfinite(loss.detach()).all()
            loss_message = "finite peer-skipped loss" if finite_on_this_rank else str(loss.detach().item())
            metadata_message = self._format_batch_metadata(batch_metadata)
            recent_metadata_message = (
                self._format_recent_batch_metadata() if self.nonfinite_loss_batches == 1 else ""
            )
            warnings.warn(
                f"[rank {_process_rank()}] Skipping non-finite training loss "
                f"{loss_message} at skipped batch #{self.nonfinite_loss_batches}.{metadata_message}"
                f"{recent_metadata_message}",
                UserWarning,
                stacklevel=2,
            )
            zero_loss = self._skipped_loss_like(loss)
            if return_outputs:
                return zero_loss, result[1]
            return zero_loss
        if torch.is_tensor(loss):
            self._consecutive_nonfinite_losses = 0
        return result


def apply_trainable_modules(model: LlavaAnythingForConditionalGeneration, trainable_modules: str = "projector") -> list[str]:
    """Freeze all parameters except the requested trainable module groups."""

    modules = {part.strip() for part in trainable_modules.split(",") if part.strip()}
    supported = {"projector", "vision_tower", "language_model", "full"}
    unknown = modules - supported
    if unknown:
        raise ValueError(f"Unsupported trainable module selection: {sorted(unknown)}")

    model.requires_grad_(False)
    if "full" in modules:
        model.requires_grad_(True)
    if "projector" in modules or "full" in modules:
        model.multi_modal_projector.requires_grad_(True)
    if "vision_tower" in modules or "full" in modules:
        model.vision_tower.requires_grad_(True)
    if "language_model" in modules or "full" in modules:
        model.language_model.requires_grad_(True)

    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def apply_frozen_parameter_patterns(
    model: LlavaAnythingForConditionalGeneration,
    frozen_parameter_patterns: list[str],
) -> list[str]:
    """Freeze trainable parameters whose names match any shell-style pattern."""

    frozen_names: list[str] = []
    patterns = [pattern.strip() for pattern in frozen_parameter_patterns if pattern.strip()]
    if not patterns:
        return frozen_names

    for name, parameter in model.named_parameters():
        if parameter.requires_grad and any(fnmatch(name, pattern) for pattern in patterns):
            parameter.requires_grad_(False)
            frozen_names.append(name)
    return frozen_names


@dataclass
class LlavaPretrainingResult:
    train_result: Any
    output_dir: Path
    trainable_parameter_names: list[str]


def configure_wandb(training_section: dict[str, Any], wandb_section: dict[str, Any] | None) -> None:
    """Apply optional Weights & Biases settings to training args and environment variables."""

    if wandb_section is None:
        return

    wandb_section = dict(wandb_section)
    if wandb_section.get("enabled") is False:
        training_section.setdefault("report_to", [])
        return

    training_section.setdefault("report_to", ["wandb"])
    if isinstance(training_section["report_to"], str):
        training_section["report_to"] = [training_section["report_to"]]
    if "wandb" not in training_section["report_to"]:
        training_section["report_to"] = [*training_section["report_to"], "wandb"]

    env_map = {
        "project": "WANDB_PROJECT",
        "entity": "WANDB_ENTITY",
        "mode": "WANDB_MODE",
        "name": "WANDB_NAME",
    }
    for key, env_name in env_map.items():
        value = wandb_section.get(key)
        if value is not None:
            os.environ[env_name] = str(value)
    if wandb_section.get("name") and "run_name" not in training_section:
        training_section["run_name"] = str(wandb_section["name"])


def _coerce_training_arguments(training_section: dict[str, Any]) -> TrainingArguments:
    """Convert a YAML training section into Hugging Face TrainingArguments."""

    kwargs = dict(training_section)
    kwargs.setdefault("remove_unused_columns", False)
    kwargs.setdefault("report_to", [])
    kwargs.setdefault("save_strategy", "no")
    if kwargs.get("save_strategy") is False:
        kwargs["save_strategy"] = "no"
    kwargs.setdefault("logging_steps", 1)
    kwargs.setdefault("logging_nan_inf_filter", False)
    kwargs.setdefault("disable_tqdm", True)
    if "output_dir" not in kwargs:
        raise ValueError("training.output_dir is required.")
    return TrainingArguments(**kwargs)


def _resolve_resume_from_checkpoint(training_args: TrainingArguments) -> str | bool | None:
    """Resolve the checkpoint to resume from, auto-detecting the latest checkpoint by default."""

    explicit_resume = getattr(training_args, "resume_from_checkpoint", None)
    if explicit_resume is not None:
        return explicit_resume

    output_dir = Path(training_args.output_dir)
    if not output_dir.is_dir():
        return None
    return get_last_checkpoint(str(output_dir))


def _load_training_yaml(path: str | Path) -> dict[str, Any]:
    """Load a training YAML file and require a top-level mapping."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _coerce_torch_dtype(value: Any) -> Any:
    """Convert string dtype names into torch dtype objects when possible."""

    if not isinstance(value, str) or value == "auto":
        return value
    if hasattr(torch, value):
        dtype = getattr(torch, value)
        if isinstance(dtype, torch.dtype):
            return dtype
    return value


def _coerce_model_kwargs(model_kwargs: dict[str, Any] | None) -> dict[str, Any] | None:
    """Recursively coerce model-loading dtype kwargs from YAML-friendly strings."""

    if model_kwargs is None:
        return None
    coerced = dict(model_kwargs)
    if "torch_dtype" in coerced:
        coerced["torch_dtype"] = _coerce_torch_dtype(coerced["torch_dtype"])
    for nested_key in ("text_model_kwargs", "vision_model_kwargs"):
        if isinstance(coerced.get(nested_key), dict) and "torch_dtype" in coerced[nested_key]:
            coerced[nested_key] = dict(coerced[nested_key])
            coerced[nested_key]["torch_dtype"] = _coerce_torch_dtype(coerced[nested_key]["torch_dtype"])
    return coerced


def _build_model_and_processor(
    model_yaml: str | Path,
    load_pretrained_components: bool = True,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[LlavaAnythingForConditionalGeneration, LlavaAnythingProcessor]:
    """Build model and processor from a model YAML, resizing embeddings if needed."""

    model_data = load_yaml(model_yaml)
    config = config_from_yaml_dict(model_data)
    processor = processor_from_yaml_dict(model_data, config)
    model = model_from_yaml_dict(
        model_data,
        config=config,
        load_pretrained_components=load_pretrained_components,
        model_kwargs=_coerce_model_kwargs(model_kwargs),
    )

    tokenizer_vocab_size = len(processor.tokenizer)
    embedding_vocab_size = model.get_input_embeddings().num_embeddings
    if tokenizer_vocab_size > embedding_vocab_size:
        try:
            model.resize_token_embeddings(tokenizer_vocab_size, mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(tokenizer_vocab_size)
        resized_vocab_size = model.get_input_embeddings().num_embeddings
        model.config.text_config.vocab_size = resized_vocab_size
        model.config.vocab_size = resized_vocab_size
    return model, processor


def _load_checkpoint_model_and_processor(
    model_checkpoint: str | Path,
    model_kwargs: dict[str, Any] | None = None,
    processor_checkpoint: str | Path | None = None,
) -> tuple[LlavaAnythingForConditionalGeneration, LlavaAnythingProcessor]:
    """Load a saved LLaVa-Anything checkpoint and its processor."""

    checkpoint = Path(model_checkpoint)
    processor_source = Path(processor_checkpoint) if processor_checkpoint is not None else checkpoint
    processor = LlavaAnythingProcessor.from_pretrained(processor_source)
    model = LlavaAnythingForConditionalGeneration.from_pretrained(
        checkpoint,
        **(_coerce_model_kwargs(model_kwargs) or {}),
    )
    return model, processor


def run_training_from_yaml(path: str | Path) -> LlavaPretrainingResult:
    """Run the full pretraining loop described by a YAML configuration."""

    data = _load_training_yaml(path)
    model_yaml = data.get("model_yaml")
    model_checkpoint = data.get("model_checkpoint")
    if bool(model_yaml) == bool(model_checkpoint):
        raise ValueError("Exactly one of model_yaml or model_checkpoint is required.")
    data_section = data.get("data", {})
    if not isinstance(data_section, dict):
        raise ValueError("data must be a mapping.")
    training_section = data.get("training", {})
    if not isinstance(training_section, dict):
        raise ValueError("training must be a mapping.")
    if "output_dir" not in training_section:
        raise ValueError("training.output_dir is required.")
    logging_section = data.get("logging", {}) or {}
    if not isinstance(logging_section, dict):
        raise ValueError("logging must be a mapping when provided.")
    wandb_section = data["wandb"] if "wandb" in data else None
    if wandb_section is None and "wandb" in data:
        wandb_section = {}
    if wandb_section is not None and not isinstance(wandb_section, dict):
        raise ValueError("wandb must be a mapping when provided.")
    configure_wandb(training_section, wandb_section)

    if model_checkpoint:
        model, processor = _load_checkpoint_model_and_processor(
            model_checkpoint,
            model_kwargs=data.get("model_kwargs"),
            processor_checkpoint=data.get("processor_checkpoint"),
        )
    else:
        model, processor = _build_model_and_processor(
            model_yaml,
            load_pretrained_components=bool(data.get("load_pretrained_components", True)),
            model_kwargs=data.get("model_kwargs"),
        )
    model.config.use_cache = False
    if hasattr(model.language_model, "config"):
        model.language_model.config.use_cache = False

    trainable_modules = str(training_section.get("trainable_modules", "projector"))
    training_args_data = dict(training_section)
    training_args_data.pop("trainable_modules", None)
    model_max_length = training_args_data.pop("model_max_length", None)
    skip_nonfinite_loss = bool(training_args_data.pop("skip_nonfinite_loss", False))
    skip_nonfinite_gradients = training_args_data.pop("skip_nonfinite_gradients", None)
    if skip_nonfinite_gradients is not None:
        skip_nonfinite_gradients = bool(skip_nonfinite_gradients)
    max_consecutive_nonfinite_losses = int(training_args_data.pop("max_consecutive_nonfinite_losses", 8))
    finite_parameter_check_steps = int(training_args_data.pop("finite_parameter_check_steps", 0))
    frozen_parameter_patterns = training_args_data.pop("frozen_parameter_patterns", [])
    if frozen_parameter_patterns is None:
        frozen_parameter_patterns = []
    if isinstance(frozen_parameter_patterns, str):
        frozen_parameter_patterns = [frozen_parameter_patterns]
    if not isinstance(frozen_parameter_patterns, list) or not all(
        isinstance(pattern, str) for pattern in frozen_parameter_patterns
    ):
        raise ValueError("training.frozen_parameter_patterns must be a string or list of strings.")
    trainable_names = apply_trainable_modules(model, trainable_modules)
    frozen_names = apply_frozen_parameter_patterns(model, frozen_parameter_patterns)
    if frozen_names and _is_main_process():
        print(f"Froze {len(frozen_names)} trainable parameter(s) matching frozen_parameter_patterns.")
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    dataloader_num_workers = int(training_section.get("dataloader_num_workers") or 0)
    image_token_mismatch_num_workers = data_section.get("image_token_mismatch_num_workers")
    if image_token_mismatch_num_workers is None:
        image_token_mismatch_num_workers = data_section.get("image_token_prefilter_num_workers")
    if image_token_mismatch_num_workers is not None:
        image_token_mismatch_num_workers = int(image_token_mismatch_num_workers)
    available_images_cache_dir = data_section.get("available_images_cache_dir")
    if available_images_cache_dir is None and bool(
        data_section.get("available_images_cache", True)
    ):
        available_images_cache_dir = training_section["output_dir"]
    data_path = data_section.get("data_path")
    hf_dataset_path = data_section.get("hf_dataset_path")
    if bool(data_path) == bool(hf_dataset_path):
        raise ValueError("data must provide exactly one of data_path or hf_dataset_path.")
    image_folder = data_section.get("image_folder")
    if data_path is not None and image_folder is None:
        raise ValueError("data.image_folder is required when data.data_path is used.")
    hf_dataset_name = data_section.get(
        "hf_dataset_name",
        data_section.get("hf_dataset_config", data_section.get("hf_dataset_config_name")),
    )

    dataset = LlavaPretrainDataset(
        data_path=data_path,
        image_folder=image_folder,
        processor=processor,
        max_samples=data_section.get("max_samples"),
        available_images_only=bool(data_section.get("available_images_only", True)),
        available_images_cache_dir=available_images_cache_dir,
        available_images_num_workers=dataloader_num_workers,
        refresh_available_images_cache=bool(
            data_section.get("refresh_available_images_cache", False)
        ),
        require_image=bool(data_section.get("require_image", False)),
        min_image_width=data_section.get("min_image_width"),
        min_image_height=data_section.get("min_image_height"),
        max_image_aspect_ratio=data_section.get("max_image_aspect_ratio"),
        max_image_tokens=data_section.get("max_image_tokens"),
        image_constraint_prefilter=bool(data_section.get("image_constraint_prefilter", False)),
        image_constraint_num_workers=data_section.get("image_constraint_num_workers"),
        image_token_mismatch_prefilter=bool(
            data_section.get(
                "image_token_mismatch_prefilter",
                data_section.get("image_token_prefilter", False),
            )
        ),
        image_token_mismatch_num_workers=image_token_mismatch_num_workers,
        system_prompt=data_section.get("system_prompt"),
        model_max_length=model_max_length,
        hf_dataset_path=hf_dataset_path,
        hf_dataset_name=hf_dataset_name,
        hf_dataset_split=data_section.get("hf_dataset_split", "train"),
        hf_dataset_revision=data_section.get("hf_dataset_revision"),
    )
    log_preview_samples(dataset, int(logging_section.get("preview_samples", 0)))
    collator = LlavaPretrainDataCollator(processor.tokenizer, include_metadata=True)
    training_args = _coerce_training_arguments(training_args_data)
    trainer = LlavaAnythingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        skip_nonfinite_loss=skip_nonfinite_loss,
        skip_nonfinite_gradients=skip_nonfinite_gradients,
        max_consecutive_nonfinite_losses=max_consecutive_nonfinite_losses,
        finite_parameter_check_steps=finite_parameter_check_steps,
    )
    resume_from_checkpoint = _resolve_resume_from_checkpoint(training_args)
    if resume_from_checkpoint and _is_main_process():
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    return LlavaPretrainingResult(
        train_result=train_result,
        output_dir=output_dir,
        trainable_parameter_names=trainable_names,
    )


run_pretraining_from_yaml = run_training_from_yaml


def main() -> None:
    """CLI entry point for running LLaVa-Anything pretraining from YAML."""

    parser = argparse.ArgumentParser(description="Run LLaVa-Anything training from a YAML config.")
    parser.add_argument("training_yaml", type=Path)
    args = parser.parse_args()
    result = run_training_from_yaml(args.training_yaml)
    print(f"training_loss: {result.train_result.training_loss}")
    print(f"output_dir: {result.output_dir}")
    print(f"trainable_parameters: {len(result.trainable_parameter_names)}")


__all__ = [
    "IGNORE_INDEX",
    "LlavaPretrainDataCollator",
    "LlavaPretrainDataset",
    "LlavaAnythingTrainer",
    "LlavaPretrainingResult",
    "apply_frozen_parameter_patterns",
    "apply_trainable_modules",
    "configure_wandb",
    "log_preview_samples",
    "run_pretraining_from_yaml",
    "run_training_from_yaml",
]


if __name__ == "__main__":
    main()
