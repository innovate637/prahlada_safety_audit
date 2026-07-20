"""
ao_lib.py — Activation Oracle inference helpers.

Adapted from Karvonen et al.'s demo notebook (activation_oracle_demo.ipynb).
This file contains the minimum machinery needed to:
  1. Load the base model + AO LoRA adapter.
  2. Inject pre-extracted activation vectors into the AO via forward hook
     (norm-matched, addition-based steering at injection layer 1).
  3. Generate the AO's natural-language response.

The original demo also includes target-model extraction logic; we strip that
because we already have activations cached from extract_activations.py.

CRITICAL: The injection mechanism is from equation (1) of the paper:
    h'_i = h_i + ||h_i|| * (v_i / ||v_i||) * steering_coefficient
This is in get_hf_activation_steering_hook below. Do not modify.
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import contextlib
from typing import Any, Callable, Mapping, Optional
from dataclasses import dataclass

import torch
import torch._dynamo as dynamo
from peft import LoraConfig
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Match Karvonen's layer counts for sanity checks
LAYER_COUNTS = {
    "Qwen/Qwen3-8B": 36,
    "google/gemma-2-9b-it": 42,
}
SPECIAL_TOKEN = " ?"   # the AO's activation placeholder
INJECTION_LAYER = 1     # where the AO accepts the injected activation


# ===========================================================================
# Pydantic-friendly data containers (copied from demo)
# ===========================================================================
class TrainingDataPoint(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    datapoint_type: str
    input_ids: list[int]
    labels: list[int]
    layer: int
    steering_vectors: torch.Tensor | None
    positions: list[int]
    feature_idx: int
    target_output: str
    target_input_ids: list[int] | None
    target_positions: list[int] | None
    ds_label: str | None
    meta_info: Mapping[str, Any] = {}


class BatchData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    steering_vectors: list[torch.Tensor]
    positions: list[list[int]]
    feature_indices: list[int]


# ===========================================================================
# Steering hook — verbatim from demo, this is the heart of the AO
# ===========================================================================
@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable):
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_hf_activation_steering_hook(
    vectors: list[torch.Tensor],
    positions: list[list[int]],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """Norm-matched, additive activation injection. See paper eq. (1)."""
    assert len(vectors) == len(positions)
    B = len(vectors)
    if B == 0:
        raise ValueError("Empty batch")
    normed_list = [torch.nn.functional.normalize(v, dim=-1).detach() for v in vectors]

    def hook_fn(module, _input, output):
        if isinstance(output, tuple):
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            resid_BLD = output
            output_is_tuple = False

        B_actual, L, _ = resid_BLD.shape
        if B_actual != B:
            raise ValueError(f"Batch mismatch: module B={B_actual}, vectors B={B}")
        if L <= 1:
            return (resid_BLD, *rest) if output_is_tuple else resid_BLD

        for b in range(B):
            pos_b = torch.tensor(positions[b], dtype=torch.long, device=device)
            assert pos_b.min() >= 0 and pos_b.max() < L
            orig_KD = resid_BLD[b, pos_b, :]
            norms_K1 = orig_KD.norm(dim=-1, keepdim=True)
            steered_KD = (normed_list[b].to(device) * norms_K1 * steering_coefficient).to(dtype)
            resid_BLD[b, pos_b, :] = steered_KD.detach() + orig_KD

        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn


# ===========================================================================
# Prompt construction (verbatim from demo)
# ===========================================================================
def get_introspection_prefix(layer: int, num_positions: int) -> str:
    return f"Layer: {layer}\n" + SPECIAL_TOKEN * num_positions + " \n"


def find_pattern_in_tokens(token_ids, special_token_str, num_positions, tokenizer):
    special_token_id = tokenizer.encode(special_token_str, add_special_tokens=False)
    assert len(special_token_id) == 1
    special_token_id = special_token_id[0]
    positions = []
    for i in range(len(token_ids)):
        if len(positions) == num_positions:
            break
        if token_ids[i] == special_token_id:
            positions.append(i)
    assert len(positions) == num_positions, f"want {num_positions}, got {len(positions)}"
    assert positions[-1] - positions[0] == num_positions - 1, "non-consecutive"
    return positions


def make_oracle_datapoint(
    oracle_prompt: str,
    activation_vector: torch.Tensor,
    layer: int,
    tokenizer: AutoTokenizer,
    meta: dict | None = None,
) -> TrainingDataPoint:
    """
    Build a single AO inference input from one cached activation vector.

    activation_vector: shape [hidden_dim] (single token) or [K, hidden_dim] (segment)
    """
    if activation_vector.dim() == 1:
        activation_vector = activation_vector.unsqueeze(0)   # -> [1, D]
    num_positions = activation_vector.shape[0]

    prefix = get_introspection_prefix(layer, num_positions)
    full_prompt = prefix + oracle_prompt
    input_messages = [{"role": "user", "content": full_prompt}]

    input_ids = tokenizer.apply_chat_template(
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    labels = [-100] * len(input_ids)   # inference: nothing to learn against
    positions = find_pattern_in_tokens(input_ids, SPECIAL_TOKEN, num_positions, tokenizer)

    return TrainingDataPoint(
        datapoint_type="inference",
        input_ids=input_ids,
        labels=labels,
        layer=layer,
        steering_vectors=activation_vector.cpu().clone().detach().contiguous(),
        positions=positions,
        feature_idx=-1,
        target_output="",
        target_input_ids=None,
        target_positions=None,
        ds_label=None,
        meta_info=meta or {},
    )


def construct_batch(
    points: list[TrainingDataPoint],
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> BatchData:
    """Left-pad a list of datapoints into a batched tensor."""
    max_length = max(len(dp.input_ids) for dp in points)
    batch_ids, batch_labels, batch_mask = [], [], []
    batch_pos, batch_vec, batch_feat = [], [], []

    for dp in points:
        pad_len = max_length - len(dp.input_ids)
        padded_ids = [tokenizer.pad_token_id] * pad_len + dp.input_ids
        padded_lbl = [-100] * pad_len + dp.labels

        ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        lbl = torch.tensor(padded_lbl, dtype=torch.long, device=device)
        msk = torch.ones_like(ids, dtype=torch.bool)
        msk[:pad_len] = False

        batch_ids.append(ids); batch_labels.append(lbl); batch_mask.append(msk)
        batch_pos.append([p + pad_len for p in dp.positions])
        batch_vec.append(dp.steering_vectors.to(device))
        batch_feat.append(dp.feature_idx)

    return BatchData(
        input_ids=torch.stack(batch_ids),
        labels=torch.stack(batch_labels),
        attention_mask=torch.stack(batch_mask),
        steering_vectors=batch_vec,
        positions=batch_pos,
        feature_indices=batch_feat,
    )


# ===========================================================================
# Submodule resolution (LoRA-aware)
# ===========================================================================
def get_injection_submodule(model, use_lora: bool = True):
    """Return the decoder layer at INJECTION_LAYER where activations get injected."""
    name = model.config._name_or_path
    if use_lora:
        # PeftModel wraps the base as: peft_model.base_model.model.<layers>
        # NOT peft_model.base_model.model.model.<layers> (that extra .model
        # was an older PEFT version). We probe both for safety.
        if "gemma" in name or "Qwen" in name:
            inner = model.base_model.model
            if hasattr(inner, "model"):
                inner = inner.model
            return inner.layers[INJECTION_LAYER]
        else:
            raise ValueError(f"Unsupported model: {name}")
    else:
        return model.model.layers[INJECTION_LAYER]


# ===========================================================================
# The inference call
# ===========================================================================
@dynamo.disable
@torch.no_grad()
def run_ao_batch(
    batch: BatchData,
    model,
    tokenizer,
    submodule,
    device,
    dtype,
    steering_coefficient: float = 1.0,
    generation_kwargs: dict | None = None,
) -> list[str]:
    """Inject activations and generate AO responses for a batch of datapoints."""
    if generation_kwargs is None:
        generation_kwargs = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 50}

    hook = get_hf_activation_steering_hook(
        vectors=batch.steering_vectors,
        positions=batch.positions,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )
    inputs = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
    with add_hook(submodule, hook):
        out_ids = model.generate(**inputs, **generation_kwargs)
    new_ids = out_ids[:, batch.input_ids.shape[1]:]
    return tokenizer.batch_decode(new_ids, skip_special_tokens=True)


# ===========================================================================
# Model loading
# ===========================================================================
def load_base_with_ao(
    base_model_name: str,
    ao_lora_path: str,
    use_8bit: bool = False,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Load the base instruction-tuned model and attach Karvonen's AO LoRA.

    base_model_name: 'Qwen/Qwen3-8B' or 'google/gemma-2-9b-it'
    ao_lora_path:    e.g. 'adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B'
    use_8bit:        only needed if GPU memory is tight

    Unlike Karvonen's demo (which adds a dummy `default` adapter first, then
    loads the AO on top with `model.load_adapter`), we use PeftModel.from_pretrained
    directly. The dummy-adapter pattern in the demo exists so they can later
    swap in target-model adapters too; we only need the AO, and using
    PeftModel.from_pretrained gives a cleaner load with no MISSING warnings.
    """
    from peft import PeftModel

    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    print(f"[*] Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs = {"device_map": "auto", "torch_dtype": dtype}
    if use_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    print(f"[*] Loading base model: {base_model_name} (8-bit={use_8bit}, dtype={dtype})")
    base = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
    base.eval()

    print(f"[*] Attaching AO adapter: {ao_lora_path}")
    model = PeftModel.from_pretrained(
        base,
        ao_lora_path,
        adapter_name="ao",
        is_trainable=False,
    )
    model.set_adapter("ao")
    model.eval()
    print(f"[*] AO adapter active: ao")

    return model, tokenizer, device, "ao"
