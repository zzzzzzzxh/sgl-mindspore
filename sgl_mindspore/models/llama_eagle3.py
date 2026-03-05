# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

"""
EAGLE-3 draft model for Llama model

Adapted from
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama_eagle3.py
"""

import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import mindspore as ms
import torch
from mindspore import Tensor, dtype, jit, mint, mutable, nn, ops
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix

from sgl_mindspore.layers import (
    ColParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RMSNorm,
    VocabParallelEmbedding,
)
from sgl_mindspore.models.llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaMLP
from sgl_mindspore.utils import get_ms_dtype, tensor_torch2ms

logger = logging.getLogger(__name__)

LlamaConfig = None


class LlamaDecoderLayerEagle3(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)

        # override qkv
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * self.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
            param_dtype=config.param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.mlp = LlamaMLP(config, quant_config, prefix)

        self.hidden_norm = RMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
        )

    def construct(
        self,
        embeds: Tensor,
        hidden_states: Tensor,
        residual: Tensor,
        positions: Tensor,
        batch_valid_length: Tensor,
        is_prefill: bool,
        layer_idx: int,
        attn_mask: Tensor,
        q_seq_lens: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        out_cache_loc: Tensor,
        block_tables: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        hidden_states = mint.cat([embeds, hidden_states], dim=-1)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            batch_valid_length=batch_valid_length,
            is_prefill=is_prefill,
            layer_idx=layer_idx,
            attn_mask=attn_mask,
            q_seq_lens=q_seq_lens,
            key_cache=key_cache,
            value_cache=value_cache,
            out_cache_loc=out_cache_loc,
            block_tables=block_tables,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Fully Connected
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class LlamaModelEagle3(nn.Cell):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(config=config)

        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        self.midlayer = LlamaDecoderLayerEagle3(config, 0, quant_config, prefix)

        self.norm = RMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
        )

        # For tensor loading (comparison experiment)
        self.load_tensors = os.environ.get("MS_LOAD_TENSORS", "0") == "1"
        self.load_dir = os.environ.get("MS_LOAD_DIR", "/tmp/sglang_tensors")
        self.forward_count = 0
        self.loaded_tensors = {}

        if self.load_tensors:
            self._load_tensors_from_sglang()

    def _load_tensors_from_sglang(self):
        """Load tensors saved from sglang for comparison"""
        forward_dir = os.path.join(self.load_dir, "forward_0")
        inputs_dir = os.path.join(forward_dir, "inputs")

        if not os.path.exists(inputs_dir):
            print(f"[MS] Warning: No saved tensors found at {inputs_dir}")
            return

        # Load all input tensors
        tensor_files = {
            "input_ids": "input_ids.pt",
            "positions": "positions.pt",
            "embeds": "embeds.pt",
            "hidden_states_before_fc": "hidden_states_before_fc.pt",
            "hidden_states_after_fc": "hidden_states_after_fc.pt",
        }

        for key, filename in tensor_files.items():
            filepath = os.path.join(inputs_dir, filename)
            if os.path.exists(filepath):
                # Load torch tensor and convert to mindspore
                torch_tensor = torch.load(
                    filepath, map_location="cpu", weights_only=True
                )
                ms_tensor = tensor_torch2ms(torch_tensor)
                self.loaded_tensors[key] = ms_tensor
                print(
                    f"[MS] Loaded {key}: shape={ms_tensor.shape}, dtype={ms_tensor.dtype}"
                )
            else:
                print(f"[MS] Warning: {filename} not found, skipping")

        # Load forward_batch_info separately (it's a dict, not a tensor)
        batch_info_path = os.path.join(inputs_dir, "forward_batch_info.pt")
        if os.path.exists(batch_info_path):
            batch_info = torch.load(
                batch_info_path, map_location="cpu", weights_only=False
            )
            self.loaded_tensors["forward_batch_info"] = batch_info
            print(
                f"[MS] Loaded forward_batch_info: keys={list(batch_info.keys()) if isinstance(batch_info, dict) else 'not a dict'}"
            )

        print(f"[MS] Total loaded {len(self.loaded_tensors)} tensors from sglang")

    @jit
    def construct(
        self,
        input_ids,
        hidden_states=None,
        position_ids=None,
        attention_mask=None,
        batch_valid_length=None,
        is_prefill=True,
        q_seq_lens=None,
        key_cache=None,
        value_cache=None,
        out_cache_loc=None,
        block_tables=None,
    ):
        # Force use loaded tensors if enabled
        if self.load_tensors and len(self.loaded_tensors) > 0:
            print(
                f"[MS] Using loaded tensors from sglang (forward {self.forward_count})"
            )

            # Override input_ids
            if "input_ids" in self.loaded_tensors:
                input_ids = self.loaded_tensors["input_ids"]
                print(f"[MS] Overriding input_ids: shape={input_ids.shape}")

            # Override positions
            if "positions" in self.loaded_tensors:
                position_ids = self.loaded_tensors["positions"]
                print(f"[MS] Overriding position_ids: shape={position_ids.shape}")

            # Override hidden_states (before fc projection)
            if "hidden_states_before_fc" in self.loaded_tensors:
                hidden_states = self.loaded_tensors["hidden_states_before_fc"]
                print(
                    f"[MS] Overriding hidden_states_before_fc: shape={hidden_states.shape}"
                )

        embeds = self.embed_tokens(input_ids)

        # Override embeds if loaded
        if self.load_tensors and "embeds" in self.loaded_tensors:
            embeds = self.loaded_tensors["embeds"]
            print(f"[MS] Overriding embeds: shape={embeds.shape}")

        residual = None

        # Apply fc projection if needed
        if (
            hidden_states is not None
            and hidden_states.shape[-1] != self.config.hidden_size
        ):
            # Note: fc is in LlamaForCausalLMEagle3, so we skip here
            # The fc projection will be handled in LlamaForCausalLMEagle3.construct
            pass

        # Override hidden_states after fc if loaded
        if self.load_tensors and "hidden_states_after_fc" in self.loaded_tensors:
            hidden_states = self.loaded_tensors["hidden_states_after_fc"]
            print(
                f"[MS] Overriding hidden_states_after_fc: shape={hidden_states.shape}"
            )

        hidden_states, residual = self.midlayer(
            embeds=embeds,
            hidden_states=hidden_states,
            residual=residual,
            positions=position_ids,
            batch_valid_length=batch_valid_length,
            is_prefill=is_prefill,
            layer_idx=0,
            attn_mask=attention_mask,
            q_seq_lens=q_seq_lens,
            key_cache=key_cache[0],
            value_cache=value_cache[0],
            out_cache_loc=out_cache_loc,
            block_tables=block_tables,
        )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

        # Increment forward count
        if self.load_tensors:
            self.forward_count += 1

        # For draft decode, we capture the hidden state before norm
        return hidden_states_to_logits, [hidden_states_to_aux]


class LlamaForCausalLMEagle3(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.config = config
        self.quant_config = quant_config

        if self.config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 currently only supports 1 layer")

        if self.config.dtype:
            param_dtype = get_ms_dtype(self.config.dtype)
        else:
            param_dtype = ms.dtype.bfloat16
        setattr(self.config, "param_dtype", param_dtype)

        self.model = LlamaModelEagle3(config, quant_config, add_prefix("model", prefix))

        self.load_lm_head_from_target = False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            if config.draft_vocab_size is None:
                self.load_lm_head_from_target = True
                config.draft_vocab_size = config.vocab_size
            self.lm_head = ColParallelLinear(
                input_size=self.config.hidden_size,
                output_size=self.config.draft_vocab_size,
                bias=False,
                param_dtype=self.config.param_dtype,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        self.lm_head.construct = jit(self.lm_head.construct)
        self.capture_aux_hidden_states = True
        self.hot_token_id = None
        self.fc = ReplicatedLinear(
            input_size=config.hidden_size * 3,
            output_size=config.hidden_size,
            bias=getattr(config, "bias", False),
            param_dtype=config.param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("fc", prefix),
        )

        # For output comparison
        self.save_outputs = os.environ.get("MS_SAVE_OUTPUTS", "0") == "1"
        self.load_dir = os.environ.get("MS_LOAD_DIR", "/tmp/sglang_tensors")
        self.forward_count = 0

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = self.parameters_dict()
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", "gate"),
            (".gate_up_proj", ".up_proj", "up"),
        ]

        for name, loaded_weight in weights:
            if "d2t" in name:
                # d2t stores diffs between draft id and target id
                loaded_weight = tensor_torch2ms(loaded_weight).move_to("Ascend")
                self.hot_token_id = loaded_weight + mint.arange(loaded_weight.shape[0])
                continue

            if "t2d" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param_name = f"model.{name}" if name not in params_dict else name
                if param_name in params_dict:
                    param = params_dict[param_name]
                    assert hasattr(param, "weight_load")
                    weight_load = getattr(param, "weight_load")
                    weight_load(param, loaded_weight, shard_id)
                    param.set_data(param.move_to("Ascend"))
                    break
            else:
                param_name = name if name in params_dict else f"model.{name}"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    if hasattr(param, "weight_load"):
                        weight_load = getattr(
                            param, "weight_load", default_weight_loader
                        )
                        weight_load(param, loaded_weight)
                        param.set_data(param.move_to("Ascend"))
                    else:
                        param.set_data(tensor_torch2ms(loaded_weight).move_to("Ascend"))
                    # Make sure the weight is loaded on device, so the kv cache calculation is correct.

    def set_model_inputs(self, is_prefill):
        dyn_input_ids = Tensor(shape=[None], dtype=dtype.int32)
        dyn_position_ids = Tensor(shape=[None], dtype=dtype.int64)

        head_size = self.config.head_dim
        # use pa, if use ifa, the shape should (None, None, head_size)
        kv_cache_shape = (None, None, None, head_size)

        kv_cache_dtype = self.config.param_dtype

        num_layers = self.config.num_hidden_layers

        dyn_key_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
        dyn_value_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
        dyn_key_caches = mutable([dyn_key_cache for _ in range(num_layers)])
        dyn_value_caches = mutable([dyn_value_cache for _ in range(num_layers)])

        dyn_out_cache_loc = Tensor(
            shape=[
                None,
            ],
            dtype=dtype.int32,
        )
        dynamic_attention_mask = Tensor(
            shape=[None, None], dtype=self.config.param_dtype
        )
        dyn_batch_valid_length = Tensor(
            shape=[
                None,
            ],
            dtype=dtype.int32,
        )
        dyn_q_seq_lens = Tensor(
            shape=[
                None,
            ],
            dtype=dtype.int32,
        )
        dynamic_hidden_states = Tensor(
            shape=[None, None], dtype=self.config.param_dtype
        )
        dyn_block_tables = Tensor(shape=[None, None], dtype=dtype.int32)
        # dyn_intermediate_tensors = None
        # dyn_inputs_embeds = None
        self.model.set_inputs(
            input_ids=dyn_input_ids,
            position_ids=dyn_position_ids,
            hidden_states=dynamic_hidden_states,
            attention_mask=dynamic_attention_mask,
            batch_valid_length=dyn_batch_valid_length,
            is_prefill=is_prefill,
            q_seq_lens=dyn_q_seq_lens,
            key_cache=dyn_key_caches,
            value_cache=dyn_value_caches,
            out_cache_loc=dyn_out_cache_loc,
            block_tables=dyn_block_tables,
        )
        self.lm_head.set_inputs(dynamic_hidden_states)

    def construct(self, **model_inputs) -> Tensor:
        q_seq_lens = model_inputs["q_seq_lens"]
        is_prefill = model_inputs["is_prefill"]
        if "capture_hidden_mode" in model_inputs:
            capture_hidden_mode = model_inputs.pop("capture_hidden_mode")
        if "forward_mode" in model_inputs:
            forward_mode = model_inputs.pop("forward_mode")
        else:
            forward_mode = None

        if self.prev_prefill != is_prefill:
            self.set_model_inputs(is_prefill)
        self.prev_prefill = is_prefill

        if is_prefill:
            self.model.phase = "prefill"
        else:
            self.model.phase = "increment"

        hidden_states = model_inputs["hidden_states"]
        if hidden_states.shape[-1] != self.config.hidden_size:
            hidden_states = self.fc(hidden_states)
        model_inputs["hidden_states"] = hidden_states
        hidden_states = self.model(**model_inputs)

        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
            if capture_hidden_mode.need_capture():
                if capture_hidden_mode.is_full():
                    aux_hidden_states = mint.cat(aux_hidden_states, dim=-1)
                elif capture_hidden_mode.is_last():
                    aux_hidden_states = mint.cat(aux_hidden_states, dim=-1)
                    aux_hidden_states = mint.index_select(
                        aux_hidden_states, 0, mint.cumsum(q_seq_lens, 0) - 1
                    )
                else:
                    assert False, "Unsupported capture hidden mode"

        # TODO: In pure decode scenarios, cumsum and gather operations will be redundant .

        # Save outputs for comparison if enabled (only first forward pass)
        # Save BEFORE index_select to match SGLang's hidden_states_to_logits shape
        if self.save_outputs and self.forward_count == 0:
            outputs_dir = os.path.join(
                self.load_dir, f"forward_{self.forward_count}", "outputs_ms"
            )
            os.makedirs(outputs_dir, exist_ok=True)

            # Convert to torch for saving
            # Note: bfloat16 numpy arrays can't be directly converted via from_numpy
            # So we convert to float32 first, then save with original dtype info
            import numpy as np

            # Save hidden_states BEFORE index_select (same as SGLang's hidden_states_to_logits)
            hidden_states_np = hidden_states.asnumpy()
            hidden_states_dtype = str(hidden_states_np.dtype)
            is_hs_bfloat16 = (
                hidden_states_dtype == "bfloat16" or "bfloat16" in hidden_states_dtype
            )
            if is_hs_bfloat16:
                hidden_states_np_float = hidden_states_np.astype(np.float32)
                hidden_states_torch = torch.from_numpy(hidden_states_np_float).to(
                    torch.bfloat16
                )
            else:
                hidden_states_torch = torch.from_numpy(hidden_states_np)
            torch.save(
                hidden_states_torch,
                os.path.join(outputs_dir, "hidden_states_to_logits.pt"),
            )
            print(
                f"[MS] Saved hidden_states_to_logits: shape={hidden_states.shape}, dtype={hidden_states_dtype}"
            )

            if self.capture_aux_hidden_states and aux_hidden_states is not None:
                aux_np = aux_hidden_states.asnumpy()
                aux_dtype = str(aux_np.dtype)
                is_aux_bfloat16 = aux_dtype == "bfloat16" or "bfloat16" in aux_dtype
                if is_aux_bfloat16:
                    aux_np_float = aux_np.astype(np.float32)
                    aux_torch = torch.from_numpy(aux_np_float).to(torch.bfloat16)
                else:
                    aux_torch = torch.from_numpy(aux_np)
                torch.save(aux_torch, os.path.join(outputs_dir, "aux_hidden_states.pt"))
                print(
                    f"[MS] Saved aux_hidden_states: shape={aux_hidden_states.shape}, dtype={aux_dtype}"
                )

            self.forward_count += 1
            print(f"[MS] Completed output saving for forward pass {self.forward_count}")

        q_seq_lens = mint.cumsum(q_seq_lens, 0)
        if not (forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2()):
            # In target verify mode, all tokens' logits are needed.
            hidden_states = mint.index_select(hidden_states, 0, q_seq_lens - 1)

        logits = self.lm_head(hidden_states)
        if self.tp_size:
            logits = self.all_gather(logits)
        logits = mint.reshape(logits, (-1, logits.shape[-1]))

        if self.capture_aux_hidden_states:
            return logits, aux_hidden_states
        else:
            return logits

    def get_hot_token_id(self):
        return self.hot_token_id

    def prepare_inputs(self, forward_batch: ForwardBatch, model_inputs: Dict[str, Any]):
        if forward_batch.spec_info:
            model_inputs["hidden_states"] = tensor_torch2ms(
                forward_batch.spec_info.hidden_states
            )
        if forward_batch.capture_hidden_mode:
            model_inputs["capture_hidden_mode"] = forward_batch.capture_hidden_mode
        return model_inputs


EntryClass = [LlamaForCausalLMEagle3]
