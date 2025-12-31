# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

"""
EAGLE-3 draft model for Qwen3 model

Adapted from
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama_eagle3.py
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import mindspore as ms
import torch
from mindspore import Tensor, dtype, jit, mint, mutable, nn, ops
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import add_prefix

from sgl_mindspore.layers import (
    ColParallelLinear,
    QKVParallelLinear,
    RMSNorm,
    VocabParallelEmbedding,
)
from sgl_mindspore.models.qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM, Qwen3MLP
from sgl_mindspore.utils import get_ms_dtype, tensor_torch2ms

logger = logging.getLogger(__name__)

Qwen3Config = None


class Qwen3DecoderLayerEagle3(Qwen3DecoderLayer):
    def __init__(
        self,
        config: Qwen3Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)

        # override qkv
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * self.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.num_heads,
            self.self_attn.num_kv_heads,
            bias=False,
            param_dtype=config.param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.mlp = Qwen3MLP(config, quant_config, prefix)

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


class Qwen3ModelEagle3(nn.Cell):
    def __init__(
        self,
        config: Qwen3Config,
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

        self.fc = ColParallelLinear(
            input_size=self.hidden_size_in * 3,
            output_size=config.hidden_size,
            bias=getattr(config, "bias", False),
            param_dtype=config.param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("fc", prefix),
        )

        self.midlayer = Qwen3DecoderLayerEagle3(config, 0, quant_config, prefix)

        self.norm = RMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
        )

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
        print(f"[Qwen3ModelEagle3] construct called. is_prefill={is_prefill}")
        if hidden_states is not None:
            print(f"[Qwen3ModelEagle3] hidden_states shape: {hidden_states.shape}")
        else:
            print("[Qwen3ModelEagle3] hidden_states is None")

        embeds = self.embed_tokens(input_ids)
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        # idle batch
        if hidden_states.shape[0] == 0:
            return hidden_states, [hidden_states]

        print(
            f"[Qwen3ModelEagle3] hidden_states shape: {hidden_states.shape}, embeds shape: {embeds.shape}"
        )
        print(f"[Qwen3ModelEagle3] is_prefill: {is_prefill}")

        residual = None
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

        # For draft decode, we capture the hidden state before norm
        return hidden_states_to_logits, [hidden_states_to_aux]


class LlamaForCausalLMEagle3(Qwen3ForCausalLM):
    def __init__(
        self,
        config: Qwen3Config,
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

        self.model = Qwen3ModelEagle3(config, quant_config, add_prefix("model", prefix))

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

        self.capture_aux_hidden_states = True
        self.hot_token_id = None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        param_dict = self.parameters_dict()
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", "gate"),
            (".gate_up_proj", ".up_proj", "up"),
        ]

        direct_params_mapping = [
            (".self_attn.o_proj", ".self_attn.o_proj"),
            (".mlp.down_proj", ".mlp.down_proj"),
            (".hidden_norm", ".hidden_norm"),
            (".input_layernorm", ".input_layernorm"),
            (".post_attention_layernorm", ".post_attention_layernorm"),
            (".norm", ".norm"),
            (".fc", ".fc"),
        ]

        for name, weight in weights:
            if "d2t" in name:
                # d2t stores diffs between draft id and target id
                weight = tensor_torch2ms(weight).move_to("Ascend")
                self.hot_token_id = weight + mint.arange(weight.shape[0])
                continue

            if "t2d" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if not mapped_name.startswith("model."):
                    mapped_name = "model." + mapped_name
                if mapped_name in param_dict:
                    param = param_dict[mapped_name]
                    assert hasattr(param, "weight_load")
                    weight_load = getattr(param, "weight_load")
                    weight_load(param, weight, shard_id)
                    param.set_data(param.move_to("Ascend"))
                    break
            else:
                if name in param_dict:
                    param = param_dict[name]
                    if hasattr(param, "weight_load"):
                        weight_load = getattr(param, "weight_load")
                        weight_load(param, weight)
                        param.set_data(param.move_to("Ascend"))
                    else:
                        param.set_data(tensor_torch2ms(weight).move_to("Ascend"))
                    # Make sure the weight is loaded on device, so the kv cache calculation is correct.

    def get_hot_token_id(self):
        return self.hot_token_id

    def prepare_inputs(self, forward_batch: ForwardBatch, model_inputs: Dict[str, Any]):
        if forward_batch.spec_info:
            print(
                f"[LlamaForCausalLMEagle3] prepare_inputs: Found spec_info. hidden_states shape: {forward_batch.spec_info.hidden_states.shape}"
            )
            model_inputs["hidden_states"] = tensor_torch2ms(
                forward_batch.spec_info.hidden_states
            )
        else:
            print("[LlamaForCausalLMEagle3] prepare_inputs: No spec_info found.")

        # Handle the case where hidden_states might be missing in some spec decoding steps
        # We need to ensure hidden_states is always present for MindSpore graph compilation
        # even if it's a dummy tensor when not strictly needed by logic but required by signature.
        # However, for Eagle, it seems hidden_states IS required.
        # If spec_info is None, we must provide a dummy hidden_states to match the set_inputs signature.
        if "hidden_states" not in model_inputs:
            target_hidden_size = self.config.hidden_size
            if hasattr(self.config, "target_hidden_size"):
                target_hidden_size = self.config.target_hidden_size
            # Create a dummy tensor with correct shape (0, hidden_size) or (1, hidden_size) depending on batch
            # Using (0, hidden_size) might be safer for concatenation logic if any
            # But here we probably just need a placeholder that matches the dtype and shape rank
            model_inputs["hidden_states"] = Tensor(
                shape=(0, target_hidden_size), dtype=self.config.param_dtype
            )

        if forward_batch.capture_hidden_mode:
            model_inputs["capture_hidden_mode"] = forward_batch.capture_hidden_mode
        return model_inputs

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
        dyn_block_tables = Tensor(shape=[None, None], dtype=dtype.int32)

        target_hidden_size = self.config.hidden_size
        if hasattr(self.config, "target_hidden_size"):
            target_hidden_size = self.config.target_hidden_size

        dyn_hidden_states = Tensor(
            shape=[None, target_hidden_size], dtype=self.config.param_dtype
        )

        # Explicitly define the inputs order for set_inputs to match construct signature
        self.set_inputs(
            input_ids=dyn_input_ids,
            position_ids=dyn_position_ids,
            attention_mask=dynamic_attention_mask,
            batch_valid_length=dyn_batch_valid_length,
            is_prefill=is_prefill,
            q_seq_lens=dyn_q_seq_lens,
            key_cache=dyn_key_caches,
            value_cache=dyn_value_caches,
            out_cache_loc=dyn_out_cache_loc,
            block_tables=dyn_block_tables,
            hidden_states=dyn_hidden_states,
        )

    def construct(
        self,
        input_ids,
        position_ids,
        attention_mask,
        batch_valid_length,
        is_prefill,
        q_seq_lens,
        key_cache,
        value_cache,
        out_cache_loc,
        block_tables,
        hidden_states=None,
        capture_hidden_mode=None,
        forward_mode=None,
    ) -> Tensor:
        """
        Explicit construct method to ensure parameter alignment with set_inputs.
        """
        print(f"[LlamaForCausalLMEagle3] construct called. is_prefill={is_prefill}")
        if hidden_states is not None:
            print(
                f"[LlamaForCausalLMEagle3] hidden_states shape: {hidden_states.shape}"
            )
            # Print first few elements to check for corruption
            # if hidden_states.shape[0] > 0:
            #    print(f"[LlamaForCausalLMEagle3] hidden_states[0, :5]: {hidden_states[0, :5]}")

        if self.prev_prefill != is_prefill:
            self.set_model_inputs(is_prefill)
        self.prev_prefill = is_prefill

        if is_prefill:
            self.model.phase = "prefill"
        else:
            self.model.phase = "increment"

        print(
            f"DEBUG: hidden_states passed to self.model: {hidden_states.shape if hidden_states is not None else 'None'}"
        )

        # Call the underlying model
        hidden_states_out, aux_hidden_states = self.model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            batch_valid_length=batch_valid_length,
            is_prefill=is_prefill,
            q_seq_lens=q_seq_lens,
            key_cache=key_cache,
            value_cache=value_cache,
            out_cache_loc=out_cache_loc,
            block_tables=block_tables,
        )

        print(f"DEBUG: hidden_states_out shape: {hidden_states_out.shape}")
        # print(f"DEBUG: aux_hidden_states type: {type(aux_hidden_states)}, len: {len(aux_hidden_states) if isinstance(aux_hidden_states, (list, tuple)) else 'N/A'}")

        if self.capture_aux_hidden_states:
            # Logic from Qwen3ForCausalLM.construct for handling aux_hidden_states
            if capture_hidden_mode is not None and capture_hidden_mode.need_capture():
                if capture_hidden_mode.is_full():
                    aux_hidden_states = mint.cat(aux_hidden_states, dim=-1)
                elif capture_hidden_mode.is_last():
                    aux_hidden_states = mint.cat(aux_hidden_states, dim=-1)
                    aux_hidden_states = mint.index_select(
                        aux_hidden_states, 0, mint.cumsum(q_seq_lens, 0) - 1
                    )
                else:
                    assert False, "Unsupported capture hidden mode"

        # Logic from Qwen3ForCausalLM.construct for logits
        # TODO: In pure decode scenarios, cumsum and gather operations will be redundant .
        q_seq_lens_cumsum = mint.cumsum(q_seq_lens, 0)

        print(f"DEBUG: q_seq_lens: {q_seq_lens}")
        # print(f"DEBUG: forward_mode: {forward_mode}")

        if forward_mode is not None and not forward_mode.is_target_verify():
            hidden_states_out = mint.index_select(
                hidden_states_out, 0, q_seq_lens_cumsum - 1
            )
        elif forward_mode is None:
            # Fallback if forward_mode is missing (e.g. during simple tests)
            # Assume we need last token logic for generation
            hidden_states_out = mint.index_select(
                hidden_states_out, 0, q_seq_lens_cumsum - 1
            )

        logits = self.lm_head(hidden_states_out)
        if self.tp_size:
            logits = self.all_gather(logits)
        logits = mint.reshape(logits, (-1, logits.shape[-1]))

        if self.capture_aux_hidden_states:
            return logits, aux_hidden_states
        else:
            return logits


EntryClass = [LlamaForCausalLMEagle3]
