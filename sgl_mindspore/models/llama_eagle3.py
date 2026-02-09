# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

"""
EAGLE-3 draft model for Llama model

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

# Model context for static graph compilation
_eagle3_model_context = {"is_prefill": True}


def set_model_context(key, value):
    """Set model context variable for static graph compilation."""
    global _eagle3_model_context
    _eagle3_model_context[key] = value


def get_model_context(key):
    """Get model context variable for static graph compilation."""
    return _eagle3_model_context[key]


class LlamaAttentionEagle3(nn.Cell):
    """EAGLE-3 specialized attention that uses model context instead of is_prefill parameter.

    This avoids control flow splitting in static graph compilation by getting is_prefill
    from model context rather than as a function parameter.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // self.num_heads
        self.q_size = self.head_dim * self.num_heads
        self.kv_size = self.head_dim * self.num_kv_heads
        self.scaling = float(self.head_dim**-0.5)
        self.rope_theta = int(config.rope_theta)
        self.param_dtype = config.param_dtype
        self.max_position = config.max_position_embeddings
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling["rope_type"]
            self.rope_factor = config.rope_scaling["factor"]
            self.rope_max_position_embeddings = config.rope_scaling[
                "original_max_position_embeddings"
            ]
        else:
            self.rope_type = "default_rope"

        from sgl_mindspore.layers import (
            BaseRotaryEmbedding,
            MsNativeAttnBackend,
            QKVParallelLinear,
            RowParallelLinear,
            YaRNScalingRotaryEmbedding,
        )

        self.attn = MsNativeAttnBackend(
            self.num_heads,
            self.head_dim,
            self.num_kv_heads,
        )

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=config.attention_bias,
            param_dtype=self.param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            param_dtype=self.param_dtype,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = None
        if self.rope_type == "yarn":
            self.rotary_emb = YaRNScalingRotaryEmbedding(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position_embeddings=self.rope_max_position_embeddings,
                base=self.rope_theta,
                is_neox_style=True,
                scaling_factor=self.rope_factor,
                dtype=self.param_dtype,
            )
        else:
            self.rotary_emb = BaseRotaryEmbedding(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position_embeddings=self.max_position,
                base=self.rope_theta,
                dtype=self.param_dtype,
            )

    def construct(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        batch_valid_length: Tensor,
        layer_idx: int,
        attn_mask: Tensor,
        q_seq_lens: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        out_cache_loc: Tensor,
        block_tables: Tensor,
    ) -> Tensor:
        """Construct method that gets is_prefill from model context instead of parameter."""
        token_lens, hidden_dim = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [
                self.q_size,
                self.kv_size,
                self.kv_size,
            ],
            dim=-1,
        )

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Get is_prefill from model context to avoid control flow splitting
        is_prefill = get_model_context("is_prefill")

        q, k = self.rotary_emb(
            positions,
            q,
            k,
            batch_valid_length=batch_valid_length,
            is_prefill=is_prefill,
        )

        key_out = self.attn(
            k,
            v,
            key_cache=key_cache,
            value_cache=value_cache,
            out_cache_loc=out_cache_loc,
        )
        q = ops.depend(q, key_out)

        if is_prefill:
            attn_output = self.attn.extend(
                q,
                k,
                v,
                attn_mask,
                None,
                None,
                None,
                batch_valid_length,
                batch_valid_length,
            )
        else:
            attn_output = self.attn.decode(
                q,
                batch_valid_length,
                attn_mask,
                q_seq_lens,
                key_cache,
                value_cache,
                block_tables,
            )

        output = self.o_proj(attn_output).view(token_lens, -1)
        return output


class LlamaDecoderLayerEagle3(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)

        # Replace self_attn with LlamaAttentionEagle3
        self.self_attn = LlamaAttentionEagle3(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
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

        self.fc = ReplicatedLinear(
            input_size=self.hidden_size_in * 3,
            output_size=config.hidden_size,
            bias=getattr(config, "bias", False),
            param_dtype=config.param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("fc", prefix),
        )

        self.midlayer = LlamaDecoderLayerEagle3(config, 0, quant_config, prefix)

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
        q_seq_lens=None,
        key_cache=None,
        value_cache=None,
        out_cache_loc=None,
        block_tables=None,
    ):
        embeds = self.embed_tokens(input_ids)
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        # idle batch
        if hidden_states.shape[0] == 0:
            return hidden_states, [hidden_states]

        residual = None
        hidden_states, residual = self.midlayer(
            embeds=embeds,
            hidden_states=hidden_states,
            residual=residual,
            positions=position_ids,
            batch_valid_length=batch_valid_length,
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

        # Initialize static graph compilation for EAGLE3
        self.prefill_graph = None
        self.decode_graph = None
        self.set_flags = False

    def construct(self, **model_inputs) -> Tensor:
        """Override construct to use exec_model for separate prefill/decode graphs."""
        # Extract parameters needed for post-processing
        q_seq_lens = model_inputs.get("q_seq_lens")
        is_prefill = model_inputs.get("is_prefill", True)
        capture_hidden_mode = None
        if "capture_hidden_mode" in model_inputs:
            capture_hidden_mode = model_inputs.pop("capture_hidden_mode")
        forward_mode = None
        if "forward_mode" in model_inputs:
            forward_mode = model_inputs.pop("forward_mode")

        # Call exec_model for the core forward pass
        model_output = self.exec_model(**model_inputs)

        # Handle EAGLE3 output format
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = model_output
            # Post-process aux hidden states
            if capture_hidden_mode and capture_hidden_mode.need_capture():
                if capture_hidden_mode.is_full():
                    aux_hidden_states = mint.cat(aux_hidden_states, dim=-1)
                elif capture_hidden_mode.is_last():
                    aux_hidden_states = mint.cat(aux_hidden_states, dim=-1)
                    aux_hidden_states = mint.index_select(
                        aux_hidden_states, 0, mint.cumsum(q_seq_lens, 0) - 1
                    )
                else:
                    assert False, "Unsupported capture hidden mode"
        else:
            hidden_states = model_output
            aux_hidden_states = None

        # Select last token for logits
        if q_seq_lens is not None:
            q_seq_lens = mint.cumsum(q_seq_lens, 0)
            if forward_mode is None or not (
                forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2()
            ):
                # In target verify mode, all tokens' logits are needed.
                hidden_states = mint.index_select(hidden_states, 0, q_seq_lens - 1)

        # Compute logits
        logits = self.lm_head(hidden_states)
        if self.tp_size:
            logits = self.all_gather(logits)
        logits = mint.reshape(logits, (-1, logits.shape[-1]))

        # Return with aux hidden states if captured
        if self.capture_aux_hidden_states:
            return logits, aux_hidden_states
        else:
            return logits

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

    def get_hot_token_id(self):
        return self.hot_token_id

    def set_model_inputs(self, is_prefill):
        dyn_input_ids = Tensor(shape=[None], dtype=dtype.int32)
        dyn_position_ids = Tensor(shape=[None], dtype=dtype.int64)

        head_size = self.config.head_dim
        # use pa, if use ifa, shape should (None, None, head_size)
        kv_cache_shape = (None, None, None, head_size)

        kv_cache_dtype = self.config.param_dtype

        # Eagle3 has only 1 layer
        num_layers = 1

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

        # For Eagle3, hidden_states can have different batch sizes from embeds
        # Use dynamic shapes for hidden_states (which is passed in)
        dyn_hidden_states = Tensor(shape=[None, None], dtype=self.config.param_dtype)

        # Set inputs for main model
        # Note: embeds is generated internally from input_ids in construct,
        # so we don't need to declare it separately in set_inputs
        self.model.set_inputs(
            input_ids=dyn_input_ids,
            hidden_states=dyn_hidden_states,
            position_ids=dyn_position_ids,
            attention_mask=dynamic_attention_mask,
            batch_valid_length=dyn_batch_valid_length,
            q_seq_lens=dyn_q_seq_lens,
            key_cache=dyn_key_caches,
            value_cache=dyn_value_caches,
            out_cache_loc=dyn_out_cache_loc,
            block_tables=dyn_block_tables,
        )

    def prepare_inputs(self, forward_batch: ForwardBatch, model_inputs: Dict[str, Any]):
        if forward_batch.spec_info:
            model_inputs["hidden_states"] = tensor_torch2ms(
                forward_batch.spec_info.hidden_states
            )
        if forward_batch.capture_hidden_mode:
            model_inputs["capture_hidden_mode"] = forward_batch.capture_hidden_mode
        return model_inputs

    def exec_model(self, **model_inputs):
        """Execute model with separate prefill and decode graphs."""
        is_prefill = model_inputs.get("is_prefill", True)

        # Set model inputs for first compilation
        if not self.set_flags:
            self.set_model_inputs(is_prefill)
            self.set_flags = True

        # Set model context for is_prefill
        set_model_context("is_prefill", is_prefill)

        # Eager mode (for debugging)
        # Uncomment to enable eager mode
        # return self.model(**model_inputs)

        # Remove is_prefill from model_inputs before passing to compiled graph
        # The compiled graph's construct method doesn't have is_prefill parameter
        model_inputs_for_graph = {
            k: v for k, v in model_inputs.items() if k != "is_prefill"
        }

        # Graph mode: separate compilation for prefill and decode
        if is_prefill:
            self.model.phase = "prefill"
            if self.prefill_graph is None:
                self.model._set_jit_graph_name("prefill")
                self.prefill_graph = ms.jit(function=self.model, jit_level="O0")
            model_output = self.prefill_graph(**model_inputs_for_graph)
        else:
            self.model.phase = "increment"
            if self.decode_graph is None:
                self.model._set_jit_graph_name("decode")
                self.decode_graph = ms.jit(function=self.model, jit_level="O0")
            model_output = self.decode_graph(**model_inputs_for_graph)

        return model_output


EntryClass = [LlamaForCausalLMEagle3]
