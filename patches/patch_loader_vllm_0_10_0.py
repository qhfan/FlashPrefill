import os
import torch
import types
# from vllm_models.ops import flash_sparse_infer_out
from ops import flash_prefill_varlen_func, flash_prefill
# from vllm_models.ops.flashparseinfernotune import flash_sparse_infer_out
import vllm.model_executor.model_loader as vllm_loader
from vllm.attention.backends.flash_attn import FlashAttentionMetadata, _get_causal_option, _get_query_key_seq_metadata
from vllm.attention.backends.abstract import (AttentionLayer,
                                              AttentionType)
from vllm.attention.utils.fa_utils import (flash_attn_supports_fp8)
from vllm import _custom_ops as ops

from vllm.attention.backends.utils import (
    get_num_prefill_decode_query_kv_tokens,
    get_seq_len_block_table_args
)
from vllm.vllm_flash_attn import (flash_attn_varlen_func,
                                  flash_attn_with_kvcache)


from typing import Optional


ATTENTION_CONFIG_EXAMPLE = {
    "default": {},
    "flash": {},
    "flashprefill_llama": {
        "block_size": 128,
        "attention_sink": 2,
        "window_size": 4,
        "alpha": 0.18,
        "last_n_full_block": 2,
        # "min_budget": 4
    },
    "flashprefill_qwen2": {
        "block_size": 128,
        "attention_sink": 2,
        "window_size": 4,
        "alpha": 0.08,
        "last_n_full_block": 2,
        # "min_budget": 4
    },
    "flashprefill_qwen3moe": {
        "block_size": 128,
        "attention_sink": 2,
        "window_size": 4,
        "alpha": 0.12,
        "last_n_full_block": 2,
        # "min_budget": 4
    }
}

# use qwen2_5 as example
cfg = ATTENTION_CONFIG_EXAMPLE["flashprefill_qwen2"]

def apply_patch_to_worker(worker=None):

    try:
        import vllm.attention.backends.flash_attn as fa_backend
    except ImportError:
        import vllm.model_executor.layers.attention.backends.flash_attn as fa_backend


    rank = str(getattr(worker, "rank", os.environ.get("RANK", "0")))
    cache_dir = f"/tmp/triton_cache_rank_{rank}"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = cache_dir

    print(cache_dir)

    original_forward = fa_backend.FlashAttentionImpl.forward

    def flash_sparse_infer_forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            output: shape = [num_tokens, num_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
        NOTE: It in-place updates the output tensor.
        NOTE: FP8 quantization, flash-attn expect the size of
            {q,k,v}_descale to be (num_sequences, num_kv_heads).
            We use torch's .expand() to avoid duplicating values
        """


        assert output is not None, "Output tensor must be provided."

        if output_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for FlashAttentionImpl")

        # NOTE(woosuk): FlashAttention2 does not support FP8 KV cache.
        if not flash_attn_supports_fp8() or output.dtype != torch.bfloat16:
            assert (
                layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0), (
                    "key/v_scale is only supported in FlashAttention 3 with "
                    "base dtype bfloat16")

        attn_type = self.attn_type
        if (attn_type == AttentionType.ENCODER
                and (not attn_metadata.is_all_encoder_attn_metadata_set)):
            raise AttributeError("Encoder attention requires setting "
                                "encoder metadata attributes.")
        elif (attn_type == AttentionType.ENCODER_DECODER
            and (not attn_metadata.is_all_cross_attn_metadata_set)):
            raise AttributeError("Encoder/decoder cross-attention "
                                "requires setting cross-attention "
                                "metadata attributes.")

        kv_cache_dtype: str = self.kv_cache_dtype
        softmax_scale: float = self.scale
        window_size = self.sliding_window
        alibi_slopes: Optional[torch.Tensor] = self.alibi_slopes
        logits_soft_cap: Optional[float] = self.logits_soft_cap
        fp8_attention = kv_cache_dtype.startswith("fp8")

        if fp8_attention and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support FP8 kv-cache on this device.")

        if kv_cache.numel() > 0:
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]
            # We skip updating the KV cache under two conditions:
            #  a. When the Attention Type is ENCODER. In this phase, we compute
            #     only the encoder attention without updating the cache.
            #  b. When both Key and Value are None. This occurs during
            #     cross-attention computation in the decoding phase, where the
            #     KV cache is already populated with the cross-attention
            #     tensor. Thus, we skip cache updates during this time.
            if (attn_type != AttentionType.ENCODER) and (key is not None) and (
                    value is not None):
                if attn_type == AttentionType.ENCODER_DECODER:
                    # Update cross-attention KV cache (prefill-only)
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    # Update self-attention KV cache (prefill/decode)
                    updated_slot_mapping = attn_metadata.slot_mapping

                # Reshape the input keys and values and store them in the cache.
                # If kv_cache is not provided, the new key and value tensors are
                # not cached. This happens during the initial memory
                # profiling run.
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    key,
                    value,
                    kv_cache[0],
                    kv_cache[1],
                    updated_slot_mapping.flatten(),  # type: ignore[union-attr]
                    kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )

                if fp8_attention:
                    kv_cache = kv_cache.view(torch.float8_e4m3fn)
                    key_cache = key_cache.view(torch.float8_e4m3fn)
                    value_cache = value_cache.view(torch.float8_e4m3fn)

        if fp8_attention:
            num_tokens, num_heads, head_size = query.shape
            query, _ = ops.scaled_fp8_quant(
                query.reshape(
                    (num_tokens, num_heads * head_size)).contiguous(),
                layer._q_scale)
            query = query.reshape((num_tokens, num_heads, head_size))

        (num_prefill_query_tokens, num_prefill_kv_tokens,
        num_decode_query_tokens) = \
            get_num_prefill_decode_query_kv_tokens(attn_metadata, attn_type)
        decode_query = query[num_prefill_query_tokens:]
        decode_output = output[num_prefill_query_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_query_tokens]
        prefill_output = output[:num_prefill_query_tokens]
        assert query.shape[0] == num_prefill_query_tokens
        assert decode_query.shape[0] == num_decode_query_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if (kv_cache.numel() == 0 or prefill_meta.block_tables is None
                    or prefill_meta.block_tables.numel() == 0):
                # normal attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                q_seq_start_loc, q_seq_len, k_seq_start_loc, k_seq_len = \
                    _get_query_key_seq_metadata(prefill_meta, True, attn_type)

                key = key[:num_prefill_kv_tokens]
                value = value[:num_prefill_kv_tokens]

                if fp8_attention:
                    num_kv_tokens, num_kv_heads, head_size = key.shape

                    key, _ = ops.scaled_fp8_quant(
                        key.reshape((num_kv_tokens,
                                    num_kv_heads * head_size)).contiguous(),
                        layer._k_scale)
                    key = key.reshape((num_kv_tokens, num_kv_heads, head_size))

                    value, _ = ops.scaled_fp8_quant(
                        value.reshape((num_kv_tokens,
                                    num_kv_heads * head_size)).contiguous(),
                        layer._v_scale)
                    value = value.reshape(
                        (num_kv_tokens, num_kv_heads, head_size))

                descale_shape = (q_seq_start_loc.shape[0] - 1, key.shape[1])
                # flash_attn_varlen_func(
                #     q=query,
                #     k=key,
                #     v=value,
                #     cu_seqlens_q=q_seq_start_loc,
                #     cu_seqlens_k=k_seq_start_loc,
                #     max_seqlen_q=q_seq_len,
                #     max_seqlen_k=k_seq_len,
                #     softmax_scale=softmax_scale,
                #     causal=_get_causal_option(attn_type),
                #     window_size=window_size,
                #     alibi_slopes=alibi_slopes,
                #     softcap=logits_soft_cap,
                #     out=prefill_output,
                #     fa_version=self.vllm_flash_attn_version,
                #     q_descale=layer._q_scale.expand(descale_shape),
                #     k_descale=layer._k_scale.expand(descale_shape),
                #     v_descale=layer._v_scale.expand(descale_shape),
                # )
                flash_prefill_varlen_func(
                    query,
                    key,
                    value,
                    prefill_output,
                    q_seq_start_loc,
                    q_seq_len,
                    cfg["alpha"],
                    cfg["block_size"],
                    cfg["attention_sink"],
                    cfg["window_size"],
                    cfg["last_n_full_block"]
                )
                # flash_sparse_infer_out(
                #     query.unsqueeze(0),
                #     key.unsqueeze(0),
                #     value.unsqueeze(0),
                #     prefill_output.unsqueeze(0),
                #     cfg["block_size"],
                #     cfg["attention_sink"],
                #     cfg["window_size"],
                #     cfg["alpha"],
                #     cfg["last_n_full_block"],
                #     cfg["min_budget"]
                # )
                
            else:
                # prefix-enabled attention
                assert attn_type == AttentionType.DECODER, (
                    "Only decoder-only models support prefix caching")
                assert prefill_meta.seq_lens is not None
                assert prefill_meta.query_start_loc is not None
                max_seq_len = max(prefill_meta.seq_lens)
                descale_shape = (prefill_meta.query_start_loc.shape[0] - 1,
                                key.shape[1])
                flash_attn_varlen_func(  # noqa
                    q=query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=prefill_meta.query_start_loc,
                    max_seqlen_q=prefill_meta.max_query_len,
                    seqused_k=prefill_meta.seq_lens_tensor,
                    max_seqlen_k=max_seq_len,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    block_table=prefill_meta.block_tables,
                    softcap=logits_soft_cap,
                    out=prefill_output,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
                # flash_sparse_infer_out(
                #     query.unsqueeze(0),
                #     key.unsqueeze(0),
                #     value.unsqueeze(0),
                #     prefill_output.unsqueeze(0),
                #     cfg['block_size'],
                #     cfg['attention_sink'],
                #     cfg['window_size'],
                #     cfg['alpha'],
                #     cfg['last_n_block_full'],
                #     cfg['min_budget']
                # )

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            # Use flash_attn_varlen_func kernel for speculative decoding
            # because different queries might have different lengths.

            assert decode_meta.max_decode_query_len is not None
            # use only for actual varlen decoding
            if decode_meta.max_decode_query_len > 1:
                assert attn_type == AttentionType.DECODER, (
                    "Only decoder-only models support max_decode_query_len > 1"
                )
                assert decode_meta.query_start_loc is not None
                descale_shape = (decode_meta.query_start_loc.shape[0] - 1,
                                key.shape[1])
                flash_attn_varlen_func(
                    q=decode_query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=decode_meta.query_start_loc,
                    max_seqlen_q=decode_meta.max_decode_query_len,
                    seqused_k=decode_meta.seq_lens_tensor,
                    max_seqlen_k=decode_meta.max_decode_seq_len,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    softcap=logits_soft_cap,
                    block_table=decode_meta.block_tables,
                    out=decode_output,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
            else:
                # Use flash_attn_with_kvcache for normal decoding.
                (
                    seq_lens_arg,
                    _,
                    block_tables_arg,
                ) = get_seq_len_block_table_args(decode_meta, False, attn_type)
                descale_shape = (seq_lens_arg.shape[0], key_cache.shape[-2])
                flash_attn_with_kvcache(
                    q=decode_query.unsqueeze(1),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    block_table=block_tables_arg,
                    cache_seqlens=seq_lens_arg,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    softcap=logits_soft_cap,
                    out=decode_output.unsqueeze(1),
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
        return output



    fa_backend.FlashAttentionImpl.forward = flash_sparse_infer_forward
    print(f"Successfully patched Rank {rank}")

apply_patch_to_worker()
