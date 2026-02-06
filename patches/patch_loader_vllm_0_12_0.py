import os
import torch
import types
# from vllm_models.ops import flash_sparse_infer_out
from ops import flash_prefill_varlen_func, flash_prefill
# from vllm_models.ops.flashparseinfernotune import flash_sparse_infer_out
import vllm.model_executor.model_loader as vllm_loader
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata, cascade_attention, FlashAttentionBackend
from vllm.attention.backends.abstract import (AttentionLayer,
                                              AttentionType)
from vllm.attention.utils.fa_utils import reshape_and_cache_flash, flash_attn_varlen_func


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
    import vllm.v1.attention.backends.flash_attn as fa_backend

    rank = str(getattr(worker, "rank", os.environ.get("RANK", "0")))
    cache_dir = f"/tmp/triton_cache_rank_{rank}"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = cache_dir

    print(cache_dir)

    original_forward = fa_backend.FlashAttentionImpl.forward

    def flash_prefill_forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)
        
        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(0)

        # key and value may be None in the case of cross attention. They are
        # calculated once based on the output from the encoder and then cached
        # in KV cache.
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            # queries are quantized in the attention layer
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            is_prefill = attn_metadata.max_query_len > 1

            if self.dcp_world_size > 1:
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
                return output
            else:
                if is_prefill:
                    flash_prefill_varlen_func(
                        query[:num_actual_tokens],
                        key[:num_actual_tokens],
                        value[:num_actual_tokens],
                        output[:num_actual_tokens],
                        cu_seqlens_q,
                        max_seqlen_q,
                        cfg["alpha"],
                        cfg["block_size"],
                        cfg["attention_sink"],
                        cfg["window_size"],
                        cfg["last_n_full_block"]
                    )
                    return output
                else:
                    flash_attn_varlen_func(
                        q=query[:num_actual_tokens],
                        k=key_cache,
                        v=value_cache,
                        out=output[:num_actual_tokens],
                        cu_seqlens_q=cu_seqlens_q,
                        max_seqlen_q=max_seqlen_q,
                        seqused_k=seqused_k,
                        max_seqlen_k=max_seqlen_k,
                        softmax_scale=self.scale,
                        causal=attn_metadata.causal,
                        alibi_slopes=self.alibi_slopes,
                        window_size=self.sliding_window,
                        block_table=block_table,
                        softcap=self.logits_soft_cap,
                        scheduler_metadata=scheduler_metadata,
                        fa_version=self.vllm_flash_attn_version,
                        q_descale=layer._q_scale.expand(descale_shape),
                        k_descale=layer._k_scale.expand(descale_shape),
                        v_descale=layer._v_scale.expand(descale_shape),
                        num_splits=attn_metadata.max_num_splits,
                        s_aux=self.sinks,
                    )
                    return output

        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output
    
    fa_backend.FlashAttentionImpl.forward = flash_prefill_forward
    print(f"Successfully patched Rank {rank}")

apply_patch_to_worker()
