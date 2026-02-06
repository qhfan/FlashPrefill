from numpy import dtype
import torch
import triton
import triton.language as tl
from fla.utils import autocast_custom_fwd, contiguous
import math

def get_cu_tile_seqlens(cu_seqlens, BLOCK_SIZE):
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    tiles_per_seq = (seqlens + BLOCK_SIZE - 1) // BLOCK_SIZE
    tiles_2_per_seq = tiles_per_seq ** 2
    cu_tile_seqlens = torch.zeros(cu_seqlens.shape, dtype=torch.int32, device=cu_seqlens.device)
    cu_tile_seqlens[1:] = torch.cumsum(tiles_per_seq, dim=0)

    cu_tile_2_seqlens = torch.zeros(cu_seqlens.shape, dtype=torch.int32, device=cu_seqlens.device)
    cu_tile_2_seqlens[1:] = torch.cumsum(tiles_2_per_seq, dim=0)
    
    return cu_tile_seqlens, cu_tile_2_seqlens

def get_mean_configs():
    configs = []
    warps = [4, 8]
    stages = [2, 3, 4, 5]
    for w in warps:
        for s in stages:
            configs.append(
                triton.Config({}, num_warps=w, num_stages=s)
            )
    return configs
@triton.autotune(
    configs=get_mean_configs(),
    key=['max_seq_len', 'num_q_heads'],
)
@triton.jit
def compute_mean_vector_varlen(
    Q_ptr, mQ_ptr,
    cu_seqlens_ptr,      
    cu_tile_seqlens_ptr, 
    stride_qs, stride_qh, stride_qd,   
    stride_mqt, stride_mqh, stride_mqd, 
    num_q_heads,
    max_seq_len,         
    BLOCK_SIZE: tl.constexpr,
    D_HEAD: tl.constexpr
):

    query_tile_index = tl.program_id(0).to(tl.int64) 
    offset_zh = tl.program_id(1).to(tl.int64)        

    offset_batch = offset_zh // num_q_heads
    offset_q_head = offset_zh % num_q_heads

    start_token_idx = tl.load(cu_seqlens_ptr + offset_batch)
    end_token_idx = tl.load(cu_seqlens_ptr + offset_batch + 1)
    curr_seq_len = end_token_idx - start_token_idx

    if query_tile_index * BLOCK_SIZE >= curr_seq_len:
        return

    start_tile_idx = tl.load(cu_tile_seqlens_ptr + offset_batch)
    end_tile_idx = tl.load(cu_tile_seqlens_ptr + offset_batch + 1)
    curr_tile_len = end_tile_idx - start_tile_idx

    Q_base_ptr = Q_ptr + start_token_idx * stride_qs + offset_q_head * stride_qh
    mQ_base_ptr = mQ_ptr + start_tile_idx * stride_mqt + offset_q_head * stride_mqh

    offset_q = query_tile_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_dim = tl.arange(0, D_HEAD)

    load_mask = (offset_q[:, None] < curr_seq_len) & (offset_dim[None, :] < D_HEAD)
    q = tl.load(
        Q_base_ptr + offset_q[:, None] * stride_qs + offset_dim[None, :] * stride_qd,
        mask=load_mask,
        other=0.0
    )

    q_sum = tl.sum(q, axis=0)
    
    q_mask = (offset_q < curr_seq_len)
    num_ele = tl.sum(q_mask.to(tl.int32)).to(q.dtype)
    
    q_mean = tl.where(num_ele > 0, q_sum / num_ele, 0.0)

    store_ptr = mQ_base_ptr + query_tile_index * stride_mqt + offset_dim * stride_mqd
    tl.store(store_ptr, q_mean, mask=(offset_dim < D_HEAD) & (query_tile_index < curr_tile_len))


def get_score_configs():
    configs = []
    k_tiles = [32, 64, 128]
    warps = [4, 8]
    stages = [2, 3, 4, 5]

    for k in k_tiles:
        for w in warps:
            for s in stages:
                if k == 256 and w == 4: continue
                configs.append(
                    triton.Config({'K_TILE_SIZE': k}, num_warps=w, num_stages=s)
                )
    return configs
@triton.autotune(
    configs=get_score_configs(),
    key=['query_len_max', 'sub_key_len_max', 'num_q_heads', 'num_k_heads'],
)
@triton.jit
def compute_block_score_varlen(
    Q_ptr, K_ptr, scale,
    sc_ptr, mx_ptr,
    cu_seqlens_ptr,
    cu_tile_seqlens_ptr,
    cu_tile_2_seqlens_ptr,
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_scl, stride_sch,
    stride_mxl, stride_mxh,
    num_q_heads, num_k_heads,
    query_len_max, sub_key_len_max,
    BLOCK_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    K_STRIDE: tl.constexpr,
    D_HEAD: tl.constexpr
):
    '''
    optimized with stride
    '''
    num_stride_per_block: tl.constexpr = BLOCK_SIZE // K_STRIDE
    num_block_per_tile: tl.constexpr = K_TILE_SIZE // num_stride_per_block

    query_tile_index = tl.program_id(0).to(tl.int64)
    offset_zh = tl.program_id(1).to(tl.int64)

    offset_batch = offset_zh // num_q_heads
    group_size = num_q_heads // num_k_heads

    offset_q_heads = offset_zh % num_q_heads
    offset_k_heads = offset_q_heads // group_size

    start_token_idx = tl.load(cu_seqlens_ptr + offset_batch)
    end_token_idx = tl.load(cu_seqlens_ptr + offset_batch + 1)
    curr_seq_len = end_token_idx - start_token_idx

    if query_tile_index * BLOCK_SIZE >= curr_seq_len:
        return
    
    start_tile_idx = tl.load(cu_tile_seqlens_ptr + offset_batch)
    end_tile_idx = tl.load(cu_tile_seqlens_ptr + offset_batch + 1)
    curr_tiles_len = end_tile_idx - start_tile_idx

    Q_base_ptr = Q_ptr + start_token_idx * stride_qm + offset_q_heads * stride_qh
    K_base_ptr = K_ptr + start_tile_idx * stride_kn + offset_k_heads * stride_kh

    start_score_tile_idx = tl.load(cu_tile_2_seqlens_ptr + offset_batch)

    sc_base_ptr = sc_ptr + start_score_tile_idx * stride_scl + offset_q_heads * stride_sch
    mx_base_ptr = mx_ptr + start_score_tile_idx * stride_mxl + offset_q_heads * stride_mxh

    offset_q = query_tile_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_dim = tl.arange(0, D_HEAD)

    q_index_max = query_tile_index * BLOCK_SIZE + BLOCK_SIZE - 1
    q_index = offset_q

    q = tl.load(
        Q_base_ptr + offset_q[:, None] * stride_qm + offset_dim[None, :] * stride_qd,
        mask=(offset_q[:, None] < curr_seq_len) & (offset_dim[None, :] < D_HEAD),
        other=0.0
    )

    lo = 0
    hi = tl.cdiv(q_index_max + 1, K_STRIDE)

    sm_scale = scale * 1.4426950408889634

    for j in range(lo, hi, K_TILE_SIZE):
        offset_k = j + tl.arange(0, K_TILE_SIZE)
        k_index_min = offset_k * K_STRIDE
        k_index_max = offset_k * K_STRIDE + K_STRIDE - 1
        k = tl.load(
            K_base_ptr + offset_k[:, None] * stride_kn + offset_dim[None, :] * stride_kd,
            mask=(offset_k[:, None] < curr_tiles_len) & (offset_dim[None, :] < D_HEAD),
            other=0.0
        )

        qk = tl.dot(q, tl.trans(k))
        causal_mask = (q_index[:, None] >= k_index_max[None, :]) # (block_size, k_tile_size)

        qk = tl.where(causal_mask, qk, float('-inf'))
        qk *= sm_scale # (block_size, (num_block_per_tile, num_stride_per_block))

        qk = tl.reshape(qk, (BLOCK_SIZE, num_block_per_tile, num_stride_per_block))
        m_i_block = tl.max(qk, axis=2) #(block_size, num_block_per_tile)
        m_i_block = tl.max(m_i_block, axis=0) # (num_block_per_tile)

        qk_block = qk - m_i_block[None, :, None]
        p_block = tl.exp2(qk_block)

        causal_mask_block = tl.reshape(causal_mask, (BLOCK_SIZE, num_block_per_tile, num_stride_per_block))
        p_block = tl.where(causal_mask_block, p_block, 0.0) #(BLOCK_SIZE, num_block_per_tile, num_stride_per_block)


        p_block = tl.sum(p_block, 2) 
        p_block = tl.sum(p_block, 0) # num_block_per_tile

        offset_k_blocks = (j // K_TILE_SIZE) * num_block_per_tile + tl.arange(0, num_block_per_tile)

        tl.store(
            sc_base_ptr + (query_tile_index * curr_tiles_len + offset_k_blocks) * stride_scl,
            p_block,
            mask=(query_tile_index < curr_tiles_len) & (offset_k_blocks < curr_tiles_len)
        )

        tl.store(
            mx_base_ptr + (query_tile_index * curr_tiles_len + offset_k_blocks) * stride_mxl,
            m_i_block,
            mask=(query_tile_index < curr_tiles_len) & (offset_k_blocks < curr_tiles_len)
        )

def get_score_sub_max_configs():
    configs = []
    q_tiles = [32, 64, 128]
    k_tiles = [32, 64, 128]
    warps = [4, 8]
    stages = [2, 3, 4, 5]

    for q in q_tiles:
        for k in k_tiles:
            for w in warps:
                for s in stages:
                    # 过滤掉一些可能不合理的组合
                    configs.append(
                        triton.Config({'Q_TILE_SIZE': q, 'K_TILE_SIZE': k}, num_warps=w, num_stages=s)
                    )
    return configs
@triton.autotune(
    configs=get_score_sub_max_configs(),
    key=['block_len_max', 'num_q_heads'],
)
@triton.jit
def compute_block_sub_max_varlen(
    mx_ptr, out_ptr,
    cu_tile_seqlens_ptr,
    cu_tile_2_seqlens_ptr,
    stride_mxl, stride_mxh,
    stride_outl, stride_outh,
    num_q_heads,
    block_len_max,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr
):
    '''
    get_max varlen
    '''
    query_tile_index = tl.program_id(0).to(tl.int64)
    offset_zh = tl.program_id(1).to(tl.int64)

    offset_batch = offset_zh // num_q_heads
    offset_q_heads = offset_zh % num_q_heads

    start_tile_idx = tl.load(cu_tile_seqlens_ptr + offset_batch)
    end_tile_idx = tl.load(cu_tile_seqlens_ptr + offset_batch + 1)
    curr_tiles_len = end_tile_idx - start_tile_idx

    if query_tile_index * Q_TILE_SIZE >= curr_tiles_len:
        return

    start_score_tile_idx = tl.load(cu_tile_2_seqlens_ptr + offset_batch)

    mx_base_ptr = mx_ptr + start_score_tile_idx * stride_mxl + offset_q_heads * stride_mxh
    out_base_ptr = out_ptr + start_score_tile_idx * stride_outl + offset_q_heads * stride_outh

    offset_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    lo = 0
    hi = query_tile_index * Q_TILE_SIZE + Q_TILE_SIZE

    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)

    for j in range(lo, hi, K_TILE_SIZE): #type: ignore
        offset_k = j + tl.arange(0, K_TILE_SIZE)
        mx_tile = tl.load(
            mx_base_ptr + (offset_q[:, None] * curr_tiles_len + offset_k[None, :]) * stride_mxl,
            mask=(offset_q[:, None] < curr_tiles_len) & (offset_k[None, :] < curr_tiles_len),
            other=float('-inf')
        )
        m_i = tl.maximum(m_i, tl.max(mx_tile, axis=1))

    # m_i = tl.where(m_i > float('-inf'), m_i, float('inf'))

    for j in range(lo, hi, K_TILE_SIZE): #type: ignore
        offset_k = j + tl.arange(0, K_TILE_SIZE)
        mx_tile = tl.load(
            mx_base_ptr + (offset_q[:, None] * curr_tiles_len + offset_k[None, :]) * stride_mxl,
            mask=(offset_q[:, None] < curr_tiles_len) & (offset_k[None, :] < curr_tiles_len),
            other=0.0
        )

        out_tile = tl.exp2(mx_tile - m_i[:, None])

        tl.store(
            out_base_ptr + (offset_q[:, None] * curr_tiles_len + offset_k[None, :]) * stride_outl,
            out_tile,
            mask=(offset_q[:, None] < curr_tiles_len) & (offset_k[None, :] < curr_tiles_len)
        )
        

def get_score_max_configs():
    configs = []
    q_tiles = [32, 64, 128]
    k_tiles = [32, 64, 128]
    warps = [4, 8]
    stages = [2, 3, 4, 5]

    for q in q_tiles:
        for k in k_tiles:
            for w in warps:
                for s in stages:
                    configs.append(
                        triton.Config({'Q_TILE_SIZE': q, 'K_TILE_SIZE': k}, num_warps=w, num_stages=s)
                    )
    return configs
@triton.autotune(
    configs=get_score_max_configs(),
    key=['block_len_max', 'num_q_heads'],
)
@triton.jit
def compute_block_max_varlen(
    sc_ptr, out_ptr, out_number_ptr,
    cu_tile_seqlens_ptr,
    cu_tile_2_seqlens_ptr,
    stride_scl, stride_sch,
    stride_outl, stride_outh,
    stride_numberl,
    alpha,
    attention_sink,
    window_size,
    last_n_block,
    num_q_heads,
    block_len_max,
    total_seq_len, # this is the number of block
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    '''
    filter blocks
    '''
    query_tile_index = tl.program_id(0).to(tl.int64)
    offset_zh = tl.program_id(1).to(tl.int64)

    offset_batch = offset_zh // num_q_heads
    offset_q_heads = offset_zh % num_q_heads

    start_tile_idx = tl.load(cu_tile_seqlens_ptr + offset_batch)
    end_tile_idx = tl.load(cu_tile_seqlens_ptr + offset_batch + 1)
    curr_tiles_len = end_tile_idx - start_tile_idx

    if query_tile_index * Q_TILE_SIZE >= curr_tiles_len:
        return
    
    start_score_tile_idx = tl.load(cu_tile_2_seqlens_ptr + offset_batch)

    sc_base_ptr = sc_ptr + start_score_tile_idx * stride_scl + offset_q_heads * stride_sch
    out_base_ptr = out_ptr + start_score_tile_idx * stride_outl + offset_q_heads * stride_outh
    out_number_base_ptr = out_number_ptr + (total_seq_len * offset_q_heads + start_tile_idx) * stride_numberl

    offset_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    mask_last_full = (offset_q >= (curr_tiles_len - last_n_block))[:, None]

    lo = 0
    hi = query_tile_index * Q_TILE_SIZE + Q_TILE_SIZE
    # hi = tl.minimum(curr_tiles_len, query_tile_index * Q_TILE_SIZE + Q_TILE_SIZE)

    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    number_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.int32)

    for j in range(lo, hi, K_TILE_SIZE): #type: ignore
        offset_k = j + tl.arange(0, K_TILE_SIZE)
        sc_tile = tl.load(
            sc_base_ptr + (offset_q[:, None] * curr_tiles_len + offset_k[None, :]) * stride_scl,
            mask=(offset_q[:, None] < curr_tiles_len) & (offset_k[None, :] < curr_tiles_len),
            other=float('-inf')
        )
        m_i = tl.maximum(m_i, tl.max(sc_tile, axis=1))

    for j in range(lo, hi, K_TILE_SIZE): #type: ignore
        offset_k = j + tl.arange(0, K_TILE_SIZE)
        sc_tile = tl.load(
            sc_base_ptr + (offset_q[:, None] * curr_tiles_len + offset_k[None, :]) * stride_scl,
            mask=(offset_q[:, None] < curr_tiles_len) & (offset_k[None, :] < curr_tiles_len),
            other=float('-inf')
        )
        out_tile_mask = (sc_tile >= alpha * m_i[:, None])
        window_mask = (offset_q[:, None] >= offset_k[None, :]) & ((offset_q[:, None] - offset_k[None, :]) < window_size)
        sink_mask = (offset_k[None, :] < attention_sink)
        causal_mask = (offset_q[:, None] >= offset_k[None, :])
        out_tile_mask = (out_tile_mask | window_mask | sink_mask | mask_last_full) & causal_mask
        out_tile = tl.where(out_tile_mask, offset_k[None, :], -1)
        tl.store(
            out_base_ptr + (offset_q[:, None] * curr_tiles_len + offset_k[None, :]) * stride_outl,
            out_tile,
            mask=(offset_q[:, None] < curr_tiles_len) & (offset_k[None, :] < curr_tiles_len)
        )
        number_i += tl.sum(out_tile_mask.to(tl.int32), axis=1)

    tl.store(
        out_number_base_ptr + offset_q * stride_numberl,
        number_i,
        mask=offset_q < curr_tiles_len
    )


def get_configs():
    configs = []
    tile_sizes = [
        (64, 64),   
        (128, 64),  
        (64, 128)   
    ]
    warps_list = [4, 8]
    stages_list = [2, 3, 4, 5]

    for (q_tile, k_tile) in tile_sizes:
        for w in warps_list:
            for s in stages_list:
                if q_tile == 128 and w == 4:
                    continue
                
                configs.append(
                    triton.Config(
                        {'Q_TILE_SIZE': q_tile, 'K_TILE_SIZE': k_tile}, 
                        num_warps=w, 
                        num_stages=s
                    )
                )
    return configs

@triton.autotune(
    configs=get_configs(),
    key=['query_len', 'key_len', 'num_q_heads', 'num_k_heads', 'BLOCK_SIZE'],
    prune_configs_by={
        'early_config_prune': lambda configs, named_args, **kwargs: [
            c for c in configs 
            if c.kwargs['Q_TILE_SIZE'] <= kwargs['BLOCK_SIZE'] 
            and c.kwargs['K_TILE_SIZE'] <= kwargs['BLOCK_SIZE']
        ]
    }
)
@triton.jit
def _flash_block_sparse_varlen(
    Q_ptr, K_ptr, V_ptr, O_ptr, index_ptr, valid_ptr, scale,
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    stride_om, stride_oh, stride_od,
    stride_indexl,
    stride_validl,
    cu_seqlens_ptr,
    cu_tile_seqlens_ptr,
    cu_tile_2_seqlens_ptr,
    total_seq_len, 
    num_q_heads, num_k_heads,
    Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    D_HEAD: tl.constexpr
):
    query_tile_index = tl.program_id(0).to(tl.int64)
    offset_zh = tl.program_id(1).to(tl.int64)

    offset_batch = offset_zh // num_q_heads
    group_size = num_q_heads // num_k_heads
    index_group_size = BLOCK_SIZE // Q_TILE_SIZE
    '''
    BLOCK_SIZE >= Q_TILE_SIZE
    BLOCK_SIZE >= K_TILE_SIZE
    only support key_len == query_len
    '''
    offset_q_heads = offset_zh % num_q_heads
    offset_k_heads = offset_q_heads // group_size

    start_token_idx = tl.load(cu_seqlens_ptr + offset_batch)
    end_token_idx = tl.load(cu_seqlens_ptr + offset_batch + 1)
    curr_seq_len = end_token_idx - start_token_idx

    if query_tile_index * Q_TILE_SIZE >= curr_seq_len:
        return
    
    Q_base_ptr = Q_ptr + start_token_idx * stride_qm + offset_q_heads * stride_qh
    K_base_ptr = K_ptr + start_token_idx * stride_kn + offset_k_heads * stride_kh
    V_base_ptr = V_ptr + start_token_idx * stride_vn + offset_k_heads * stride_vh
    O_base_ptr = O_ptr + start_token_idx * stride_om + offset_q_heads * stride_oh
    
    start_tile_idx = tl.load(cu_tile_seqlens_ptr + offset_batch)

    valid_base_ptr = valid_ptr + (total_seq_len * offset_q_heads + start_tile_idx) * stride_validl

    valid_start = tl.load(valid_base_ptr + (query_tile_index // index_group_size) * stride_validl)
    valid_end = tl.load(valid_base_ptr + ((query_tile_index // index_group_size) + 1) * stride_validl)

    index_base_ptr = index_ptr + valid_start * stride_indexl

    offset_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    offset_dim = tl.arange(0, D_HEAD)

    q_mask = (offset_q[:, None] < curr_seq_len) & (offset_dim[None, :] < D_HEAD)
    q = tl.load(
        Q_base_ptr + offset_q[:, None] * stride_qm + offset_dim[None, :] * stride_qd,
        mask=q_mask,
        other=0.0
    )

    sm_scale = scale * 1.4426950408889634

    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l_i = tl.full((Q_TILE_SIZE,), 1.0, dtype=tl.float32)
    acc = tl.zeros((Q_TILE_SIZE, D_HEAD), dtype=tl.float32)

    lo = 0
    hi = valid_end - valid_start

    block_index = tl.load(index_base_ptr + lo * stride_indexl, mask=(lo < hi), other=0)

    for i in range(lo, hi):
        next_block_index = tl.load(index_base_ptr + (i+1) * stride_indexl, mask=(i+1 < hi), other=0)
        key_start_index = block_index * BLOCK_SIZE
        key_end_index = (block_index + 1) * BLOCK_SIZE

        is_diagonal_block = (block_index == (query_tile_index // index_group_size))

        for j in range(key_start_index, key_end_index, K_TILE_SIZE):
            k_index = j + tl.arange(0, K_TILE_SIZE)
            k_j = tl.load(
                K_base_ptr + k_index[:, None] * stride_kn + offset_dim[None, :] * stride_kd,
                mask=(k_index[:, None]<curr_seq_len) & (offset_dim[None, :]<D_HEAD),
                other=0.0
            )
            v_j = tl.load(
                V_base_ptr + k_index[:, None] * stride_vn + offset_dim[None, :] * stride_vd,
                mask=(k_index[:, None]<curr_seq_len) & (offset_dim[None, :]<D_HEAD),
                other=0.0
            )

            qk = tl.dot(q, tl.trans(k_j))
            if is_diagonal_block:
                dist = offset_q[:, None] - k_index[None, :]
                causal_mask = (dist >= 0)
                qk = tl.where(causal_mask, qk, float('-inf'))
            qk *= sm_scale
            m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
            qk -= m_i_new[:, None]
            p = tl.exp2(qk) 
            # if is_diagonal_block:
            #     p = tl.where(causal_mask, p, 0.0)
            lij = tl.sum(p, 1)
            alpha = tl.exp2(m_i - m_i_new)
            alpha_mask = (alpha != alpha)
            alpha = tl.where(alpha_mask, 1.0, alpha)
            acc *= alpha[:, None]
            l_i = l_i * alpha + lij
            p = p.to(v_j.dtype) 
            acc += tl.dot(p, v_j)
            m_i = m_i_new
        block_index = next_block_index

    l_rec = 1 / l_i[:, None]
    acc = acc * l_rec

    o_offset_m = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    o_offset_d = tl.arange(0, D_HEAD)

    o_mask = (o_offset_m[:, None] < curr_seq_len) & (o_offset_d[None, :] < D_HEAD)
    tl.store(
        O_base_ptr + o_offset_m[:, None] * stride_om + o_offset_d[None, :] * stride_od,
        acc.to(q.dtype),
        mask=o_mask
    )

class FlashPrefill_varlen(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, o, cu_seqlens, max_seq_len, block_size=128, k_stride=128, attention_sink=2, window=4, alpha=0.7, last_n_blocks=2):
        '''
        q: (l n d)
        k: (l n d)
        v: (l n d)
        o: (l n d)
        cu_seqlens: (batch + 1,)
        '''
        assert block_size == k_stride, "block_size must equal to k_stride"
        cu_tile_seqlens, cu_tile_2_seqlens = get_cu_tile_seqlens(cu_seqlens, block_size)
        _, num_q_heads, head_dim = q.shape
        _, num_k_heads, _ = k.shape
        batch_size = cu_seqlens.shape[0] - 1

        mean_k = torch.empty(cu_tile_seqlens[-1], num_k_heads, head_dim, dtype=k.dtype, device=k.device) #type: ignore

        def grid_mean_varlen(meta):
            return (triton.cdiv(max_seq_len, k_stride), batch_size * num_k_heads, 1)
        compute_mean_vector_varlen[grid_mean_varlen](
            k, mean_k,
            cu_seqlens,
            cu_tile_seqlens,
            *k.stride(),
            *mean_k.stride(),
            num_k_heads,
            max_seq_len,
            block_size,
            head_dim
        )

        query_len_max = max_seq_len
        block_len_max = sub_key_len_max = triton.cdiv(max_seq_len, k_stride)

        output_score = torch.full(size=(int(cu_tile_2_seqlens[-1]), num_q_heads), fill_value=float('-inf'), dtype=torch.float32, device=q.device)
        output_max = torch.full(size=(int(cu_tile_2_seqlens[-1]), num_q_heads), fill_value=float('-inf'), dtype=torch.float32, device=q.device)


        def grid_block_score_varlen(meta):
            return (triton.cdiv(max_seq_len, block_size), batch_size * num_q_heads, 1)
        compute_block_score_varlen[grid_block_score_varlen](
            q, mean_k, 1 / (head_dim ** 0.5), 
            output_score, output_max,
            cu_seqlens,
            cu_tile_seqlens,
            cu_tile_2_seqlens,
            *q.stride(),
            *mean_k.stride(),
            *output_score.stride(),
            *output_max.stride(),
            num_q_heads, num_k_heads,
            query_len_max, sub_key_len_max,
            block_size,
            K_STRIDE=block_size,
            D_HEAD=head_dim
        )

        output_max_sub_max = torch.full_like(output_max, fill_value=0)

        def grid_block_sub_max_varlen(meta):
            return (triton.cdiv(block_len_max, meta["Q_TILE_SIZE"]), batch_size * num_q_heads, 1)
        compute_block_sub_max_varlen[grid_block_sub_max_varlen](
            output_max, output_max_sub_max,
            cu_tile_seqlens,
            cu_tile_2_seqlens,
            *output_max.stride(),
            *output_max_sub_max.stride(),
            num_q_heads,
            block_len_max
        )

        # print(torch.any(torch.isinf(output_max_sub_max)))

        # print(output_score)

        output_score = output_score * output_max_sub_max

        # print(output_score)

        total_block_seq_len = cu_tile_seqlens[-1]

        out_number = torch.full(size=(total_block_seq_len * num_q_heads,), fill_value=0, dtype=torch.int32, device=q.device)
        out_index = torch.full(size=(int(cu_tile_2_seqlens[-1]), num_q_heads), fill_value=-1, dtype=torch.int32, device=q.device)

        def grid_compute_block_max_varlen(meta):
            return (triton.cdiv(block_len_max, meta["Q_TILE_SIZE"]), batch_size * num_q_heads, 1)
        compute_block_max_varlen[grid_compute_block_max_varlen](
            output_score, out_index, out_number,
            cu_tile_seqlens,
            cu_tile_2_seqlens,
            *output_score.stride(),
            *out_index.stride(),
            *out_number.stride(),
            alpha,
            attention_sink,
            window,
            last_n_blocks,
            num_q_heads,
            block_len_max,
            total_block_seq_len.item()
        )

        out_index = out_index.transpose(0, 1).flatten(0, 1)

        # index_for_debug = out_index.reshape(num_q_heads, -1)
        # seq_len = int(math.sqrt(int(index_for_debug.shape[-1])))
        # print('*'*120)
        # print(seq_len)
        # index_for_debug = index_for_debug.reshape(num_q_heads, seq_len, seq_len)
        # print(index_for_debug[4][-4])

        # number_for_debug = out_number.reshape(num_q_heads, int(total_block_seq_len)).contiguous()
        # print(total_block_seq_len)
        # print(number_for_debug[4][-4])

        out_index = out_index[out_index!=-1].contiguous()
        out_number_cu = torch.zeros(out_number.shape[0]+1, dtype=torch.int32, device=q.device)
        out_number_cu[1:] = torch.cumsum(out_number, dim=0)
        # print(out_number_cu[-1])



        def grid_flash_block_sparse_varlen(meta):
            return (triton.cdiv(max_seq_len, meta['Q_TILE_SIZE']), batch_size * num_q_heads, 1)
        _flash_block_sparse_varlen[grid_flash_block_sparse_varlen](
            q, k, v, o, out_index, out_number_cu, 1 / (head_dim ** 0.5),
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *o.stride(),
            *out_index.stride(),
            *out_number_cu.stride(),
            cu_seqlens,
            cu_tile_seqlens,
            cu_tile_2_seqlens,
            total_block_seq_len.item(),
            num_q_heads, num_k_heads,
            BLOCK_SIZE=block_size,
            D_HEAD=head_dim
        )

@torch.no_grad()
def flash_prefill_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seq_len: int,
    alpha: float,
    block_size: int=128,
    attention_sink: int=2,
    window_size: int=4,
    last_n_block: int=2,
):
    '''
    q, k, v: (s h d)
    '''
    FlashPrefill_varlen.apply(
        q, k, v, o, cu_seqlens, max_seq_len,
        block_size, block_size, attention_sink,
        window_size, alpha, last_n_block
    )
