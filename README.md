<h1 align="center">FlashPrefill</h1>

This repository provides the code for the paper [FlashPreill: Instantaneous Pattern Discovery and Thresholding for Ultra-Fast Long-Context Prefilling]()

## Abstract
Long-context modeling is a pivotal capability for Large Language Models, yet the quadratic complexity of attention remains a critical bottleneck, particularly during the compute-intensive prefilling phase. While various sparse attention mechanisms have been explored, they typically suffer from either significant search latency or insufficient sparsity. In this paper, we propose FlashPrefill, a framework enabling ultra-fast prefilling via instantaneous pattern discovery and thresholding. FlashPrefill leverages a fast block-searching technique to simultaneously locate dynamic vertical, slash, and block-sparse attention patterns. Crucially, it introduces a dynamic thresholding mechanism that bypasses the prohibitive overhead of sorting or accumulating attention scores while effectively eliminating the long-tail distribution to enhance sparsity. Extensive evaluations demonstrate that FlashPrefill achieves a substantial leap in efficiency, delivering an unprecedented $27.78\times$ speedup on 256K sequences. Notably, unlike existing methods that incur efficiency degradation on shorter contexts, FlashPrefill maintains a $1.71\times$ speedup even at a 4K context length, demonstrating its robustness and practical utility across varying sequence scales.

## Requirements
To use FlashPrefill, you will need the following packages:
- `torch==2.9.0`
- `triton==3.3.0`
- `transformers==4.56.1`
- `flash_attn==2.8.3` (optional)
- `vllm==0.10.0 or 0.12.0` (optional)

## Quick Start
For evaluation instructions, please refer to the README file in each respective subfolder.

## Integration with vLLM

We have integrated FlashPrefill into vLLM (version 0.10.0 or 0.12.0). You can use the following code to directly patch FlashPrefill into vLLM.

```python
from patches import patch_loader_vllm_0_10_0
# from patches import patch_loader_vllm_0_12_0
from vllm import LLM, SamplingParams

model = LLM(
    model=model_name,
    tokenizer=model_name,
    trust_remote_code=True,
    dtype="bfloat16",
    tensor_parallel_size=4,
    max_model_len=32 * 1024, 
    gpu_memory_utilization=0.9,
    enable_chunked_prefill=False,
    enable_prefix_caching=False
)
sampling_params = SamplingParams(temperature=0, max_tokens=64)

model.generate(prompts=[prompt], sampling_params=sampling_params)
output = outputs[0].outputs[0].text
```

## Acknowledgments
The codebase is based on the [MInference](https://github.com/microsoft/MInference), [XAttention](https://github.com/mit-han-lab/x-attention), and [FlexPrefill](https://github.com/ByteDance-Seed/FlexPrefill), we acknowledge the contributions of these works.
