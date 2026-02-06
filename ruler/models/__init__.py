from .llama import LlamaForCausalLM
from .qwen2 import Qwen2ForCausalLM
from .qwen3 import Qwen3MoeForCausalLM


model_dict = {
    'LlamaForCausalLMFlashPrefill': LlamaForCausalLM,
    'QwenForCausalLMFlashPrefill': Qwen2ForCausalLM,
    'Qwen3MoeForCausalLMFlashPrefill': Qwen3MoeForCausalLM
}
