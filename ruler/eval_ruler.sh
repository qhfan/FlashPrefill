model_path=ckpt/Qwen3-30B-A3B-Instruct-2507
output=RULER/output_res 
model_name=Qwen3-30B-A3B-Instruct-2507
self_defined_model=Qwen3MoeForCausalLMFlashPrefill

for task in ruler_64k; do
    python eval.py --config configs/${task}.yaml \
    --model_name_or_path $model_path \
    --output_dir $output/$model_name/$task \
    --self_defined_model $self_defined_model
done
