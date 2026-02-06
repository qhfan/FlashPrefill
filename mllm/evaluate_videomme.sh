VIDEO_PATH=Video-MME/data
MODEL_PATH=Qwen/Qwen3-VL-30B-A3B-Instruct
GT_FILE=videomme/test-00000-of-00001.parquet
OUTPUT_DIR=videomme/Qwen3-VL-30B-A3B-Instruct-flashprefill
MAX_FRAEMES=128

gpu_num=8


rm -rf $OUTPUT_DIR


for IDX in $(seq 0 $((gpu_num-1))); do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$IDX python run_videomme.py \
        --model_path $MODEL_PATH \
        --video_dir $VIDEO_PATH \
        --gt_file $GT_FILE \
        --num-chunks $gpu_num \
        --chunk-idx $IDX \
        --output_dir $OUTPUT_DIR \
        --max_frames $MAX_FRAEMES \
        --use_patch &
done
wait

MERGED_FILE="${OUTPUT_DIR}/result.jsonl"

if [ -f "$MERGED_FILE" ]; then
    rm "$MERGED_FILE"
fi


for IDX in $(seq 0 $((gpu_num-1))); do
    INPUT_FILE="${OUTPUT_DIR}/result_chunk_${IDX}_of_${gpu_num}.jsonl"
    
    if [ -f "$INPUT_FILE" ]; then
        cat "$INPUT_FILE" >> "$MERGED_FILE"
        echo "merged: $INPUT_FILE"
    else
        echo "not found $INPUT_FILE"
    fi
done

echo "cleaning..."
for IDX in $(seq 0 $((gpu_num-1))); do
    INPUT_FILE="${OUTPUT_DIR}/result_chunk_${IDX}_of_${gpu_num}.jsonl"
    if [ -f "$INPUT_FILE" ]; then
        rm "$INPUT_FILE"
    fi
done

python3 evaluate_videomme.py --file $MERGED_FILE
