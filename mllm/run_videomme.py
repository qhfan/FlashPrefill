import os
import json
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen3VLMoeForConditionalGeneration
from qwen_vl_utils import process_vision_info
from patches.patch_qwen2_5_vl import patch_qwen2_5_vl_model
from patches.patch_qwen3_moe_vl import patch_qwen3_moe_vl_model


def load_video_as_frames(video_path, max_frames=128):

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    indices = np.linspace(0, total_frames - 1, min(total_frames, max_frames), dtype=int)
    
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(frame) for frame in frames]


def get_chunk(lst, n, k):

    chunk_size = (len(lst) + n - 1) // n
    start = k * chunk_size
    end = min((k + 1) * chunk_size, len(lst))
    return lst[start:end]

def run_inference(args):

    device = f"cuda:0"

    if "qwen2" in args.model_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype="bfloat16",
                attn_implementation="flash_attention_2",
                device_map={"": device}
            )
        if args.use_patch:
            patch_qwen2_5_vl_model(model)
    elif "qwen3" in args.model_path.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
            device_map={"": device}
        )
        if args.use_patch:
            patch_qwen3_moe_vl_model(model)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    df = pd.read_parquet(args.gt_file)
    data_list = df.to_dict('records')
    
    chunk_data = get_chunk(data_list, args.num_chunks, args.chunk_idx)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"result_chunk_{args.chunk_idx}_of_{args.num_chunks}.jsonl")
    
    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                existing_ids.add(json.loads(line)['question_id'])

    ans_file = open(output_file, "a")

    for sample in tqdm(chunk_data, desc=f"GPU {args.chunk_idx}"):
        if sample['question_id'] in existing_ids:
            continue

        video_id = sample['videoID']
        video_path = None

        for fmt in ['.mp4', '.mkv', '.avi', '.webm']:
            temp_path = os.path.join(args.video_dir, f"{video_id}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break
        
        if video_path is None:
            print(f"Skip: Video {video_id} not found.")
            continue

        try:
            frames = load_video_as_frames(video_path, max_frames=args.max_frames)
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            continue

        question = sample['question']
        options = "\n".join(sample['options'])
        prompt = f"Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\nQuestion: {question}\nChoices:\n{options}\nAnswer with only the letter (A, B, C, or D) directly."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames}, 
                    {"type": "text", "text": prompt},
                ],
            }
        ]


        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=16,
                do_sample=False,
                temperature=0.0
            )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        res_item = {
            "question_id": sample['question_id'],
            "video_id": video_id,
            "prediction": output_text.strip(),
            "ground_truth": sample['answer'],
            "duration": sample['duration']
        }
        ans_file.write(json.dumps(res_item) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True, help='Video storage directory')
    parser.add_argument('--gt_file', type=str, required=True, help='Video-MME Parquet file path')
    parser.add_argument('--output_dir', type=str, default="results_videomme")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num-chunks", type=int, default=1, help='Total number of GPUs/chunks')
    parser.add_argument("--chunk-idx", type=int, default=0, help='Current GPU/chunk index')
    parser.add_argument("--max_frames", type=int, default=128, help='Max frames to sample per video')
    parser.add_argument("--use_patch", action='store_true', help='Whether to patch the model')
    
    args = parser.parse_args()
    
    run_inference(args)
