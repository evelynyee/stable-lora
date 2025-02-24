#!/usr/bin/env python
# coding: utf-8

# import diffusers
from diffusers import DDPMScheduler, DiffusionPipeline
import torch
import os
# from huggingface_hub import model_info
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Parse command-line arguments.")

    # Boolean flag for lora
    parser.add_argument("--lora", action="store_true", help="Enable LoRA (default: false)")
    parser.add_argument("--prompt_file", type=str, default="eval/eval_prompts.jsonl", help="Path to the prompt file")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    pretrained = "runwayml/stable-diffusion-v1-5"
    weight_dtype = torch.float32
    output_dir = "train"
    eval_dir = 'test_img'

    print('Reading prompts from', args.prompt_file)
    prompt_file = args.prompt_file
    prompts = []
    with open(prompt_file, 'r') as f:
        for l in f.readlines():
            prompts.append(json.loads(l))
    [len(p['prompt'].split()) for p in prompts]

    # load pretrained model
    pipeline = DiffusionPipeline.from_pretrained(
                    pretrained,
                    revision=None,
                    variant=None,
                    torch_dtype=weight_dtype,
                )
    pipeline.scheduler = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler")

    # load attention processors
    print(f"Using { 'LoRA' if args.lora else 'base model'}.")
    file_prefix = 'base-'
    if args.lora:
        file_prefix = 'lora-'
        pipeline.load_lora_weights(output_dir)

    pipeline.to("cuda")

    # run inference
    for prompt in prompts:
        print(f"evaluating model on prompt: {prompt['prompt']}")
        fname = os.path.join(eval_dir, file_prefix+prompt['file'])
        image = pipeline(prompt['prompt'], num_inference_steps=30).images[0]
        image.save(fname)
        print(f"file saved at {fname}")
