#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --time=1:00:00
#SBATCH --output=eval_output.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yeevelyn@stanford.edu

# load modules?
ml python/3.12
ml cuda/12.4
source /scratch/users/yeevelyn/stable-lora/env/bin/activate

# Set environment variables


# run command
python3 inference.py
python3 inference.py --lora
python3 inference.py --prompt_file eval/eval_prompts-nodesc.jsonl
python3 inference.py --lora --prompt_file eval/eval_prompts-nodesc.jsonl
python3 inference.py --prompt_file eval/eval_prompts-noname.jsonl
python3 inference.py --lora --prompt_file eval/eval_prompts-noname.jsonl