#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --time=15:00:00
#SBATCH --output=job_output.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yeevelyn@stanford.edu

# load modules?
ml python/3.12
ml cuda/12.4
source /scratch/users/yeevelyn/stable-lora/env/bin/activate

# Set environment variables
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="train/vash"
export REF_IMG="/scratch/users/yeevelyn/stable-lora/reference_img/vash/cropped"
export NEW_PROMPT="Vash kneels on the ground in the desert, petting a dog."

# run command
accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$REF_IMG \
  --image_column="image" \
  --caption_column="caption" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=14000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --resume_from_checkpoint="latest" \
  --report_to=tensorboard \
  --checkpointing_steps=500 \
  --validation_prompt="$NEW_PROMPT" \
  --seed=1337