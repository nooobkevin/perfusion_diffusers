export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./output/tom2"
export INSTANCE_DIR="/nfs/node4/heichtai/Perfusion/data"

# Training on human faces
# For fine-tuning on human faces we found the following configuration to work better: learning_rate=5e-6, max_train_steps=1000 to 2000, and freeze_model=crossattn with at least 15-20 images.
# To collect the real images use this command first before training.
# pip install clip-retrieval
# python retrieve.py --class_prompt person --class_data_dir real_reg/samples_person --num_class_images 200

# Removed --freeze_model=crossattn since can't get it running 

accelerate launch train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=./real_reg/samples_person/ \
  --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
  --class_prompt="face" --num_class_images=200 \
  --instance_prompt="Face of <Tom-face>"  \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=5e-6  \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --scale_lr --hflip \
  --modifier_token "<Tom-face>" \
  --report_to="wandb" \
  --no_safe_serialization \
  --enable_xformers_memory_efficient_attention 