#!/bin/bash
export CUDA_VISIBLE_DEVICES=0   # or e.g. 0,1,2,3
NUM_GPUS=`python -c "import torch; print(torch.cuda.device_count(), end=\"\")"`
DATA_PATH="./output"

python3 main_text2image.py \
  "cagliostrolab/animagine-xl-4.0" \
  dataset/animaginexl4_prompts4Quant.csv \
  dataset/animaginexl4_prompts4Quant.csv \
  --scheduler ode \
  --guidance_scale 5.0 \
  --calibration_nsamples=256 \
  --evaluation_nsamples=32 \
  --finetune_nsamples=300 \
  --xtx_batch_size=4 \
  --num_inference_steps=32 \
  --relative_mse_tolerance=0.01 \
  --dtype float16 \
  --finetune_lr=0.00001 \
  --finetune_relative_mse_tolerance=0.001 \
  --num_codebooks=4 \
  --nbits_per_codebook=8 \
  --out_group_size=1 \
  --in_group_size=8 \
  --beam_size=8\
  --finetune_batch_size=2 \
  --local_batch_size=1 \
  --finetune_max_epoch=1 \
  --print_frequency=1 \
  --num_intermediate_finetunes=0 \
  --finetune_method=teacher \
  --snapshot_step=1 \
  --eval_step=1 \
  --group_channels \
  --save=$DATA_PATH \
  --resume
