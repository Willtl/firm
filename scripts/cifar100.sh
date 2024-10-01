#!/bin/bash

# Number of trials
num_trial=5

# Loop through normal classes (0 to 19)
for normal_class in {0..19}; do
  # Loop through trials (1 to num_trial)
  for trial in $(seq 0 $((num_trial - 1))); do
    python main.py \
      --save_freq 10 \
      --workers 16 \
      --epochs 2000 \
      --batch_size 32 \
      --opt 'SGD' \
      --momentum 0.9 \
      --lr 0.01 \
      --lr_scheduler 'cosineannealinglr' \
      --lr_min 1e-7 \
      --weight_decay 0.0003 \
      --loss 'FIRMLoss' \
      --temperature 0.2 \
      --model 'resnet18' \
      --dataset 'cifar100' \
      --normal_class $normal_class \
      --gamma 0.0 \
      --augs 'cnr0.25+jitter_b0.4_c0.4_s0.4_h0.4_p1.0+blur_k3_s0.5_p0.75' \
      --shift_transform 'rot90 rot180 rot270' \
      --oe '' \
      --trial $trial \
      --verbose True \
      --test_verbose False \
      --seed $trial \
      --experiment_name "FIRM/cls${normal_class}" \
      --reproducible True \
      --gpu 0

    # Wait for the python command to finish before proceeding
    wait

    echo "Completed trial $trial for normal_class $normal_class"
  done
done
