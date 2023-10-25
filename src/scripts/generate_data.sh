#!/bin/bash
# conda activate diff-eq-learning

model="repressilator"
noise_types=("additive" "multiplicative")
noise_levels=(0.001 0.005 0.01 0.05 0.1 0.2)
t_test_span="0 30"

for nt in "${noise_types[@]}"; do
    for nl in "${noise_levels[@]}"; do
        python src/scripts/generate_data.py \
            --model $model \
            --noise_type $nt \
            --noise_level $nl \
            --clean_x0 \
            --t_test_span $t_test_span
    done
done
