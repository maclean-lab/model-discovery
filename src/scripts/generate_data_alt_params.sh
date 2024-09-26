#!/bin/bash
# conda activate diff-eq-learning

noise_types=("additive" "multiplicative")
noise_levels=(0.001 0.005 0.01 0.05 0.1 0.2)

# generate data for lotka-volterra model
echo "Generating data for Lotka-Volterra model..."
model="lotka_volterra"
x0="2.5 0.7"
param_values="2.1 1.2 0.7 1.1"
t_test_span="0 20"

for nt in "${noise_types[@]}"; do
    for nl in "${noise_levels[@]}"; do
        python src/scripts/generate_data.py \
            --model $model --x0 $x0 --param_values $param_values \
            --noise_type $nt --noise_level $nl --t_test_span $t_test_span \
            --output_suffix "alt_params"
    done
done

# generate data for repressilator model
echo "Generating data for repressilator model..."
model="repressilator"
t_test_span="0 30"
x0="1.5 1.5 1.5"
param_values="7.0 2 6.0 2 8.0 2"

for nt in "${noise_types[@]}"; do
    for nl in "${noise_levels[@]}"; do
        python src/scripts/generate_data.py \
            --model $model --x0 $x0 --param_values $param_values \
            --noise_type $nt --noise_level $nl --t_test_span $t_test_span \
            --output_suffix "alt_params"
    done
done

echo "Done."
