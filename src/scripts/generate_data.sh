#!/bin/bash
conda activate diff-eq-learning

noise_types=("additive" "multiplicative")
noise_levels=(0.001 0.005 0.01 0.05 0.1 0.2)

# generate data for lotka-volterra model
echo "Generating data for Lotka-Volterra model..."
model="lotka_volterra"
t_test_span="0 20"

for nt in "${noise_types[@]}"; do
    for nl in "${noise_levels[@]}"; do
        python src/scripts/generate_data.py \
            --model $model --noise_type $nt --noise_level $nl \
            --t_test_span $t_test_span
    done
done

# generate data for repressilator model
echo "Generating data for repressilator model..."
model="repressilator"
t_test_span="0 30"

for nt in "${noise_types[@]}"; do
    for nl in "${noise_levels[@]}"; do
        python src/scripts/generate_data.py \
            --model $model --noise_type $nt --noise_level $nl \
            --t_test_span $t_test_span
    done
done

# generate data for EMT model
echo "Generating data for EMT model..."
python src/scripts/generate_data.py \
    --model emt --noise_type fixed --t_train_span 1 9 --t_valid_span 1 9 \
    --t_test_span 0 13

echo "Done."
