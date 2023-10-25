# Data

## Format of filenames
Each HDF5 file has the filename of the following pattern:
```
{model}_{noise_type}_noise_{noise_level}_seed_{seed}_{source}.h5
```

### Details
- `model`: dynamical model from which the data is generated, either `lv` for
Lotka-Volterra or `rep` for repressilator
- `noise_type`: type of noise, either `additive` or `multiplicative`
- `noise_level`: noise level in decimal, e.g. 0.05
- `seed`: seed for random number generator
- `source`: source from which the data was produced
    - `raw`: noise added at all time points
    - `raw_lstm_{m}_{n}`: the above data denoised by LSTM which has `n` layers
    with `m` units
    - `clean_x0`: noise added at all time points except the first
    - `clean_x0_lstm_{m}_{n}`: the above data denoised by LSTM which has
    `n` layers with `m` units