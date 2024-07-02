# Data
This directory contains data files, mostly in HDF5 format.

## Format of filenames
Each HDF5 file has the filename of the following pattern:
```
{model}_{noise_type}_noise[_{noise_level}]_seed_{seed}_{variant}.h5
```

### Parameters
- `model`: dynamical model from which the data is generated, which can be one
  of the following:
  - `lv`: Lotka-Volterra
  - `rep`: repressilator
  - `emt`: epithelial-mesenchymal transition (EMT)
- `noise_type`: type of noise, which can be one of the following:
  - `additive`: additive noise
  - `multiplicative `: multiplicative noise
  - `fixed`: noise with fixed variance
- `noise_level`: noise level in decimal, e.g. 0.05. Omitted for fixed noise.
- `seed`: seed for random number generator from which the data is generated.
- `variant`: variant of the data, which can be one of the following:
  - `raw`: noise added at all time points
  - `clean_x0`: noise added at all time points except the first

### EMT data
There are two `csv` files for the EMT data:
- `emt_mean.csv`: mean proportions of cells in epithelial, intermediate, and
  mesenchymal states at each pseudotime point.
- `emt_std.csv`: standard deviations of the proportions of cells in epithelial,
  intermediate, and mesenchymal states at each pseudotime point.

## File structure
Each HDF5 file contains the following top-level attributes:
- `noise_type`: type of noise, which can be `additive`, `multiplicative`, or
  `fixed`.
- `noise_level`: noise level in decimal, e.g. 0.05. Set to `fixed` for fixed
  noise.
- `rng_seed`: seed for random number generator from which the data is
  generated.
- `param_values`: parameter values used to generate the data for known models
  (Lotka-Volterra, repressilator).
- `x0`: initial condition
- `clean_x0`: whether the initial condition is noise-free or not. `0` if
  noise-free, `1` otherwise.

Each HDF5 file contains the following top-level data groups:
- `train`: training data
- `valid`: validation data
- `test`: test data

Each of the three data groups above is organized as follows:
- Attributes:
  - `t_span`: time span of all time series samples in the dataset, e.g.
    `[0, 10]`
  - `t_step`: time step of all time series samples in the dataset, e.g. `1.0`
  - `num_samples`: number of time series samples
- Data groups: there are `num_samples` data groups, each corresponding to a
  time series sample. Each sample is named by `sample_{i}`, where `i` is the
  index of the sample, and has the following datasets:
  - `t`: time points of shape `(num_time_points, )`
  - `x`: values of all state variables at each time point of shape
    `(num_time_points, num_states)`
