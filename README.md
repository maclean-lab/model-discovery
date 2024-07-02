# Data-driven model discovery and model selection for noisy biological systems

## Table of contents
- [Project setup](#project-setup)
  - [Dependent packages](#dependent-packages)
  - [Setting up in terminal](#setting-up-in-terminal)
  - [Setting up in IDEs](#setting-up-in-ides)
- [Project content](#project-content)
- [Model discovery pipeline](#model-discovery-pipeline)
  - [Generate data](#generate-data)
  - [Learn ODEs with neural networks](#learn-odes-with-neural-networks)
  - [Infer ODEs using SINDy](#infer-odes-using-sindy)
  - [Plot data and results](#plot-data-and-results)
- [License](#license)

## Project setup
### Dependent packages
Install the dependent packages using `virtualenv` or `conda`.
- NumPy
- SciPy
- PyTorch
- torchdiffeq
- torchode
- PySINDy
- h5py
- Matplotlib
- seaborn
- tqdm
- Jupyter (only necessary for running code cells in IDEs)

### Setting up in terminal
To make sure that modules in the `src/utils` directory can be properly imported
when running scripts in bash or zsh, run the following command to set the
`PYTHONPATH` environment variable:
```
export PYTHONPATH=$(pwd)/src/utils:$PYTHONPATH
```

### Setting up in IDEs
In IDEs that recognize the `.env` file (e.g. Visual Studio Code with the Python
extension installed), run the following command in the terminal at project root
to set the `PYTHONPATH` in the `.env` file:
```
echo PYTHONPATH=\"$(pwd)/src/utils\" > .env
```
For Visual Studio Code, the `.env` file created by the above command should be
recognized by default. If not, find `python.envFile` in Settings and set it
to `${workspaceFolder}/.env`. In addition, add one of the following settings to
`${workspaceFolder}/.vscode/settings.json` depending on the operating system:
```
"terminal.integrated.env.linux": {
  "PYTHONPATH": "${workspaceFolder}/src/utils"
},
"terminal.integrated.env.osx": {
  "PYTHONPATH": "${workspaceFolder}/src/utils"
},
```

## Project content
- `data/`: data files used for model discovery. See [here](data/README.md) for
  more details.
- `src/`: source code.
  - `utils/`: utility modules.
  - `scripts/`: scripts for various tasks in the model discovery pipeline.
- `outputs`: generated output files; automatically created when running certain
  scripts.

## Model discovery pipeline
### Generate data
All data files used in the paper are stored in the `data/` directory. They can
be generated using the following command:
```
bash scripts/generate_data.sh
```

To generate a single dataset and save in HDF5 format, run the following line:
```
python src/scripts/generate_data.py \
  --model {model} --noise_type {noise_type} --noise_level {noise_level} \
  --seed {seed} --data_source {data_source}
```
The arguments are:
  - `--model`: one of `lotka_volterra`, `repressilator`, or `emt`.
  - `--noise_type`: for `lotka_volterra` and `repressilator`, both `additive`
    and `multiplicative` are supported. For `emt`, only `fixed` is supported.
  - `--noise_level`: noise level in decimal, e.g. 0.05. Omitted for fixed
    noise.
  - `--seed`: seed for random number generator.
  - `--variant`: one of `raw` or `clean_x0`.

Additional useful arguments:
  - `--t_train_span`: time span of training data, e.g. `0 10`.
  - `--t_valid_span`: time span of validation data, e.g. `0 10`.
  - `--t_test_span`: time span of test data, e.g. `0 10`.

Find out more arguments by running `python src/scripts/generate_data.py -h`.

### Learn ODEs with neural networks
To learn ODEs from a dataset using a neural network, run the following script:
```
python src/scripts/ude_model_selection.py \
  --model {model} --noise_type {noise_type} --noise_level {noise_level} \
  --seed {seed} --data_source {data_source} --ude_rhs {ude_rhs} \
  --num_hidden_neurons {num_hidden_neurons} --activation {activation} \
  --learning_rates {learning_rates} --window_sizes {window_sizes} \
  --batch_sizes {batch_sizes} --num_epochs {num_epochs}
```
The arguments are:
  - `--model`: one of `lotka_volterra`, `repressilator`, or `emt`.
  - `--noise_type`: for `lotka_volterra` and `repressilator`, both `additive`
    and `multiplicative` are supported. For `emt`, only `fixed` is supported.
  - `--noise_level`: noise level in decimal, e.g. `0.05`. Omitted for fixed
    noise.
  - `--seed`: seed of random number generator from which the data is generated.
    The same seed is used for operations involving random numbers in this
    script, such as initializing neural network weights.
  - `--data_source`: one of `raw` or `clean_x0`.
  - `--ude_rhs`: either `nn` for pure neural network formulation or `hybrid`
    for hybrid formulation.
  - `--num_hidden_neurons`: number of hidden neurons in the neural network,
    e.g. `8 8` for two layers with 8 neurons each.
  - `--activation`: activation function of the neural network. Supported
    functions are `tanh`, `relu`, `rbf`, `sigmoid`, `softplus`, `identity`.
  - `--learning_rates`: learning rates for training the neural network, e.g.
    `0.001 0.01 0.1`.
  - `--window_sizes`: window sizes for training the neural network, e.g.
    `5 10`.
  - `--batch_sizes`: batch sizes for training the neural network, e.g. `8 16`.
  - `--num_epochs`: number of epochs for training the neural network, e.g.
    `10`.

Note that arguments related to data (such as `--noise_type`) must correspond to
data files that have been generated. The script will automatically search for
the data file matching the specified parameters.

Find out more arguments by running
`python src/scripts/ude_model_selection.py -h`.

The script creates a folder named after learning specification and saves
learning results into that folder. Let's say we learn hybrid ODEs with the
following specifications:
- Model: Lotka-Volterra
- Noise type: Additive
- Noise level: 0.05
- Seed: 2023
- Data source: raw
- Number of hidden neurons: 8
- Activation function: tanh
- Learning rates: 0.001 0.01 0.1
- Window sizes: 5 10
- Batch sizes: 5 10 20
- Number of epochs: 10

The learning results will be saved under
`outputs/lv-raw-hybrid-8-tanh/additive-noise-0.050-seed-2023-ude-model-selection/`.
The following files will be saved:
- `ude_model_metrics.csv`: metrics of all learned models.
- Snapshots of trained model after each epoch, e.g.
  `lr_0.010_window_size_10_batch_size_05_model_state_epoch_007.pt`.
- Plots of training losses, e.g.
  `lr_0.010_window_size_10_batch_size_05_training_losses.pdf`.
- Plots of trained dynamics, e.g.
  `lr_0.010_window_size_10_batch_size_05_pred_data.pdf`.

### Infer ODEs using SINDy
To infer the ODEs of a dataset from learned neural networks using SINDy, run
the following script:
```
python src/scripts/sindy_model_selection.py \
  --model {model} --noise_type {noise_type} --noise_level {noise_level} \
  --seed {seed} --data_source {data_source} --data_preprocessor ude \
  --ude_rhs {ude_rhs} --num_hidden_neurons {num_hidden_neurons} \
  --activation {activation} --sindy_t_span {sindy_t_span}
```
The arguments are:
  - `--model`: one of `lotka_volterra`, `repressilator`, or `emt`.
  - `--noise_type`: for `lotka_volterra` and `repressilator`, both `additive`
    and `multiplicative` are supported. For `emt`, only `fixed` is supported.
  - `--noise_level`: noise level in decimal, e.g. `0.05`. Omitted for fixed
    noise.
  - `--seed`: seed of random number generator from which the data is generated.
  - `--data_source`: one of `raw` or `clean_x0`.
  - `--data_preprocessor`: use `ude` here to infer ODEs from learned neural
    networks.
  - `--ude_rhs`: either `nn` for pure neural network formulation or `hybrid`
    for hybrid formulation.
  - `--num_hidden_neurons`: number of hidden neurons in the neural network,
    e.g. `8 8` for two layers with 8 neurons each.
  - `--activation`: activation function of the neural network. Supported
    functions are `tanh`, `relu`, `rbf`, `sigmoid`, `softplus`, `identity`.
  - `--sindy_t_span`: time span for SINDy, e.g. `2 8`.

Note that arguments related to data (such as `--noise_type`) must correspond to
data files that have been generated. The script will automatically search for
the data file matching the specified parameters. Also, the specified neural
networks must have been trained using the `ude_model_selection.py` script.
This script will automatically use the trained neural network with the lowest
validation loss.

A SINDy model can also be inferred directly from data using the following
command:
```
python src/scripts/sindy_model_selection.py \
  --model {model} --noise_type {noise_type} --noise_level {noise_level} \
  --seed {seed} --data_source {data_source} --data_preprocessor none \
  --sindy_t_span {sindy_t_span}
```

Find out more arguments by running
`python src/scripts/sindy_model_selection.py -h`.

The script creates a folder named after learning specification and saves
learning results into that folder. Let's say we infer ODEs using SINDy from the
following dataset and learned neural network in a hybrid formulation:
- Model: Lotka-Volterra
- Noise type: Additive
- Noise level: 0.05
- Seed: 2023
- Data source: raw
- Number of hidden neurons: 8
- Activation function: tanh

The learning results will be saved under
`outputs/lv-raw-hybrid-8-tanh/additive-noise-0.050-seed-2023-sindy-model-selection/`.
The following files will be saved:
- `sindy_model_metrics.csv`: metrics of all inferred models.
- `{sindy_params}_ude_pred_data.pdf`: plots of derivatives predicted by the
  hybrid ODE model, which are used by SINDy.
- `{sindy_params}_pred_data_long.pdf`: plots of predicted data from test data
  using the inferred SINDy model.

> **Note**: `sindy_params` in the filenames above refer to the hyperparameters
  used in SINDy model selection, such as time step for derivative estimation
  and choice of basis functions.

### Plot data and results
Data and results can be plotted using `src/scripts/plot_data.py`. Supported
plots include:
- Raw data
  - True data for Lotka-Volterra and repressilator models (leave `--noise_type`
    as `none`).
  - Noisy data for all models given `--noise_type`, `--noise_level`, `--seed`,
    `--data_source`.
- Base SINDy on noisy data
  - Derivatives estimated by finite differences (default method in PySINDy; add
    `--plot_dx`).
  - Trajectories predicted by SINDy (leave `--ude_rhs` as `none` and pass SINDy
    options).
  > **Note**: data needs be specified by `--model`, `--noise_type`,
    `--noise_level`, `--seed`, and `--data_source`.
- Pure NN and hybrid ODEs on noisy data
  - Trajectories predicted by neural networks (set `--ude_rhs` to `nn` or
    `hybrid`).
  - Derivatives predicted by neural networks (add `--plot_dx`).
    - For Lotka-Volterra and repressilator models, the true derivatives are
      plotted as well.
  - Trajectories predicted by SINDy (add SINDy options).
  > **Note**: data needs be specified by `--model`, `--noise_type`,
    `--noise_level`, `--seed`, and `--data_source`.

Find out more in the `get_args()` function in `src/scripts/plot_data.py`.

## License
This project is licensed under the terms of the MIT License.
