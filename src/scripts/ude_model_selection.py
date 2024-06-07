import os.path
import itertools
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from torch import nn
import h5py
import matplotlib

from time_series_data import load_dataset_from_h5
from dynamical_model_learning import NeuralTimeSeriesLearner
from dynamical_model_learning import NeuralDynamicsLearner
from lstm_model_selection import LstmDynamics
from model_helpers import get_model_module, get_data_path, \
    get_model_selection_dir, print_hrule


def get_args():
    """Get arguments from command line."""
    arg_parser = ArgumentParser(
        description='Find the best UDE model that fits noisy data.')
    arg_parser.add_argument('--model', type=str, required=True,
                            choices=['lotka_volterra', 'repressilator', 'emt'],
                            help='Dynamical model from which data is '
                            'generated')
    arg_parser.add_argument('--noise_type', type=str, required=True,
                            choices=['fixed', 'additive', 'multiplicative'],
                            help='Type of noise added to data')
    arg_parser.add_argument('--noise_level', type=float, default=0.01,
                            help='Noise level for generating training data')
    arg_parser.add_argument('--seed', type=int, default=2023,
                            help='Random seed of generated data')
    arg_parser.add_argument('--data_source', type=str, required=True,
                            choices=['raw', 'clean_x0'],
                            help='Source of training data')
    arg_parser.add_argument('--data_preprocessor', type=str, default='none',
                            help='Preprocessing method for training data')
    arg_parser.add_argument('--ude_rhs', type=str, default='hybrid',
                            choices=['nn', 'hybrid'],
                            help='Form of the right-hand side of the ODE '
                            'system, either nn (a single neural network) or '
                            'hybrid (known dynamics + neural network)')
    arg_parser.add_argument('--num_hidden_neurons', nargs='+', type=int,
                            default=[5, 5],
                            help='Number of neurons in each hidden layer of '
                            'the neural network')
    arg_parser.add_argument('--activation', type=str, default='tanh',
                            choices=['tanh', 'relu', 'rbf', 'sigmoid',
                                     'softplus', 'identity'],
                            help='Activation function for the neural network')
    arg_parser.add_argument('--learning_rates', nargs='+', type=float,
                            default=[1e-3, 1e-2, 1e-1],
                            help='Learning rates to search over')
    arg_parser.add_argument('--window_sizes', nargs='+', type=int,
                            default=[5, 10],
                            help='Window sizes to search over')
    arg_parser.add_argument('--batch_sizes', nargs='+', type=int,
                            default=[5, 10, 20],
                            help='Batch sizes to search over')
    arg_parser.add_argument('--num_epochs', type=int, default=10,
                            help='Number of epochs to train for each '
                            'combination of hyperparameters')
    arg_parser.add_argument('--integrator_backend', type=str,
                            default='torchode',
                            choices=['torchdiffeq', 'torchode'],
                            help='Backend to use for ODE integration')
    arg_parser.add_argument('--torchode_step_method', type=str,
                            default='Dopri5', choices=['Dopri5', 'Tsit5'],
                            help='Step method to use for torchode')
    arg_parser.add_argument('--matplotlib_backend', type=str, default='Agg',
                            help='Matplotlib backend to use')
    arg_parser.add_argument('--verbose', action='store_true',
                            help='Print output for training progress')

    return arg_parser.parse_args()


def main():
    args = get_args()
    verbose = args.verbose
    matplotlib.use(args.matplotlib_backend)
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    # load data
    noise_type = args.noise_type
    noise_level = args.noise_level
    seed = args.seed
    data_source = args.data_source
    data_preprocessor = args.data_preprocessor
    print('Loading data...', flush=True)
    data_path = get_data_path(args.model, noise_type, noise_level, seed,
                              data_source)
    data_fd = h5py.File(data_path, 'r')
    params_true = data_fd.attrs['param_values']
    t_train_span = data_fd['train'].attrs['t_span']
    train_samples = load_dataset_from_h5(data_fd, 'train')
    valid_samples = load_dataset_from_h5(data_fd, 'valid')
    num_vars = train_samples[0].num_vars
    data_fd.close()

    # use learned LSTM model to preprocess training and validation data
    if data_preprocessor.startswith('lstm_'):
        # retrieve information about the best LSTM model
        lstm_config = {'num_hidden_features': data_preprocessor.split('_')[1],
                       'num_layers': data_preprocessor.split('_')[2]}
        lstm_output_dir = get_model_selection_dir(
            args.model, noise_type, noise_level, seed, data_source, 'lstm',
            'lstm', nn_config=lstm_config)
        lstm_metrics_path = os.path.join(
            lstm_output_dir, 'lstm_model_metrics.csv')
        lstm_model_metrics = pd.read_csv(lstm_metrics_path, index_col=False)
        if len(lstm_model_metrics) == 0:
            print('LSTM data preprocessor specified but no LSTM model was '
                  'successfully trained; will not train UDE models',
                  flush=True)
            return
        best_lstm_row = lstm_model_metrics['best_valid_loss'].idxmin()
        learning_rate = lstm_model_metrics.loc[best_lstm_row, 'learning_rate']
        window_size = lstm_model_metrics.loc[best_lstm_row, 'window_size']
        batch_size = lstm_model_metrics.loc[best_lstm_row, 'batch_size']
        best_epoch = lstm_model_metrics.loc[best_lstm_row, 'best_epoch']

        # preprocess data
        output_prefix = f'lr_{learning_rate:.3f}_window_size_{window_size:02d}'
        output_prefix += f'_batch_size_{batch_size:02d}'
        input_mask = torch.full((window_size, ), True, dtype=torch.bool)
        input_mask[window_size // 2] = False
        ts_learner = NeuralTimeSeriesLearner(train_samples, lstm_output_dir,
                                             output_prefix)
        lstm_dynamics = LstmDynamics(
            num_vars, window_size, lstm_config['num_hidden_features'],
            lstm_config['num_layers'])
        ts_learner.load_model(
            lstm_dynamics, output_suffix=f'model_state_epoch_{best_epoch:03d}',
            input_mask=input_mask)
        ts_learner.eval(eval_data=train_samples, method='rolling',
                        show_progress=False)
        train_samples = ts_learner.pred_data
        ts_learner.eval(eval_data=valid_samples, method='rolling',
                        show_progress=False)
        valid_samples = ts_learner.pred_data

    print('Data loaded:', flush=True)
    print(f'- Model: {args.model}', flush=True)
    param_str = ', '.join(str(p) for p in params_true)
    print(f'- True parameter value: [{param_str}]', flush=True)
    print(f'- Noise type: {noise_type}', flush=True)
    if noise_type != 'fixed':
        print(f'- Noise level: {noise_level}', flush=True)
    print(f'- RNG seed: {seed}', flush=True)
    print(f'- Data source: {data_source}', flush=True)
    print(f'- Data preprocessor: {data_preprocessor}', flush=True)
    t_span_str = ', '.join(str(t) for t in t_train_span)
    print(f'- Time span of training data: ({t_span_str})', flush=True)
    print(f'- Training dataset size: {len(train_samples)}', flush=True)
    print(f'- Validation dataset size: {len(valid_samples)}', flush=True)

    print_hrule()

    # set up for output files
    print('Setting up training...', flush=True)
    if data_preprocessor == 'none':
        pipeline = 'ude'
    else:
        pipeline = f'{data_preprocessor.replace("_", "-")}-ude'
    nn_config = {'num_hidden_neurons': args.num_hidden_neurons,
                 'activation': args.activation}
    output_dir = get_model_selection_dir(
        args.model, noise_type, noise_level, seed, data_source, pipeline,
        'ude', ude_rhs=args.ude_rhs, nn_config=nn_config)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # set up for training
    model_module = get_model_module(args.model)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam
    learning_rates = args.learning_rates
    window_sizes = args.window_sizes
    batch_sizes = args.batch_sizes
    num_epochs = args.num_epochs
    integrator_backend = args.integrator_backend
    model_metrics = pd.DataFrame(
        columns=['learning_rate', 'window_size', 'batch_size', 'best_epoch',
                 'best_valid_loss'])

    print('Training setup finished', flush=True)
    print('Network architecture:', flush=True)
    print('- Hidden neurons:', args.num_hidden_neurons, flush=True)
    print('- Activation function:', args.activation, flush=True)
    print('Hyperparameters to search over:', flush=True)
    print('- Learning rates:', learning_rates, flush=True)
    print('- Window sizes:', window_sizes, flush=True)
    print('- Batch sizes:', batch_sizes, flush=True)
    print('Loss functions: MSE', flush=True)
    print('Optimizer: Adam', flush=True)
    print('Number of epochs:', num_epochs, flush=True)
    print('Integrator backend:', integrator_backend, flush=True)

    print_hrule()

    # train for different combinations of hyperparameters
    for lr, ws, bs in itertools.product(learning_rates, window_sizes,
                                        batch_sizes):
        print('Learning data with the following settings:', flush=True)
        print(f'- Learning rate: {lr:.3f}', flush=True)
        print(f'- Window size: {ws}', flush=True)
        print(f'- Batch size: {bs}', flush=True)
        print(f'- Number of epochs: {num_epochs}', flush=True)
        torch.manual_seed(seed)

        # train the model
        if args.ude_rhs == 'nn':
            neural_dynamics = model_module.get_neural_dynamics(
                num_hidden_neurons=args.num_hidden_neurons,
                activation=args.activation)
        else:  # args.ude_rhs == 'hybrid'
            match args.model:
                case 'lotka_volterra':
                    growth_rates = np.array([params_true[0], -params_true[3]])
                    neural_dynamics = model_module.get_hybrid_dynamics(
                        growth_rates,
                        num_hidden_neurons=args.num_hidden_neurons,
                        activation=args.activation)
                case 'repressilator':
                    neural_dynamics = model_module.get_hybrid_dynamics(
                        num_hidden_neurons=args.num_hidden_neurons,
                        activation=args.activation)
                case 'emt':
                    growth_rates = np.array([-0.5, -0.15, -0.1])
                    neural_dynamics = model_module.get_hybrid_dynamics(
                        growth_rates,
                        num_hidden_neurons=args.num_hidden_neurons,
                        activation=args.activation)
        output_prefix = f'lr_{lr:.3f}_window_size_{ws:02d}_batch_size_{bs:02d}'
        ts_learner = NeuralDynamicsLearner(train_samples, output_dir,
                                           output_prefix)
        ts_learner.train(neural_dynamics, loss_func, optimizer, lr, ws, bs,
                         num_epochs, integrator_backend=integrator_backend,
                         torchode_step_method=args.torchode_step_method,
                         valid_data=valid_samples,
                         valid_kwargs={
                             'integrator_backend': 'scipy',
                             'integrator_kwargs': {'method': 'LSODA'},
                             'verbose': False},
                         save_epoch_model=True, verbose=verbose,
                         show_progress=verbose)
        ts_learner.plot_training_losses(output_suffix='training_losses')
        if not verbose:
            print('\nTraining finished', flush=True)

        # save model metrics
        if ts_learner.valid_metrics:
            best_epoch = np.argmin(
                [m['mse'] for m in ts_learner.valid_metrics])
            best_loss = ts_learner.valid_metrics[best_epoch]['mse']
            model_info = {'learning_rate': lr, 'window_size': ws,
                          'batch_size': bs, 'best_epoch': best_epoch,
                          'best_valid_loss': best_loss}
            model_metrics.loc[len(model_metrics)] = model_info
            print('Saved metrics of the best model', flush=True)

            # evaluate the best model on training data
            best_epoch_model_suffix = f'model_state_epoch_{best_epoch:03d}'
            ts_learner.load_model(neural_dynamics, best_epoch_model_suffix)
            ts_learner.eval(eval_data=train_samples,
                            integrator_backend='scipy',
                            integrator_kwargs={'method': 'LSODA'},
                            show_progress=False)
            ts_learner.plot_pred_data()
            print('Saved plots of dynamics predicted by the best model',
                  flush=True)
        else:
            print('Training stopped during epoch 0, no model metrics to save')

        print_hrule()

    # save model metrics for all combinations of hyperparameters
    print('Finished training for all combinations of hyperparameters',
          flush=True)
    model_metrics_path = os.path.join(output_dir, 'ude_model_metrics.csv')
    model_metrics.to_csv(model_metrics_path, index=False)
    print('Saved all model metrics', flush=True)


if __name__ == '__main__':
    main()
