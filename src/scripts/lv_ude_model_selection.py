import os.path
import itertools
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from torch import nn
import h5py

from time_series_data import load_sample_from_h5
from dynamical_model_learning import NeuralDynamicsLearner
from lotka_volterra_model import get_hybrid_dynamics


def get_args():
    """Get arguments from command line."""
    arg_parser = ArgumentParser(
        description='Find the best UDE model that fits noisy data from the '
        'Lotka-Volterra model.')
    arg_parser.add_argument('--noise_level', type=float, default=0.01,
                            help='Noise level for generating training data')
    arg_parser.add_argument('--seed', type=int, default=2023,
                            help='Random seed of generated data')
    arg_parser.add_argument('--num_hidden_neurons', nargs='+', type=int,
                            default=[5, 5],
                            help='Number of neurons in each hidden layer of '
                            'the latent network in the UDE model')
    arg_parser.add_argument('--activation', type=str, default='tanh',
                            choices=['tanh', 'relu', 'rbf'],
                            help='Activation function for the latent network'
                            ' in the UDE model')
    arg_parser.add_argument('--learning_rates', nargs='+', type=float,
                            default=[1e-1, 1e-2, 1e-3],
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
    arg_parser.add_argument('--compile_model', action='store_true',
                            help='Compile the UDE model before training')
    arg_parser.add_argument('--verbose', action='store_true',
                            help='Print output for training progress')

    return arg_parser.parse_args()


def main():
    args = get_args()
    verbose = args.verbose

    # load data
    noise_level = args.noise_level
    seed = args.seed
    print(f'Loading data with {noise_level} noise level and seed '
          f'{seed:04d}...', flush=True)
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(
        project_root, 'data',
        f'lv_noise_{noise_level:.03f}_seed_{seed:04d}.h5')
    data_fd = h5py.File(data_path, 'r')
    params_true = data_fd.attrs['param_values']
    growth_rates = np.array([params_true[0], -params_true[3]])
    t_train_span = data_fd['train'].attrs['t_span']
    train_sample = load_sample_from_h5(data_fd, 'train')[:50]
    valid_sample = load_sample_from_h5(data_fd, 'valid')
    data_fd.close()
    print('Data loaded', flush=True)
    param_str = ', '.join(str(p) for p in params_true)
    print(f'True parameter value: [{param_str}]', flush=True)
    t_span_str = ', '.join(str(t) for t in t_train_span)
    print(f'Time span of training data: ({t_span_str})', flush=True)
    print('Training sample size:', len(train_sample), flush=True)
    print('Validation sample size:', len(valid_sample), flush=True)

    stdout_delim = '\n' + '=' * 60 + '\n'
    print(stdout_delim, flush=True)

    # set up for output files
    print('Setting up training...', flush=True)
    output_dir = os.path.join(
        project_root, 'outputs', f'lv-{int(t_train_span[1])}s-ude-')
    output_dir += '-'.join(str(i) for i in args.num_hidden_neurons)
    output_dir += f'-{args.activation}'
    output_dir = os.path.join(
        output_dir,
        f'noise-{noise_level:.3f}-seed-{seed:04d}-ude-model-selection')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # set up for training
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam
    learning_rates = args.learning_rates
    window_sizes = args.window_sizes
    batch_sizes = args.batch_sizes
    num_epochs = args.num_epochs
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
    print(stdout_delim, flush=True)

    # train for different combinations of hyperparameters
    for lr, ws, bs in itertools.product(learning_rates, window_sizes,
                                        batch_sizes):
        print('Learning UDE with the following settings:', flush=True)
        print(f'- Learning rate: {lr:.3f}', flush=True)
        print(f'- Window size: {ws}', flush=True)
        print(f'- Batch size: {bs}', flush=True)
        print(f'- Number of epochs: {num_epochs}', flush=True)

        # train the model
        hybrid_dynamics = get_hybrid_dynamics(
            growth_rates, num_hidden_neurons=args.num_hidden_neurons,
            activation=args.activation, compile_model=args.compile_model)
        output_prefix = f'lr_{lr:.3f}_window_size_{ws:02d}_batch_size_{bs:02d}'
        ts_learner = NeuralDynamicsLearner(train_sample, output_dir,
                                           output_prefix)
        ts_learner.train(hybrid_dynamics, loss_func, optimizer, lr, ws, bs,
                         num_epochs, seed=seed, valid_data=valid_sample,
                         valid_kwargs={'solver_backend': 'scipy',
                                       'verbose': False},
                         save_epoch_model=True, verbose=verbose,
                         show_progress=verbose)
        ts_learner.plot_training_losses(output_suffix='training_losses')
        print('\nTraining of all epochs finished', flush=True)

        # save model metrics
        best_epoch = np.argmin([m['mse'] for m in ts_learner.valid_metrics])
        best_loss = ts_learner.valid_metrics[best_epoch]['mse']
        model_info = {'learning_rate': lr, 'window_size': ws, 'batch_size': bs,
                      'best_epoch': best_epoch, 'best_valid_loss': best_loss}
        model_metrics.loc[len(model_metrics)] = model_info
        print('Saved metrics of the best model', flush=True)

        # evaluate the best model on training data
        best_epoch_model_suffix = f'model_state_epoch_{best_epoch:03d}'
        ts_learner.load_model(hybrid_dynamics, best_epoch_model_suffix)
        ts_learner.eval(eval_data=train_sample, sub_modules=['latent'],
                        solver_backend='scipy',
                        solver_kwargs={'method': 'LSODA'},
                        show_progress=False)
        ts_learner.plot_pred_data()
        print('Saved plots of dynamics predicted by the best model',
              flush=True)

        print(stdout_delim, flush=True)

    print('Finished training for all combinations of hyperparameters',
          flush=True)
    model_metrics_path = os.path.join(output_dir, 'ude_model_metrics.csv')
    model_metrics.to_csv(model_metrics_path, index=False)
    print('Saved all model metrics', flush=True)


if __name__ == '__main__':
    main()
