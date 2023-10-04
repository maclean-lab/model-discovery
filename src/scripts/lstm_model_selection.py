import os.path
import itertools
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from torch import nn
import h5py

from time_series_data import load_sample_from_h5
from dynamical_model_learning import NeuralTimeSeriesLearner
from model_helpers import get_model_prefix, print_hrule


class LstmDynamics(nn.Module):
    def __init__(self, num_vars, window_size, hidden_size, num_layers):
        super().__init__()

        self.window_size = window_size
        self.hidden_size = hidden_size
        self.lstm_output_size = hidden_size * (window_size - 1)

        self.lstm = nn.LSTM(num_vars, hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.linear = nn.Linear(self.lstm_output_size, num_vars)

    def forward(self, t, x):
        batch_size = x.shape[0]
        h_n = torch.zeros(1, batch_size, self.hidden_size)
        c_n = torch.zeros(1, batch_size, self.hidden_size)

        x, _ = self.lstm(x, (h_n, c_n))
        x = self.linear(x.reshape(batch_size, self.lstm_output_size))

        return x


def get_args():
    arg_parser = ArgumentParser(
        description='Denoise time series data using LSTM')
    arg_parser.add_argument('--model', type=str, required=True,
                            choices=['lotka_volterra', 'repressilator'],
                            help='Dynamical model from which data is '
                            'generated')
    arg_parser.add_argument('--noise_level', type=float, required=True,
                            help='Noise level for generating training data')
    arg_parser.add_argument('--seed', type=int, default=2023,
                            help='Random seed of generated data')
    arg_parser.add_argument('--num_hidden_features', type=int, default=8,
                            help='Number of features in each hidden LSTM '
                            'layer')
    arg_parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of LSTM layers')
    arg_parser.add_argument('--learning_rates', nargs='+', type=float,
                            default=[1e-3, 1e-2, 1e-1],
                            help='Learning rates to search over')
    arg_parser.add_argument('--window_sizes', nargs='+', type=int,
                            default=[3, 5, 7], help='Window sizes for LSTM')
    arg_parser.add_argument('--batch_sizes', nargs='+', type=int,
                            default=[5, 10, 20],
                            help='Batch sizes to search over')
    arg_parser.add_argument('--num_epochs', type=int, default=10,
                            help='Number of epochs to train for each '
                            'combination of hyperparameters')
    arg_parser.add_argument('--verbose', action='store_true',
                            help='Print output for training progress')

    return arg_parser.parse_args()


def main():
    args = get_args()
    verbose = args.verbose

    # load data
    model_prefix = get_model_prefix(args.model)
    noise_level = args.noise_level
    seed = args.seed
    print('Loading data...', flush=True)
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(
        project_root, 'data',
        f'{model_prefix}_noise_{noise_level:.03f}_seed_{seed:04d}_raw.h5')
    data_fd = h5py.File(data_path, 'r')
    params_true = data_fd.attrs['param_values']
    x0_true = data_fd.attrs['x0']
    t_spans = {}
    t_steps = {}
    samples = {}
    for dataset_type in ['train', 'valid', 'test']:
        t_spans[dataset_type] = data_fd[dataset_type].attrs['t_span']
        t_steps[dataset_type] = data_fd[dataset_type].attrs['t_step']
        samples[dataset_type] = load_sample_from_h5(data_fd, dataset_type)
    num_vars = samples['train'][0].num_vars
    print('Data loaded:', flush=True)
    print(f'- Model: {args.model}', flush=True)
    param_str = ', '.join(str(p) for p in params_true)
    print(f'- True parameter value: [{param_str}]', flush=True)
    print(f'- Noise level: {noise_level}', flush=True)
    print(f'- RNG seed: {seed}', flush=True)
    t_span_str = ', '.join(str(t) for t in t_spans['train'])
    print(f'- Time span of training data: ({t_span_str})', flush=True)
    print(f'- Training sample size: {len(samples["train"])}', flush=True)
    print(f'- Validation sample size: {len(samples["valid"])}', flush=True)

    print_hrule()

    # set up for output files
    print('Setting up training...', flush=True)
    output_dir = os.path.join(
        project_root, 'outputs',
        f'{model_prefix}-{int(t_spans["train"][1])}s-lstm')
    output_dir += f'-{args.num_hidden_features}-{args.num_layers}'
    output_dir = os.path.join(
        output_dir, f'noise-{noise_level:.3f}-seed-{seed:04d}-model-selection')

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
    print('- Hidden features:', args.num_hidden_features, flush=True)
    print('Hyperparameters to search over:', flush=True)
    print('- Learning rates:', learning_rates, flush=True)
    print('- Window sizes:', window_sizes, flush=True)
    print('- Batch sizes:', batch_sizes, flush=True)
    print('Number of epochs:', num_epochs, flush=True)
    print('Loss functions: MSE', flush=True)
    print('Optimizer: Adam', flush=True)

    print_hrule()

    # train for different combinations of hyperparameters
    for lr, ws, bs in itertools.product(learning_rates, window_sizes,
                                        batch_sizes):
        print('Learning UDE with the following settings:', flush=True)
        print(f'- Learning rate: {lr:.3f}', flush=True)
        print(f'- Window size: {ws}', flush=True)
        print(f'- Batch size: {bs}', flush=True)
        print(f'- Number of epochs: {num_epochs}', flush=True)

        lstm_dynamics = LstmDynamics(num_vars, ws, args.num_hidden_features,
                                     args.num_layers)
        output_prefix = f'lr_{lr:.3f}_window_size_{ws:02d}_batch_size_{bs:02d}'
        ts_learner = NeuralTimeSeriesLearner(samples['train'], output_dir,
                                             output_prefix)
        input_mask = torch.full((ws, ), True, dtype=torch.bool)
        input_mask[ws // 2] = False
        ts_learner.train(lstm_dynamics, loss_func, optimizer, lr, ws, bs,
                         num_epochs, input_mask, seed=seed,
                         valid_data=samples['valid'],
                         valid_kwargs={'method': 'rolling'},
                         save_epoch_model=True,
                         verbose=verbose, show_progress=verbose)
        ts_learner.plot_training_losses(output_suffix='lstm_training_losses')
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
            ts_learner.load_model(lstm_dynamics, best_epoch_model_suffix)
            ts_learner.eval(eval_data=samples['train'], method='rolling',
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
    model_metrics_path = os.path.join(output_dir, 'lstm_model_metrics.csv')
    model_metrics.to_csv(model_metrics_path, index=False)
    print('Saved all model metrics', flush=True)

    print_hrule()

    # export smooth data from best model
    print('Exporting smoothed data from the best model...', flush=True)
    best_row = model_metrics['best_valid_loss'].idxmin()
    learning_rate = model_metrics.loc[best_row, 'learning_rate']
    window_size = model_metrics.loc[best_row, 'window_size']
    batch_size = model_metrics.loc[best_row, 'batch_size']
    best_epoch = model_metrics.loc[best_row, 'best_epoch']
    lstm_dynamics = LstmDynamics(num_vars, window_size,
                                 args.num_hidden_features, args.num_layers)
    best_model_prefix = f'lr_{learning_rate:.3f}_window_size_{window_size:02d}'
    best_model_prefix += f'_batch_size_{batch_size:02d}'
    best_model_suffix = f'model_state_epoch_{best_epoch:03d}'
    ts_learner = NeuralTimeSeriesLearner(samples['train'], output_dir,
                                         best_model_prefix)
    input_mask = torch.full((window_size, ), True, dtype=torch.bool)
    input_mask[window_size // 2] = False
    ts_learner.load_model(lstm_dynamics, output_suffix=best_model_suffix,
                          input_mask=input_mask)
    ts_learner.eval(eval_data=samples['train'], method='rolling',
                    show_progress=False)
    output_data_path = f'{model_prefix}_noise_{noise_level:.03f}'
    output_data_path += f'_seed_{seed:04d}'
    output_data_path += f'_lstm_{args.num_hidden_features}_{args.num_layers}'
    output_data_path += '.h5'
    output_data_path = os.path.join(project_root, 'data', output_data_path)
    with h5py.File(output_data_path, 'w') as fd:
        # save model parameters
        fd.attrs['noise_level'] = noise_level
        fd.attrs['param_values'] = params_true
        fd.attrs['x0'] = x0_true
        fd.attrs['rng_seed'] = seed

        for dataset_type in ['train', 'valid', 'test']:
            # create group for dataset
            data_group = fd.create_group(dataset_type)
            data_group.attrs['t_span'] = t_spans[dataset_type]
            data_group.attrs['t_step'] = t_steps[dataset_type]

            # save samples
            if dataset_type == 'train':
                sample_data = ts_learner.eval_data
            else:
                sample_data = samples[dataset_type]

            data_group.attrs['sample_size'] = len(sample_data)

            for idx, sample in enumerate(sample_data):
                sample_group = data_group.create_group(f'sample_{idx:04d}')
                sample_group.create_dataset('t', data=sample.t)
                sample_group.create_dataset('x', data=sample.x)

    print('Finished exporting smoothed data', flush=True)


if __name__ == "__main__":
    main()