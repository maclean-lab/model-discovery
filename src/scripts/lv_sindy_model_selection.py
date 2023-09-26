import os.path
import itertools
import numbers
from argparse import ArgumentParser

import numpy as np
from numpy.random import default_rng
import pandas as pd
from pysindy import STLSQ
import h5py

from time_series_data import load_sample_from_h5
from dynamical_model_learning import NeuralDynamicsLearner
from dynamical_model_learning import OdeSystemLearner
from lotka_volterra_model import get_hybrid_dynamics


def search_time_steps(search_config, verbose=False):
    t_train_span = search_config['t_train_span']
    t_sindy_span = search_config['t_sindy_span']
    ts_learner = search_config['ts_learner']
    train_sample = search_config['train_sample']
    train_sample_size = len(train_sample)

    for t_step in search_config['t_ude_steps']:
        print(f'Generating UDE dynamics on t_step {t_step:.03f} for SINDy '
              'training...', flush=True)
        t_train = [np.arange(t_train_span[0], t_train_span[1] + 1e-8, t_step)
                   for _ in range(train_sample_size)]
        x0_train = [ts.x[0, :] for ts in train_sample]
        # generate UDE dynamics with slightly different initial conditions
        # if all initial conditions are the same
        if all(np.array_equal(x0_train[0], x0) for x0 in x0_train):
            x0_base = x0_train[0]
            x0_train = []
            for _ in range(train_sample_size):
                x0_noise_scale = search_config['rng'].normal(
                    scale=search_config['noise_level'], size=x0_base.shape)
                x0_noise_scale = np.clip(x0_noise_scale, -0.01, 0.01)
                x0_train.append(x0_base * (1 + x0_noise_scale))

        ts_learner.output_prefix = f't_step_{t_step:.2f}_ude'
        ts_learner.eval(t_eval=t_train, x0_eval=x0_train,
                        ref_data=train_sample, integrator_backend='scipy',
                        integrator_kwargs={'method': 'LSODA'},
                        sub_modules=['latent'], verbose=verbose,
                        show_progress=False)
        ts_learner.plot_pred_data(ref_data=train_sample)

        sindy_train_sample = []
        for ts, dx_pred in zip(ts_learner.pred_data,
                               ts_learner.sub_pred_data['latent']):
            ts = ts.copy()
            ts.dx = dx_pred

            # cut off some initial and final time points
            t_sindy_indices = np.where(
                (ts.t >= t_sindy_span[0]) & (ts.t <= t_sindy_span[1]))[0]
            if t_sindy_indices.size == 0:
                raise ValueError('No data for SINDy training on the given '
                                 'time span')
            ts = ts[t_sindy_indices]

            sindy_train_sample.append(ts)

        print('UDE dynamics generated')
        print_hrule()

        search_basis(search_config, sindy_train_sample, {'t_step': t_step},
                     verbose=verbose)


def search_basis(search_config, sindy_train_sample, model_info, verbose=False):
    for basis_name in search_config['basis_libs']:
        model_info_new = model_info.copy()
        model_info_new['basis'] = basis_name

        search_basis_normalization(search_config, sindy_train_sample,
                                   model_info_new, verbose=verbose)


def search_basis_normalization(search_config, sindy_train_sample, model_info,
                               verbose=False):
    for normalize_columns in [False, True]:
        model_info_new = model_info.copy()
        model_info_new['basis_normalized'] = normalize_columns

        search_optimizers(search_config, sindy_train_sample, model_info_new,
                          verbose=verbose)


def search_optimizers(search_config, sindy_train_sample, model_info,
                      verbose=False):
    for optimizer_name in search_config['sindy_optimizers']:
        model_info_new = model_info.copy()
        model_info_new['optimizer'] = optimizer_name

        search_optimizer_args(search_config, sindy_train_sample,
                              model_info_new, verbose=verbose)


def search_optimizer_args(search_config, sindy_train_sample, model_info,
                          verbose=False):
    sindy_optimizer_args = search_config['sindy_optimizer_args']
    optimizer_name = model_info['optimizer']
    arg_names = sindy_optimizer_args[optimizer_name].keys()
    arg_vals = sindy_optimizer_args[optimizer_name].values()

    for arg_vals in itertools.product(*arg_vals):
        model_info_new = model_info.copy()
        model_info_new['optimizer_args'] = dict(zip(arg_names, arg_vals))

        get_sindy_model(search_config, sindy_train_sample, model_info_new,
                        verbose=verbose)


def get_sindy_model(search_config, sindy_train_sample, model_info,
                    verbose=False):
    # unpack search config
    output_dir = search_config['output_dir']
    sindy_optimizers = search_config['sindy_optimizers']
    basis_libs = search_config['basis_libs']
    basis_strs = search_config['basis_strs']
    valid_sample = search_config['valid_sample']
    test_sample = search_config['test_sample']
    recovered_dynamics = search_config['recovered_dynamics']
    stop_events = search_config['stop_events']
    model_metrics = search_config['model_metrics']

    # unpack model info
    t_step = model_info['t_step']
    basis = model_info['basis']
    normalize_columns = model_info['basis_normalized']
    optimizer_name = model_info['optimizer']
    optimizer_args = model_info['optimizer_args']
    output_prefix = f't_step_{t_step:.2f}_pysindy_basis_{basis}_'
    if normalize_columns:
        output_prefix += 'normalized_'
    else:
        output_prefix += 'unnormalized_'
    for arg_name, arg_val in optimizer_args.items():
        if isinstance(arg_val, numbers.Number):
            output_prefix += f'opt_{optimizer_name}_{arg_name}_{arg_val:.2f}'
        else:
            output_prefix += f'opt_{optimizer_name}_{arg_name}_{arg_val}'

    print('Running SINDy with the following settings:', flush=True)
    print(f't_step = {t_step:.2f}', flush=True)
    print(f'basis = {basis}', end='', flush=True)
    if normalize_columns:
        print(' (normalized)', flush=True)
    else:
        print(' (unnormalized)', flush=True)
    print(f'optimizer = {optimizer_name}', flush=True)
    print(f'optimizer_args = {optimizer_args}', flush=True)

    # run SINDy with given settings
    eq_learner = OdeSystemLearner(sindy_train_sample, output_dir,
                                  output_prefix)
    eq_learner.train(optimizer_type=sindy_optimizers[optimizer_name],
                     threshold=0.1, learn_dx=True,
                     normalize_columns=normalize_columns,
                     basis_funcs=basis_libs[basis],
                     basis_names=basis_strs[basis],
                     optimizer_kwargs=optimizer_args,
                     valid_data=valid_sample,
                     valid_kwargs={
                         'eval_func': recovered_dynamics,
                         'integrator_kwargs': {'args': ('model', ),
                                               'events': stop_events}},
                     verbose=verbose)
    print('SINDy learning finished', flush=True)

    # gather results
    model_info['optimizer_args'] = str(optimizer_args)[1:-1]
    model_info['mse'] = eq_learner.valid_metrics['mse']
    model_info['aicc'] = eq_learner.valid_metrics['aicc']
    for i in range(eq_learner.num_vars):
        model_info[f'indiv_mse[{i}]'] = \
            eq_learner.valid_metrics['indiv_mse'][i]
        model_info[f'indiv_aicc[{i}]'] = \
            eq_learner.valid_metrics['indiv_aicc'][i]
    model_info['recovered_eqs'] = '\n'.join(
        [f'(x{j})\' = {eq}' for j, eq in enumerate(
            eq_learner.model.equations())])
    model_metrics.loc[len(model_metrics)] = model_info
    print('Saved metrics of the learned SINDy model', flush=True)

    # evaluate model on test sample
    eq_learner.eval(eval_data=test_sample, eval_func=recovered_dynamics,
                    integrator_kwargs={'args': (eq_learner.model, ),
                                       'events': stop_events},
                    verbose=False)
    eq_learner.plot_pred_data(output_suffix='pred_data_long')
    print('Saved plots of dynamics predicted by the learned SINDy model',
          flush=True)

    print_hrule()


def get_args():
    """Get arguments from command line."""
    arg_parser = ArgumentParser(
        description='Find the best ODE model that fits noisy data from the '
        'Lotka-Volterra model.')
    arg_parser.add_argument('--noise_level', type=float, default=0.01,
                            help='Noise level of training data')
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
    arg_parser.add_argument('--t_sindy_span', nargs=2, type=float,
                            default=[0.5, 3.5], metavar=('T0', 'T_END'),
                            help='Time span of SINDy training data')
    arg_parser.add_argument('--verbose', action='store_true',
                            help='Print output for training progress')

    return arg_parser.parse_args()


# set up boundary check for integration
def check_lower_bound(t, x, model):
    return float(np.all(x > -10.0))


def check_upper_bound(t, x, model):
    return float(np.all(x < 20.0))


def print_hrule():
    print('\n' + '=' * 60 + '\n', flush=True)


def main():
    args = get_args()
    verbose = args.verbose

    # initialize model and data
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
    train_sample = load_sample_from_h5(data_fd, 'train')
    valid_sample = load_sample_from_h5(data_fd, 'valid')
    test_sample = load_sample_from_h5(data_fd, 'test')
    data_fd.close()
    print('Data loaded', flush=True)
    param_str = ', '.join(str(p) for p in params_true)
    print(f'True parameter value: [{param_str}]', flush=True)
    t_span_str = ', '.join(str(t) for t in t_train_span)
    print(f'Time span of training data: ({t_span_str})', flush=True)
    print('Training sample size:', len(train_sample), flush=True)
    print('Validation sample size:', len(valid_sample), flush=True)
    print('Test sample size:', len(valid_sample), flush=True)

    print_hrule()

    # load learned UDE model
    print('Loading UDE model with lowest validation loss...', flush=True)
    print('Network architecture:', flush=True)
    print('- Hidden neurons:', args.num_hidden_neurons, flush=True)
    print('- Activation function:', args.activation, flush=True)
    output_dir = os.path.join(
        project_root, 'outputs', f'lv-{int(t_train_span[1])}s-ude-')
    output_dir += '-'.join(str(i) for i in args.num_hidden_neurons)
    output_dir += f'-{args.activation}'
    output_dir = os.path.join(
        output_dir,
        f'noise-{noise_level:.3f}-seed-{seed:04d}-ude-model-selection')

    if not os.path.exists(output_dir):
        print('No UDE model found for the given architecture', flush=True)
        print('Terminating...', flush=True)

        return

    # get the UDE model with lowest validation loss
    ude_model_metric = pd.read_csv(
        os.path.join(output_dir, 'ude_model_metrics.csv'), index_col=False)
    best_ude_row = ude_model_metric['best_valid_loss'].idxmin()
    learning_rate = ude_model_metric.loc[best_ude_row, 'learning_rate']
    window_size = ude_model_metric.loc[best_ude_row, 'window_size']
    batch_size = ude_model_metric.loc[best_ude_row, 'batch_size']
    best_epoch = ude_model_metric.loc[best_ude_row, 'best_epoch']
    output_prefix = f'lr_{learning_rate:.3f}_window_size_{window_size:02d}'
    output_prefix += f'_batch_size_{batch_size:02d}'
    hybrid_dynamics = get_hybrid_dynamics(
        growth_rates, args.num_hidden_neurons, args.activation)
    ts_learner = NeuralDynamicsLearner(train_sample, output_dir, output_prefix)
    ts_learner.load_model(
        hybrid_dynamics, output_suffix=f'model_state_epoch_{best_epoch:03d}')

    print('UDE model loaded', flush=True)

    # set up for SINDy model selection
    output_dir = os.path.join(
        os.path.split(output_dir)[0],
        f'noise-{noise_level:.3f}-seed-{seed:04d}-sindy-model-selection-test')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ts_learner.output_dir = output_dir
    search_config = {}
    search_config['noise_level'] = noise_level
    search_config['output_dir'] = output_dir
    search_config['t_train_span'] = t_train_span
    search_config['t_sindy_span'] = tuple(args.t_sindy_span)
    search_config['ts_learner'] = ts_learner
    search_config['train_sample'] = train_sample
    search_config['valid_sample'] = valid_sample
    search_config['test_sample'] = test_sample
    search_config['t_ude_steps'] = [0.1, 0.05]
    search_config['basis_libs'] = {
        'default': None,
        'higher_order': [
            lambda x: x ** 2, lambda x, y: x * y, lambda x: x ** 3,
            lambda x, y: x * x * y, lambda x, y: x * y * y
        ]
    }
    search_config['basis_strs'] = {
        'default': None,
        'higher_order': [
            lambda x: f'{x}^2', lambda x, y: f'{x} {y}', lambda x: f'{x}^3',
            lambda x, y: f'{x}^2 {y}', lambda x, y: f'{x} {y}^2'
        ]
    }
    search_config['sindy_optimizers'] = {'stlsq': STLSQ}
    search_config['sindy_optimizer_args'] = {
        'stlsq': {'alpha': [0.05, 0.1, 0.5, 1.0, 5.0, 10.0]}}
    search_config['rng'] = default_rng(seed)

    search_config['model_metrics'] = pd.DataFrame(
        columns=[
            't_step', 'basis', 'basis_normalized', 'optimizer',
            'optimizer_args', 'mse', 'indiv_mse[0]', 'indiv_mse[1]', 'aicc',
            'indiv_aicc[0]', 'indiv_aicc[1]', 'recovered_eqs'
        ])

    # define recovered hybrid dynamics
    def recovered_dynamics(t, x, model):
        return growth_rates * x + model.predict(x[np.newaxis, :])[0]

    search_config['recovered_dynamics'] = recovered_dynamics

    check_lower_bound.terminal = True
    check_upper_bound.terminal = True
    search_config['stop_events'] = [check_upper_bound, check_lower_bound]

    print('SINDy model selection setup finished', flush=True)
    t_span_str = ', '.join(str(t) for t in search_config['t_sindy_span'])
    print(f'Time span for SINDy training: {t_span_str}', flush=True)
    print_hrule()

    search_time_steps(search_config, verbose=verbose)
    print('Search completed', flush=True)

    search_config['model_metrics'].to_csv(
        os.path.join(output_dir, 'sindy_model_metrics.csv'), index=False)
    print('Model metrics saved.', flush=True)


if __name__ == '__main__':
    main()
