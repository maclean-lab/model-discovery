import os.path
import itertools
import numbers
from argparse import ArgumentParser

import numpy as np
from numpy.random import default_rng
import pandas as pd
from pysindy import STLSQ
import h5py
import matplotlib

from time_series_data import load_sample_from_h5
from dynamical_model_learning import NeuralDynamicsLearner
from dynamical_model_learning import OdeSystemLearner
from model_helpers import get_model_module, get_model_prefix, print_hrule


def search_time_steps(search_config, verbose=False):
    t_sindy_span = search_config['t_sindy_span']
    ts_learner = search_config['ts_learner']
    train_sample = search_config['train_sample']
    ude_train_sample = search_config['ude_train_sample']
    train_sample_size = len(ude_train_sample)

    # evaluate on UDE training time span
    for t_step in search_config['t_ude_steps']:
        print(f'Generating UDE dynamics on t_step {t_step:.03f} for SINDy '
              'training...', flush=True)
        t_train = [np.arange(ts.t[0], ts.t[-1] + t_step * 1e-3, t_step)
                   for ts in ude_train_sample]
        x0_train = [ts.x[0, :] for ts in ude_train_sample]

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
    valid_kwargs = {
        'eval_func': recovered_dynamics,
        'integrator_kwargs': {
            'args': ('model', ),
            'method': 'LSODA',
            'events': stop_events,
            'min_step': 1e-8,
        },
    }
    eq_learner.train(optimizer_type=sindy_optimizers[optimizer_name],
                     threshold=search_config['stlsq_threshold'],
                     learn_dx=True, normalize_columns=normalize_columns,
                     basis_funcs=basis_libs[basis],
                     basis_names=basis_strs[basis],
                     optimizer_kwargs=optimizer_args, valid_data=valid_sample,
                     valid_kwargs=valid_kwargs, verbose=verbose)
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
    eval_integrator_kwargs = {'args': (eq_learner.model, ),
                              'method': 'LSODA',
                              'events': stop_events,
                              'min_step': 1e-8}
    eq_learner.eval(eval_data=test_sample, eval_func=recovered_dynamics,
                    integrator_kwargs=eval_integrator_kwargs,
                    verbose=verbose)
    eq_learner.plot_pred_data(output_suffix='pred_data_long')
    print('Saved plots of dynamics predicted by the learned SINDy model',
          flush=True)

    print_hrule()


def get_args():
    """Get arguments from command line."""
    arg_parser = ArgumentParser(
        description='Find the best ODE model that fits noisy data from the '
        'Lotka-Volterra model.')
    arg_parser.add_argument('--model', type=str, required=True,
                            choices=['lotka_volterra', 'repressilator'],
                            help='Dynamical model from which data is '
                            'generated')
    arg_parser.add_argument('--noise_type', type=str, required=True,
                            choices=['additive', 'multiplicative'],
                            help='Type of noise added to data')
    arg_parser.add_argument('--noise_level', type=float, required=True,
                            help='Noise level of training data')
    arg_parser.add_argument('--seed', type=int, default=2023,
                            help='Random seed of generated data')
    arg_parser.add_argument('--ude_data_source', type=str, default='raw',
                            help='Data source for UDE model training')
    arg_parser.add_argument('--num_hidden_neurons', nargs='+', type=int,
                            default=[5, 5],
                            help='Number of neurons in each hidden layer of '
                            'the latent network in the UDE model')
    arg_parser.add_argument('--activation', type=str, default='tanh',
                            choices=['tanh', 'relu', 'rbf'],
                            help='Activation function for the latent network'
                            ' in the UDE model')
    arg_parser.add_argument('--t_sindy_span', nargs=2, type=float,
                            default=[-np.inf, np.inf], metavar=('T0', 'T_END'),
                            help='Time span of SINDy training data')
    arg_parser.add_argument('--matplotlib_backend', type=str, default='Agg',
                            help='Matplotlib backend to use')
    arg_parser.add_argument('--verbose', action='store_true',
                            help='Print output for training progress')

    return arg_parser.parse_args()


# set up boundary check for integration
def get_lower_bound_checker(lower_bound):
    def check_lower_bound(t, x, model):
        return float(np.all(x > lower_bound))

    return check_lower_bound


def get_upper_bound_checker(upper_bound):
    def check_upper_bound(t, x, model):
        return float(np.all(x < upper_bound))

    return check_upper_bound


def main():
    # TODO: move certain parts to a separate function
    # E.g. data loading, UDE model loading, basis function setup
    args = get_args()
    matplotlib.use(args.matplotlib_backend)
    verbose = args.verbose

    # initialize model and data
    search_config = {}
    noise_type = args.noise_type
    noise_level = args.noise_level
    seed = args.seed
    print('Loading data...', flush=True)
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..'))
    model_prefix = get_model_prefix(args.model)
    data_path = os.path.join(
        project_root, 'data',
        f'{model_prefix}_{noise_type}_noise_{noise_level:.03f}'
        f'_seed_{seed:04d}_raw.h5')
    data_fd = h5py.File(data_path, 'r')
    params_true = data_fd.attrs['param_values']
    t_train_span = data_fd['train'].attrs['t_span']
    train_sample = load_sample_from_h5(data_fd, 'train')
    valid_sample = load_sample_from_h5(data_fd, 'valid')
    test_sample = load_sample_from_h5(data_fd, 'test')
    data_fd.close()
    print('Data loaded:', flush=True)
    print(f'- Model: {args.model}', flush=True)
    param_str = ', '.join(str(p) for p in params_true)
    print(f'- True parameter value: [{param_str}]', flush=True)
    print(f'- Noise type: {noise_type}', flush=True)
    print(f'- Noise level: {noise_level}', flush=True)
    print(f'- RNG seed: {seed}', flush=True)
    t_span_str = ', '.join(str(t) for t in t_train_span)
    print(f'- Time span of training data: ({t_span_str})', flush=True)
    print(f'- Training sample size: {len(train_sample)}', flush=True)
    print(f'- Validation sample size: {len(valid_sample)}', flush=True)
    print(f'- Test sample size: {len(valid_sample)}', flush=True)

    print_hrule()

    # load learned UDE model
    print('Loading UDE model with lowest validation loss...', flush=True)
    ude_data_source = args.ude_data_source
    print(f'UDE data source: {ude_data_source}', flush=True)
    print('Network architecture:', flush=True)
    print(f'- Hidden neurons: {args.num_hidden_neurons}', flush=True)
    print(f'- Activation function: {args.activation}', flush=True)
    output_dir = f'{model_prefix}-'
    if ude_data_source != 'raw':
        output_dir += ude_data_source.replace('_', '-')
        output_dir += '-ude-'
    else:
        output_dir += 'ude-'
    output_dir += '-'.join(str(i) for i in args.num_hidden_neurons)
    output_dir += f'-{args.activation}'
    output_dir = os.path.join(
        project_root, 'outputs', output_dir,
        f'{noise_type}-noise-{noise_level:.3f}-seed-{seed:04d}'
        f'-ude-model-selection')

    if not os.path.exists(output_dir):
        print('No UDE model found for the given architecture', flush=True)
        print('Terminating...', flush=True)

        return

    # get the UDE model with lowest validation loss
    ude_model_metric = pd.read_csv(
        os.path.join(output_dir, 'ude_model_metrics.csv'), index_col=False)
    if len(ude_model_metric) == 0:
        print('No UDE model was successfully trained; will not train SINDy'
              ' models', flush=True)
        return
    best_ude_row = ude_model_metric['best_valid_loss'].idxmin()
    learning_rate = ude_model_metric.loc[best_ude_row, 'learning_rate']
    window_size = ude_model_metric.loc[best_ude_row, 'window_size']
    batch_size = ude_model_metric.loc[best_ude_row, 'batch_size']
    best_epoch = ude_model_metric.loc[best_ude_row, 'best_epoch']
    output_prefix = f'lr_{learning_rate:.3f}_window_size_{window_size:02d}'
    output_prefix += f'_batch_size_{batch_size:02d}'
    model_module = get_model_module(args.model)
    match args.model:
        case 'lotka_volterra':
            growth_rates = np.array([params_true[0], -params_true[3]])
            hybrid_dynamics = model_module.get_hybrid_dynamics(
                growth_rates, args.num_hidden_neurons, args.activation)
        case 'repressilator':
            hybrid_dynamics = model_module.get_hybrid_dynamics(
                args.num_hidden_neurons, args.activation)
    ts_learner = NeuralDynamicsLearner(train_sample, output_dir, output_prefix)
    ts_learner.load_model(
        hybrid_dynamics, output_suffix=f'model_state_epoch_{best_epoch:03d}')

    # get training sample used by the UDE model
    if ude_data_source in ('raw', 'clean_x0'):
        search_config['ude_train_sample'] = train_sample
    else:
        ude_data_path = os.path.join(
            project_root, 'data',
            f'{model_prefix}_{noise_type}_noise_{noise_level:.03f}'
            f'_seed_{seed:04d}_{ude_data_source}.h5')
        data_fd = h5py.File(ude_data_path, 'r')
        search_config['ude_train_sample'] = load_sample_from_h5(data_fd,
                                                                'train')
        data_fd.close()

    print('UDE model loaded', flush=True)

    # set up for SINDy model selection
    output_dir = os.path.join(
        os.path.split(output_dir)[0],
        f'{noise_type}-noise-{noise_level:.3f}-seed-{seed:04d}'
        '-sindy-model-selection')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ts_learner.output_dir = output_dir
    search_config['noise_type'] = noise_type
    search_config['noise_level'] = noise_level
    search_config['output_dir'] = output_dir
    search_config['t_train_span'] = t_train_span
    t_sindy_span = args.t_sindy_span
    if t_sindy_span[0] < t_train_span[0]:
        t_sindy_span[0] = t_train_span[0]
    if t_sindy_span[1] > t_train_span[1]:
        t_sindy_span[1] = t_train_span[1]
    search_config['t_sindy_span'] = tuple(t_sindy_span)
    search_config['ts_learner'] = ts_learner
    search_config['train_sample'] = train_sample
    search_config['valid_sample'] = valid_sample
    search_config['test_sample'] = test_sample
    search_config['sindy_optimizers'] = {'stlsq': STLSQ}
    search_config['sindy_optimizer_args'] = {
        'stlsq': {'alpha': [0.05, 0.1, 0.5, 1.0, 5.0, 10.0]}}
    search_config['rng'] = default_rng(seed)

    # model specific setup
    match args.model:
        case 'lotka_volterra':
            search_config['stlsq_threshold'] = 0.1
            search_config['t_ude_steps'] = [0.1, 0.05]
            search_config['basis_libs'] = {
                'default': None,
                'higher_order': [
                    lambda x: x ** 2, lambda x, y: x * y, lambda x: x ** 3,
                    lambda x, y: x * x * y, lambda x, y: x * y * y
                ]}
            search_config['basis_strs'] = {
                'default': None,
                'higher_order': [
                    lambda x: f'{x}^2', lambda x, y: f'{x} {y}',
                    lambda x: f'{x}^3', lambda x, y: f'{x}^2 {y}',
                    lambda x, y: f'{x} {y}^2'
                ]
            }

            def recovered_dynamics(t, x, model):
                return growth_rates * x + model.predict(x[np.newaxis, :])[0]

            search_config['recovered_dynamics'] = recovered_dynamics
        case 'repressilator':
            search_config['stlsq_threshold'] = 1.0
            search_config['t_ude_steps'] = [0.2, 0.1]
            search_config['basis_libs'] = {
                'hill_1': [lambda x: 1.0 / (1.0 + x)],
                'hill_2': [lambda x: 1.0 / (1.0 + x ** 2)],
                'hill_3': [lambda x: 1.0 / (1.0 + x ** 3)],
                'hill_4': [lambda x: 1.0 / (1.0 + x ** 4)],
                'hill_max_3': [lambda x: 1.0 / (1.0 + x),
                               lambda x: 1.0 / (1.0 + x ** 2),
                               lambda x: 1.0 / (1.0 + x ** 3)],
                'hill_max_4': [lambda x: 1.0 / (1.0 + x),
                               lambda x: 1.0 / (1.0 + x ** 2),
                               lambda x: 1.0 / (1.0 + x ** 3),
                               lambda x: 1.0 / (1.0 + x ** 4)]
            }
            search_config['basis_strs'] = {
                'hill_1': [lambda x: f'/(1+{x})'],
                'hill_2': [lambda x: f'/(1+{x}^2)'],
                'hill_3': [lambda x: f'/(1+{x}^3)'],
                'hill_4': [lambda x: f'/(1+{x}^4)'],
                'hill_max_3': [lambda x: f'/(1+{x})',
                               lambda x: f'/(1+{x}^2)',
                               lambda x: f'/(1+{x}^3)'],
                'hill_max_4': [lambda x: f'/(1+{x})',
                               lambda x: f'/(1+{x}^2)',
                               lambda x: f'/(1+{x}^3)',
                               lambda x: f'/(1+{x}^4)']
            }

            def recovered_dynamics(t, x, model):
                return model.predict(x[np.newaxis, :])[0] - x

            search_config['recovered_dynamics'] = recovered_dynamics

    # initialize table for model metrics
    num_variables = train_sample[0].x.shape[1]
    model_metric_columns = ['t_step', 'basis', 'basis_normalized', 'optimizer',
                            'optimizer_args', 'mse']
    model_metric_columns.extend(
        f'indiv_mse[{i}]' for i in range(num_variables))
    model_metric_columns.append('aicc')
    model_metric_columns.extend(
        f'indiv_aicc[{i}]' for i in range(num_variables))
    model_metric_columns.append('recovered_eqs')
    search_config['model_metrics'] = pd.DataFrame(columns=model_metric_columns)

    search_config['stop_events'] = [get_lower_bound_checker(-10.0),
                                    get_upper_bound_checker(20.0)]
    for event in search_config['stop_events']:
        event.terminal = True

    print('SINDy model selection setup finished', flush=True)
    t_span_str = ', '.join(str(t) for t in search_config['t_sindy_span'])
    print(f'Time span for SINDy training: ({t_span_str})', flush=True)
    print_hrule()

    search_time_steps(search_config, verbose=verbose)
    print('Search completed', flush=True)

    search_config['model_metrics'].to_csv(
        os.path.join(output_dir, 'sindy_model_metrics.csv'), index=False)
    print('Model metrics saved.', flush=True)


if __name__ == '__main__':
    main()
