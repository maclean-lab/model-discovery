import os.path
import itertools
import numbers
from argparse import ArgumentParser

import numpy as np
from numpy.random import default_rng
import pandas as pd
from pysindy import STLSQ, PolynomialLibrary
import h5py
import matplotlib

from time_series_data import load_dataset_from_h5
from dynamical_model_learning import NeuralDynamicsLearner
from dynamical_model_learning import OdeSystemLearner
from model_helpers import get_model_module, get_data_path, \
    get_model_selection_dir, print_hrule


def get_latent_learning_config(model_name, search_config):
    match model_name:
        case 'lotka_volterra':
            search_config['opt_threshold'] = 0.1
            search_config['basis_funcs'] = {
                'default': PolynomialLibrary(degree=2),
                'higher_order': [
                    lambda x: x ** 2, lambda x, y: x * y, lambda x: x ** 3,
                    lambda x, y: x * x * y, lambda x, y: x * y * y
                ]}
            search_config['basis_exprs'] = {
                'default': None,
                'higher_order': [
                    lambda x: f'{x}^2', lambda x, y: f'{x} {y}',
                    lambda x: f'{x}^3', lambda x, y: f'{x}^2 {y}',
                    lambda x, y: f'{x} {y}^2'
                ]
            }
        case 'repressilator':
            search_config['opt_threshold'] = 1.0
            search_config['basis_funcs'] = {
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
            search_config['basis_exprs'] = {
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
        case 'emt':
            search_config['opt_threshold'] = 0.1
            search_config['basis_funcs'] = {
                'default': PolynomialLibrary(degree=2),
                'no_bias': PolynomialLibrary(degree=2, include_bias=False),
                'hill_1': [lambda x: 1.0 / (1.0 + x), lambda x: x],
                'hill_2': [lambda x: 1.0 / (1.0 + x ** 2), lambda x: x],
                'hill_3': [lambda x: 1.0 / (1.0 + x ** 3), lambda x: x],
                'hill_max_3': [lambda x: 1.0 / (1.0 + x),
                               lambda x: 1.0 / (1.0 + x ** 2),
                               lambda x: 1.0 / (1.0 + x ** 3),
                               lambda x: x],
            }
            search_config['basis_exprs'] = {
                'default': None,
                'no_bias': None,
                'hill_1': [lambda x: f'/(1+{x})', lambda x: f'{x}'],
                'hill_2': [lambda x: f'/(1+{x}^2)', lambda x: f'{x}'],
                'hill_3': [lambda x: f'/(1+{x}^3)', lambda x: f'{x}'],
                'hill_max_3': [lambda x: f'/(1+{x})',
                               lambda x: f'/(1+{x}^2)',
                               lambda x: f'/(1+{x}^3)',
                               lambda x: f'{x}'],
            }


def get_full_learning_config(model_name, search_config):
    match model_name:
        case 'lotka_volterra':
            search_config['opt_threshold'] = 0.1
            search_config['basis_funcs'] = {
                'default': PolynomialLibrary(degree=2),
                'higher_order': PolynomialLibrary(degree=3)
            }
            search_config['basis_exprs'] = {
                'default': None,
                'higher_order': None
            }
        case 'repressilator':
            search_config['opt_threshold'] = 1.0
            search_config['basis_funcs'] = {
                'hill_1': [lambda x: x, lambda x: 1.0 / (1.0 + x)],
                'hill_2': [lambda x: x, lambda x: 1.0 / (1.0 + x ** 2)],
                'hill_3': [lambda x: x, lambda x: 1.0 / (1.0 + x ** 3)],
                'hill_4': [lambda x: x, lambda x: 1.0 / (1.0 + x ** 4)],
                'hill_max_3': [lambda x: x,
                               lambda x: 1.0 / (1.0 + x),
                               lambda x: 1.0 / (1.0 + x ** 2),
                               lambda x: 1.0 / (1.0 + x ** 3)],
                'hill_max_4': [lambda x: x,
                               lambda x: 1.0 / (1.0 + x),
                               lambda x: 1.0 / (1.0 + x ** 2),
                               lambda x: 1.0 / (1.0 + x ** 3),
                               lambda x: 1.0 / (1.0 + x ** 4)]
            }
            search_config['basis_exprs'] = {
                'hill_1': [lambda x: f'{x}', lambda x: f'/(1+{x})'],
                'hill_2': [lambda x: f'{x}', lambda x: f'/(1+{x}^2)'],
                'hill_3': [lambda x: f'{x}', lambda x: f'/(1+{x}^3)'],
                'hill_4': [lambda x: f'{x}', lambda x: f'/(1+{x}^4)'],
                'hill_max_3': [lambda x: f'{x}',
                               lambda x: f'/(1+{x})',
                               lambda x: f'/(1+{x}^2)',
                               lambda x: f'/(1+{x}^3)'],
                'hill_max_4': [lambda x: f'{x}',
                               lambda x: f'/(1+{x})',
                               lambda x: f'/(1+{x}^2)',
                               lambda x: f'/(1+{x}^3)',
                               lambda x: f'/(1+{x}^4)']
            }
        case 'emt':
            search_config['opt_threshold'] = 0.1
            search_config['basis_funcs'] = {
                'default': PolynomialLibrary(degree=2),
                'no_bias': PolynomialLibrary(degree=2, include_bias=False),
                'hill_1': [lambda x: 1.0 / (1.0 + x), lambda x: x],
                'hill_2': [lambda x: 1.0 / (1.0 + x ** 2), lambda x: x],
                'hill_3': [lambda x: 1.0 / (1.0 + x ** 3), lambda x: x],
                'hill_max_3': [lambda x: 1.0 / (1.0 + x),
                               lambda x: 1.0 / (1.0 + x ** 2),
                               lambda x: 1.0 / (1.0 + x ** 3),
                               lambda x: x],
            }
            search_config['basis_exprs'] = {
                'default': None,
                'no_bias': None,
                'hill_1': [lambda x: f'/(1+{x})', lambda x: f'{x}'],
                'hill_2': [lambda x: f'/(1+{x}^2)', lambda x: f'{x}'],
                'hill_3': [lambda x: f'/(1+{x}^3)', lambda x: f'{x}'],
                'hill_max_3': [lambda x: f'/(1+{x})',
                               lambda x: f'/(1+{x}^2)',
                               lambda x: f'/(1+{x}^3)',
                               lambda x: f'{x}'],
            }


def recover_from_data(args, search_config, verbose) -> bool:
    print('Setting up SINDy learning directly from data...', flush=True)
    get_full_learning_config(args.model, search_config)

    def recovered_dynamics(t, x, model):
        return model.predict(x[np.newaxis, :])[0]

    search_config['recovered_dynamics'] = recovered_dynamics

    noise_type = args.noise_type
    noise_level = args.noise_level
    data_source = args.data_source
    seed = args.seed
    output_dir = get_model_selection_dir(
        args.model, noise_type, noise_level, seed, data_source, 'baseline',
        'sindy')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    search_config['output_dir'] = output_dir

    match args.model:
        case 'lotka_volterra':
            t_step = 0.1
        case 'repressilator':
            t_step = 0.2
        case 'emt':
            t_step = 1.0

    search_config['learn_dx'] = False

    print('SINDy model selection setup finished', flush=True)
    print_hrule()

    search_basis(search_config, search_config['train_samples'],
                 {'t_step': t_step}, verbose=verbose)

    return True


def recover_from_ude(args, search_config, params_true, verbose) -> bool:
    print('Setting up SINDy learning from UDE model...', flush=True)

    # load learned UDE model
    num_hidden_neurons = args.num_hidden_neurons
    activation = args.activation
    print('Loading UDE model with lowest validation loss...', flush=True)
    print('Network architecture:', flush=True)
    print(f'- Hidden neurons: {num_hidden_neurons}', flush=True)
    print(f'- Activation function: {activation}', flush=True)
    noise_type = args.noise_type
    noise_level = args.noise_level
    data_source = args.data_source
    seed = args.seed
    nn_config = {'num_hidden_neurons': num_hidden_neurons,
                 'activation': activation}
    output_dir = get_model_selection_dir(
        args.model, noise_type, noise_level, seed, data_source, 'ude', 'ude',
        ude_rhs=args.ude_rhs, nn_config=nn_config)

    if not os.path.exists(output_dir):
        print('No UDE model found for the given architecture', flush=True)
        print('Terminating...', flush=True)

        return False

    # TODO: if specified, preprocess by LSTM
    ude_train_samples = search_config['train_samples']

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
    if args.ude_rhs == 'nn':
        get_full_learning_config(args.model, search_config)
        neural_dynamics = model_module.get_neural_dynamics(
            num_hidden_neurons=num_hidden_neurons, activation=activation)

        def recovered_dynamics(t, x, model):
            return model.predict(x[np.newaxis, :])[0]
    else:  # args.ude_rhs == 'hybrid'
        get_latent_learning_config(args.model, search_config)

        match args.model:
            case 'lotka_volterra':
                growth_rates = np.array([params_true[0], -params_true[3]])
                neural_dynamics = model_module.get_hybrid_dynamics(
                    growth_rates, num_hidden_neurons=num_hidden_neurons,
                    activation=activation)

                def recovered_dynamics(t, x, model):
                    return growth_rates * x + model.predict(
                        x[np.newaxis, :])[0]
            case 'repressilator':
                neural_dynamics = model_module.get_hybrid_dynamics(
                    num_hidden_neurons=num_hidden_neurons,
                    activation=activation)

                def recovered_dynamics(t, x, model):
                    return model.predict(x[np.newaxis, :])[0] - x
            case 'emt':
                neural_dynamics = model_module.get_hybrid_dynamics()

    search_config['ude_rhs'] = args.ude_rhs
    search_config['recovered_dynamics'] = recovered_dynamics
    match args.model:
        case 'lotka_volterra':
            search_config['t_ude_steps'] = [0.1, 0.05]
        case 'repressilator':
            search_config['t_ude_steps'] = [0.2, 0.1]
        case 'emt':
            search_config['t_ude_steps'] = [1.0, 0.5, 0.25, 0.1]
    search_config['learn_dx'] = True
    ude_learner = NeuralDynamicsLearner(
        search_config['train_samples'], output_dir, output_prefix)
    ude_learner.load_model(
        neural_dynamics, output_suffix=f'model_state_epoch_{best_epoch:03d}')
    print('UDE model loaded', flush=True)

    # set up for SINDy model selection
    output_dir = get_model_selection_dir(
        args.model, noise_type, noise_level, seed, data_source, 'ude',
        'sindy', ude_rhs=args.ude_rhs, nn_config=nn_config)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    search_config['output_dir'] = output_dir
    ude_learner.output_dir = output_dir
    search_config['ude_learner'] = ude_learner

    print('SINDy model selection setup finished', flush=True)
    print_hrule()

    search_time_steps(search_config, ude_train_samples, verbose=verbose)

    return True


def search_time_steps(search_config, ude_train_samples, verbose=False):
    sindy_t_span = search_config['sindy_t_span']
    ude_learner = search_config['ude_learner']
    train_samples = search_config['train_samples']
    num_train_samples = len(ude_train_samples)

    # evaluate on UDE training time span
    for t_step in search_config['t_ude_steps']:
        print(f'Generating UDE dynamics on t_step {t_step:.03f} for SINDy '
              'training...', flush=True)
        t_train = [np.arange(ts.t[0], ts.t[-1] + t_step * 1e-3, t_step)
                   for ts in ude_train_samples]
        x0_train = [ts.x[0, :] for ts in ude_train_samples]

        # generate UDE dynamics with slightly different initial conditions
        # if all initial conditions are the same
        if search_config['noise_type'] != 'fixed' \
                and all(np.array_equal(x0_train[0], x0) for x0 in x0_train):
            x0_base = x0_train[0]
            x0_train = []
            for _ in range(num_train_samples):
                x0_noise_scale = search_config['rng'].normal(
                    scale=search_config['noise_level'], size=x0_base.shape)
                x0_noise_scale = np.clip(x0_noise_scale, -0.01, 0.01)
                x0_train.append(x0_base * (1 + x0_noise_scale))

        ude_learner.output_prefix = f't_step_{t_step:.2f}_ude'
        if search_config['ude_rhs'] == 'hybrid':
            eval_modules = 'latent'
        else:  # search_config['ude_rhs'] == 'nn'
            eval_modules = 'self'
        ude_learner.eval(t_eval=t_train, x0_eval=x0_train,
                         ref_data=train_samples, integrator_backend='scipy',
                         integrator_kwargs={'method': 'LSODA'},
                         eval_modules=eval_modules, verbose=verbose,
                         show_progress=False)
        ude_learner.plot_pred_data(ref_data=train_samples)

        sindy_train_samples = []
        for ts, dx_pred in zip(ude_learner.pred_data,
                               ude_learner.module_pred_data[eval_modules]):
            ts = ts.copy()
            ts.dx = dx_pred

            # cut off some initial and final time points
            t_sindy_indices = np.where(
                (ts.t >= sindy_t_span[0]) & (ts.t <= sindy_t_span[1]))[0]
            if t_sindy_indices.size == 0:
                raise ValueError('No data for SINDy training on the given '
                                 'time span')
            ts = ts[t_sindy_indices]

            sindy_train_samples.append(ts)

        print('SINDy training samples generated from neural dynamics')
        print_hrule()

        search_basis(search_config, sindy_train_samples, {'t_step': t_step},
                     verbose=verbose)


def search_basis(search_config, sindy_train_samples, model_info,
                 verbose=False):
    for basis_type in search_config['basis_funcs']:
        model_info_new = model_info.copy()
        model_info_new['basis'] = basis_type

        search_basis_normalization(search_config, sindy_train_samples,
                                   model_info_new, verbose=verbose)


def search_basis_normalization(search_config, sindy_train_samples, model_info,
                               verbose=False):
    for normalize_columns in [False, True]:
        model_info_new = model_info.copy()
        model_info_new['basis_normalized'] = normalize_columns

        search_optimizers(search_config, sindy_train_samples, model_info_new,
                          verbose=verbose)


def search_optimizers(search_config, sindy_train_samples, model_info,
                      verbose=False):
    for optimizer_name in search_config['sindy_optimizers']:
        model_info_new = model_info.copy()
        model_info_new['optimizer'] = optimizer_name

        search_optimizer_args(search_config, sindy_train_samples,
                              model_info_new, verbose=verbose)


def search_optimizer_args(search_config, sindy_train_samples, model_info,
                          verbose=False):
    sindy_optimizer_args = search_config['sindy_optimizer_args']
    optimizer_name = model_info['optimizer']
    arg_names = sindy_optimizer_args[optimizer_name].keys()
    arg_vals = sindy_optimizer_args[optimizer_name].values()

    for arg_vals in itertools.product(*arg_vals):
        model_info_new = model_info.copy()
        model_info_new['optimizer_args'] = dict(zip(arg_names, arg_vals))

        get_sindy_model(search_config, sindy_train_samples, model_info_new,
                        verbose=verbose)


def get_sindy_model(search_config, sindy_train_samples, model_info,
                    verbose=False):
    # unpack search config
    output_dir = search_config['output_dir']
    sindy_optimizers = search_config['sindy_optimizers']
    basis_funcs = search_config['basis_funcs']
    basis_exprs = search_config['basis_exprs']
    valid_samples = search_config['valid_samples']
    test_samples = search_config['test_samples']
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
        output_prefix += f'normalized_opt_{optimizer_name}'
    else:
        output_prefix += f'unnormalized_opt_{optimizer_name}'
    for arg_name, arg_val in optimizer_args.items():
        if isinstance(arg_val, numbers.Number):
            output_prefix += f'_{arg_name}_{arg_val:.2f}'
        else:
            output_prefix += f'_{arg_name}_{arg_val}'

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
    eq_learner = OdeSystemLearner(sindy_train_samples, output_dir,
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
                     threshold=search_config['opt_threshold'],
                     learn_dx=search_config['learn_dx'],
                     normalize_columns=normalize_columns,
                     basis_funcs=basis_funcs[basis],
                     basis_exprs=basis_exprs[basis],
                     optimizer_kwargs=optimizer_args, valid_data=valid_samples,
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
    if search_config['model'] == 'emt':
        # evaluate EMT model on denser time points
        test_samples = [ts[1:] for ts in test_samples]
        t_test = test_samples[0].t
        # use time points that are powers of 2 to avoid numerical issues
        t_eval = np.arange(t_test[0], t_test[-1] + 1e-8, 2 ** -4)
        x0_eval = [ts.x[0, :] for ts in test_samples]
        eq_learner.eval(t_eval=t_eval, x0_eval=x0_eval, ref_data=test_samples,
                        eval_func=recovered_dynamics,
                        integrator_kwargs=eval_integrator_kwargs,
                        verbose=verbose)
        eq_learner.plot_pred_data(ref_data=test_samples,
                                  output_suffix='pred_data_long')
    else:
        eq_learner.eval(eval_data=test_samples, eval_func=recovered_dynamics,
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
                            choices=['lotka_volterra', 'repressilator', 'emt'],
                            help='Dynamical model from which data is '
                            'generated')
    arg_parser.add_argument('--noise_type', type=str, required=True,
                            choices=['fixed', 'additive', 'multiplicative'],
                            help='Type of noise added to data')
    arg_parser.add_argument('--noise_level', type=float, default=0.01,
                            help='Noise level of training data')
    arg_parser.add_argument('--seed', type=int, default=2023,
                            help='Random seed of generated data')
    arg_parser.add_argument('--data_source', type=str, default='raw',
                            help='Data source for UDE model training')
    arg_parser.add_argument('--data_preprocessor', type=str, default='none',
                            choices=['none', 'lstm', 'ude', 'lstm_ude'],
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
                            choices=['tanh', 'relu', 'rbf'],
                            help='Activation function for the neural network')
    arg_parser.add_argument('--sindy_t_span', nargs=2, type=float,
                            default=[-np.inf, np.inf], metavar=('T0', 'T_END'),
                            help='Time span of SINDy training data')
    arg_parser.add_argument('--matplotlib_backend', type=str, default='Agg',
                            help='Matplotlib backend to use')
    arg_parser.add_argument('--verbose', action='store_true',
                            help='Print output for training progress')

    return arg_parser.parse_args()


# set up boundary check for integration
def get_lower_bound_checker(lower_bound):
    def check_lower_bound(t, x, model=None):
        return float(np.all(x > lower_bound))

    return check_lower_bound


def get_upper_bound_checker(upper_bound):
    def check_upper_bound(t, x, model=None):
        return float(np.all(x < upper_bound))

    return check_upper_bound


def main():
    args = get_args()
    verbose = args.verbose
    matplotlib.use(args.matplotlib_backend)
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    # initialize model and data
    noise_type = args.noise_type
    noise_level = args.noise_level
    seed = args.seed
    print('Loading data...', flush=True)
    data_source = args.data_source
    data_path = get_data_path(args.model, noise_type, noise_level, seed,
                              data_source)
    data_fd = h5py.File(data_path, 'r')
    if 'param_values' in data_fd.attrs:
        params_true = data_fd.attrs['param_values']
    else:
        params_true = None
    t_train_span = data_fd['train'].attrs['t_span']
    train_samples = load_dataset_from_h5(data_fd, 'train')
    valid_samples = load_dataset_from_h5(data_fd, 'valid')
    test_samples = load_dataset_from_h5(data_fd, 'test')
    data_fd.close()

    print('Data loaded:', flush=True)
    print(f'- Model: {args.model}', flush=True)
    param_str = ', '.join(str(p) for p in params_true)
    print(f'- True parameter value: [{param_str}]', flush=True)
    print(f'- Noise type: {noise_type}', flush=True)
    if noise_type != 'fixed':
        print(f'- Noise level: {noise_level}', flush=True)
    print(f'- RNG seed: {seed}', flush=True)
    print(f'- Data source: {data_source}', flush=True)
    t_span_str = ', '.join(str(t) for t in t_train_span)
    print(f'- Time span of training data: ({t_span_str})', flush=True)
    print(f'- Training dataset size: {len(train_samples)}', flush=True)
    print(f'- Validation dataset size: {len(valid_samples)}', flush=True)
    print(f'- Test dataset size: {len(valid_samples)}', flush=True)

    # set up model selection
    search_config = {}
    search_config['model'] = args.model
    search_config['noise_type'] = noise_type
    search_config['noise_level'] = noise_level
    search_config['t_train_span'] = t_train_span
    sindy_t_span = args.sindy_t_span
    sindy_t_span[0] = max(sindy_t_span[0], t_train_span[0])
    sindy_t_span[1] = min(sindy_t_span[1], t_train_span[1])
    search_config['sindy_t_span'] = tuple(sindy_t_span)
    t_span_str = ', '.join(str(t) for t in search_config['sindy_t_span'])
    print(f'Time span for SINDy training: ({t_span_str})', flush=True)
    search_config['train_samples'] = train_samples
    search_config['valid_samples'] = valid_samples
    search_config['test_samples'] = test_samples
    search_config['sindy_optimizers'] = {'stlsq': STLSQ}
    search_config['sindy_optimizer_args'] = {
        'stlsq': {'alpha': [0.05, 0.1, 0.5, 1.0, 5.0, 10.0]}}
    search_config['rng'] = default_rng(seed)
    if args.model == 'emt':
        search_config['stop_events'] = [get_lower_bound_checker(-0.25),
                                        get_upper_bound_checker(1.25)]
    else:
        search_config['stop_events'] = [get_lower_bound_checker(-10.0),
                                        get_upper_bound_checker(20.0)]
    for event in search_config['stop_events']:
        event.terminal = True

    # initialize table for model metrics
    num_variables = train_samples[0].x.shape[1]
    model_metric_columns = ['t_step', 'basis', 'basis_normalized', 'optimizer',
                            'optimizer_args', 'mse']
    model_metric_columns.extend(
        f'indiv_mse[{i}]' for i in range(num_variables))
    model_metric_columns.append('aicc')
    model_metric_columns.extend(
        f'indiv_aicc[{i}]' for i in range(num_variables))
    model_metric_columns.append('recovered_eqs')
    search_config['model_metrics'] = pd.DataFrame(columns=model_metric_columns)

    print_hrule()

    if args.data_preprocessor in ('lstm', 'lstm_ude'):
        print('SINDy recovery not implemented yet for '
              f'{args.data_preprocessor}', flush=True)

        is_successful = False
    elif args.data_preprocessor == 'ude':
        is_successful = recover_from_ude(args, search_config, params_true,
                                         verbose)
    else:  # args.data_preprocessor == 'none'
        is_successful = recover_from_data(args, search_config, verbose)

    if is_successful:
        print('SINDy model selection completed', flush=True)

        model_metric_path = os.path.join(
            search_config['output_dir'], 'sindy_model_metrics.csv')
        search_config['model_metrics'].to_csv(model_metric_path, index=False)
        print('Model metrics saved.', flush=True)
    else:
        print('SINDy model selection failed', flush=True)


if __name__ == '__main__':
    main()
