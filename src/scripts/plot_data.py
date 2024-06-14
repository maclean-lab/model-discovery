import os
import os.path
import numbers
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pysindy as ps
from pysindy.differentiation import FiniteDifference
import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from time_series_data import load_dataset_from_h5
from dynamical_model_learning import NeuralDynamicsLearner, OdeSystemLearner
from model_helpers import get_model_module, get_model_class, \
    get_model_prefix, get_project_root, get_data_path, get_model_selection_dir
from sindy_model_selection import get_full_learning_config, \
    get_latent_learning_config, get_lower_bound_checker, \
    get_upper_bound_checker


def get_args():
    arg_parser = ArgumentParser(
        description='Plot generated data for a dynamical model.')

    arg_parser.add_argument('--model', type=str, required=True,
                            choices=['lotka_volterra', 'repressilator', 'emt'],
                            help='Dynamical model to generate data from')
    arg_parser.add_argument(
        '--noise_type', type=str, required=True,
        choices=['none', 'fixed', 'additive', 'multiplicative'],
        help='Type of noise to add to data')

    # figure related arguments
    arg_parser.add_argument('--backend', type=str, default='Agg',
                            help='Matplotlib backend to use')
    arg_parser.add_argument('--figure_size', type=float, nargs=2,
                            default=(3, 2), metavar=('WIDTH', 'HEIGHT'),
                            help='Figure size in inches')
    arg_parser.add_argument('--figure_dpi', type=int, default=300,
                            help='Figure resolution in dots per inch')
    arg_parser.add_argument('--plot_type', type=str, default='line',
                            choices=['line', 'strip'],
                            help='Type of plot to generate')
    arg_parser.add_argument('--x_label', type=str, nargs='?', const='',
                            default='', help='Label of x-axis')
    arg_parser.add_argument('--y_label', type=str, nargs='?', const='',
                            default='', help='Label of y-axis')
    arg_parser.add_argument('--legend', action='store_true',
                            help='Whether to show legend')

    # conditional arguments for different types of data
    args, _ = arg_parser.parse_known_args()
    model_class = get_model_class(args.model)

    if args.noise_type == 'none':
        # true data
        num_variables = model_class.get_num_variables()
        default_x0 = model_class.get_default_x0().tolist()
        x0_meta_names = tuple(f'X0[{i}, 0]' for i in range(num_variables))
        arg_parser.add_argument('--x0', nargs=num_variables, type=float,
                                default=default_x0, metavar=x0_meta_names,
                                help='Initial conditions of data')

        default_param_values = model_class.get_default_param_values().tolist()
        if len(default_param_values) > 0:
            param_meta_names = tuple(
                pn.upper() for pn in model_class.get_param_names())
            arg_parser.add_argument('--param_values',
                                    nargs=len(default_param_values),
                                    type=float, default=default_param_values,
                                    metavar=param_meta_names,
                                    help='Parameter values of the model')

        default_t_span = model_class.get_default_t_span()
        default_t_step = model_class.get_default_t_step()
        if default_t_step is None:
            default_t_step = (default_t_span[1] - default_t_span[0]) / 10
        arg_parser.add_argument('--t_span', nargs=2, type=float,
                                default=default_t_span,
                                metavar=('T0', 'T_END'),
                                help='Time span on which to simulate data')
        arg_parser.add_argument('--t_step', type=float, default=default_t_step,
                                metavar='T_STEP',
                                help='Time step for simulating data')
    else:
        # noisy data
        arg_parser.add_argument('--noise_level', type=float, default=0.01,
                                help='Noise level of generated data')
        arg_parser.add_argument('--seed', type=int, default=2023,
                                help='Random seed for generating data')
        arg_parser.add_argument('--data_source', type=str, default='raw',
                                help='Source of generated data, e.g. raw')
        arg_parser.add_argument('--dataset', type=str, default='train',
                                choices=['train', 'valid', 'test'],
                                help='Dataset to plot, e.g. train')
        arg_parser.add_argument('--plot_dx', action='store_true',
                                help='Whether to plot dx instead of x')

        # optional arguments for UDE
        arg_parser.add_argument('--ude_rhs', type=str, default='none',
                                choices=['none', 'nn', 'hybrid'],
                                help='Right-hand side of UDE')
        arg_parser.add_argument('--ude_t_step', type=float, default=0.0,
                                help='Time step for simulating UDE model')
        arg_parser.add_argument('--num_hidden_neurons', nargs='+', type=int,
                                default=[5, 5],
                                help='Number of neurons in each hidden layer '
                                'of the neural network used by UDE')
        arg_parser.add_argument('--activation', type=str, default='tanh',
                                choices=['tanh', 'relu', 'rbf', 'sigmoid',
                                         'softplus', 'identity'],
                                help='Activation function for the neural '
                                'network used by UDE')
        arg_parser.add_argument('--ude_learning_rate', type=float,
                                default=0.0,
                                help='Learning rate of training UDE model')
        arg_parser.add_argument('--ude_window_size', type=int, default=0,
                                help='Window size of training UDE model')
        arg_parser.add_argument('--ude_batch_size', type=int, default=0,
                                help='Batch size of training UDE model')

        # optional argument for SINDy
        arg_parser.add_argument('--sindy_optimizer', type=str, default='none',
                                choices=['none', 'stlsq', 'sr3'],
                                help='Optimizer used by SINDy')
        arg_parser.add_argument('--sindy_opt_threshold', type=float, default=0,
                                help='Threshold used by SINDy optimizer')
        arg_parser.add_argument('--stlsq_alpha', type=float, default=0.05,
                                help='alpha argument for the STLSQ optimizer')
        arg_parser.add_argument('--sindy_t_span', nargs=2, type=float,
                                default=[-np.inf, np.inf],
                                metavar=('T0', 'T_END'),
                                help='Time span of SINDy training data')
        arg_parser.add_argument('--sindy_basis', type=str, default='default',
                                help='Basis functions used by SINDy')

    return arg_parser.parse_args()


def plot_true_data(args):
    x0 = np.array(args.x0)
    param_values = np.array(args.param_values)
    t_span = np.array(args.t_span)
    t = np.arange(t_span[0], t_span[1] + 1e-8, args.t_step)
    model_class = get_model_class(args.model)
    model = model_class(param_values=param_values, x0=x0)
    x = model.simulate(t_span, x0, t_eval=t).x

    figure_dir = get_figure_dir(args.model)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    figure_path = os.path.join(figure_dir, 'true_data.pdf')

    plt.figure(figsize=args.figure_size, dpi=args.figure_dpi)
    plt.plot(t, x, label=model.get_variable_names())
    plt.xlabel(args.x_label)
    plt.ylabel(args.y_label)
    if args.legend:
        plt.legend()
    plt.savefig(figure_path, transparent=True)
    plt.close('all')


def plot_noisy_data(args):
    # load data
    noise_type = args.noise_type
    noise_level = args.noise_level
    seed = args.seed
    data_source = args.data_source
    dataset = args.dataset

    data_path = get_data_path(args.model, noise_type, noise_level, seed,
                              data_source)
    data_fd = h5py.File(data_path, 'r')
    samples = load_dataset_from_h5(data_fd, args.dataset)
    data_fd.close()

    figure_dir = get_figure_dir(args.model)
    long_data = get_long_data_from_samples(args.model, samples)

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    figure_path = f'{noise_type}_noise'
    if noise_type != 'fixed':
        figure_path += f'_{noise_level:.03f}'
    figure_path += f'_seed_{seed:04d}_{data_source}_{dataset}_data'
    if args.plot_type == 'line':
        figure_path += '.pdf'
    else:
        figure_path += f'_{args.plot_type}.pdf'
    figure_path = os.path.join(figure_dir, figure_path)
    model_class = get_model_class(args.model)
    data_colors = get_data_colors(model_class.get_num_variables())

    plt.figure(figsize=args.figure_size, dpi=args.figure_dpi)

    match args.plot_type:
        case 'line':
            g = sns.lineplot(
                data=long_data, x='Time', y='Value', hue='Variable',
                palette=data_colors, linewidth=1, errorbar=('sd', 3),
                err_style='band', legend=args.legend)
        case 'strip':
            g = sns.stripplot(
                data=long_data, x='Time', y='Value', hue='Variable',
                palette=data_colors, jitter=True, dodge=True, size=1.0,
                alpha=0.5, legend=args.legend)
    if args.model == 'emt':
        xticks = g.get_xticks()
        xtick_labels = [float(t.get_text().replace('−', '-'))
                        for t in g.get_xticklabels()]

        if all(t.is_integer() for t in xtick_labels):
            xtick_labels = [f'{int(t)}' for t in xtick_labels]
            g.set_xticks(xticks, labels=xtick_labels)
    g.set_xlabel(args.x_label)
    g.set_ylabel(args.y_label)
    plt.savefig(figure_path, transparent=True)
    plt.close('all')


def plot_sindy_dx(args):
    # unpack arguments
    noise_type = args.noise_type
    noise_level = args.noise_level
    seed = args.seed
    data_source = args.data_source
    dataset = args.dataset

    data_path = get_data_path(args.model, noise_type, noise_level, seed,
                              data_source)
    data_fd = h5py.File(data_path, 'r')
    params_true = data_fd.attrs['param_values']
    x0 = data_fd.attrs['x0']
    samples = load_dataset_from_h5(data_fd, dataset)
    data_fd.close()

    get_true_dx(args, params_true, x0, samples)
    dx_method = FiniteDifference()

    figure_dir = get_figure_dir(args.model)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    figure_path = f'{noise_type}_noise'
    if noise_type != 'fixed':
        figure_path += f'_{noise_level:.03f}'
    figure_path += f'_seed_{seed:04d}_{data_source}_{dataset}_sindy_dx.pdf'
    figure_path = os.path.join(figure_dir, figure_path)
    model_class = get_model_class(args.model)
    num_vars = model_class.get_num_variables()
    data_colors = get_data_colors(num_vars)
    data_labels = model_class.get_variable_names()

    with PdfPages(figure_path) as pdf:
        for ts in samples:
            plt.figure(figsize=args.figure_size, dpi=args.figure_dpi)
            dx = dx_method._differentiate(ts.x, ts.t)
            for i in range(num_vars):
                plt.plot(ts.t, ts.dx[:, i], marker='o', linestyle='none',
                         color=data_colors[i], alpha=0.3, label=data_labels[i])
                plt.plot(ts.t, dx[:, i], color=data_colors[i],
                         label=data_labels[i])
            plt.xlabel(args.x_label)
            plt.ylabel(args.y_label)
            if args.legend:
                plt.legend()
            pdf.savefig(transparent=True)
            plt.close('all')


def plot_ude_data(args):
    # unpack arguments
    noise_type = args.noise_type
    noise_level = args.noise_level
    seed = args.seed
    data_source = args.data_source
    t_step = args.ude_t_step
    ude_rhs = args.ude_rhs
    num_hidden_neurons = args.num_hidden_neurons
    activation = args.activation
    # for now, plotting dx for EMT model is not supported
    plot_dx = args.plot_dx and args.model != 'emt'

    data_path = get_data_path(args.model, noise_type, noise_level, seed,
                              data_source)
    data_fd = h5py.File(data_path, 'r')
    train_samples = load_dataset_from_h5(data_fd, 'train')
    params_true = data_fd.attrs['param_values']
    if t_step == 0.0:
        t_step = data_fd['train'].attrs['t_step']
    x0 = data_fd.attrs['x0']
    data_fd.close()

    # get hyperparameters from args
    learning_rate = args.ude_learning_rate
    window_size = args.ude_window_size
    batch_size = args.ude_batch_size
    if learning_rate > 0.0 and window_size > 0 and batch_size > 0:
        hyperparams = {'learning_rate': learning_rate,
                       'window_size': window_size,
                       'batch_size': batch_size}
    else:
        hyperparams = None

    # get true derivative data
    if plot_dx:
        get_true_dx(args, params_true, x0, train_samples, ude_rhs=ude_rhs)

    # get predicted data from trained UDE model
    pred_samples = simulate_ude_model(
        train_samples, t_step, args, params_true=params_true,
        hyperparams=hyperparams)

    # plot predicted data
    figure_dir = get_figure_dir(args.model)
    ude_model_alias = f'{ude_rhs}_'
    ude_model_alias += '_'.join(str(i) for i in num_hidden_neurons)
    ude_model_alias += f'_{activation}_lr_{learning_rate:.03f}'
    ude_model_alias += f'_window_size_{window_size:02d}'
    ude_model_alias += f'_batch_size_{batch_size:02d}'
    figure_path = f'{noise_type}_noise'
    if noise_type != 'fixed':
        figure_path += f'_{noise_level:.03f}'
    figure_path += f'_seed_{seed:04d}_{data_source}_{ude_model_alias}'
    if plot_dx:
        figure_path += '_pred_dx.pdf'
    else:
        figure_path += '_pred_data.pdf'
    figure_path = os.path.join(figure_dir, figure_path)
    model_class = get_model_class(args.model)
    num_vars = model_class.get_num_variables()
    data_colors = get_data_colors(num_vars)
    data_labels = model_class.get_variable_names()

    with PdfPages(figure_path) as pdf:
        for ts_pred, ts_train in zip(pred_samples, train_samples):
            plt.figure(figsize=args.figure_size, dpi=args.figure_dpi)
            for i in range(num_vars):
                if plot_dx:
                    y_train = ts_train.dx[:, i]
                    y_pred = ts_pred.dx[:, i]
                else:
                    y_train = ts_train.x[:, i]
                    y_pred = ts_pred.x[:, i]

                plt.plot(ts_train.t, y_train, marker='o', linestyle='none',
                         color=data_colors[i], alpha=0.3, label=data_labels[i])
                plt.plot(ts_pred.t, y_pred, color=data_colors[i],
                         label=data_labels[i])
            plt.xlabel(args.x_label)
            plt.xlabel(args.y_label)
            if args.legend:
                plt.legend()
            pdf.savefig(transparent=True)
            plt.close('all')


def plot_sindy_data(args):
    # unpack arguments
    noise_type = args.noise_type
    noise_level = args.noise_level
    seed = args.seed
    data_source = args.data_source
    ude_rhs = args.ude_rhs
    t_step = args.ude_t_step
    basis_name = args.sindy_basis
    optimizer_name = args.sindy_optimizer
    threshold = args.sindy_opt_threshold
    verbose = True

    # load data
    data_path = get_data_path(args.model, noise_type, noise_level, seed,
                              data_source)
    data_fd = h5py.File(data_path, 'r')
    if 'param_values' in data_fd.attrs:
        params_true = data_fd.attrs['param_values']
    else:
        params_true = None
    if ude_rhs == 'none' or t_step == 0.0:
        t_step = data_fd['train'].attrs['t_step']
    t_train_span = data_fd['train'].attrs['t_span']
    sindy_t_span = args.sindy_t_span
    sindy_t_span[0] = max(sindy_t_span[0], t_train_span[0])
    sindy_t_span[1] = min(sindy_t_span[1], t_train_span[1])
    train_samples = load_dataset_from_h5(data_fd, 'train')
    test_samples = load_dataset_from_h5(data_fd, 'test')
    data_fd.close()

    # preprocess SINDy training data by UDE model
    if ude_rhs in {'nn', 'hybrid'}:
        ude_pred_samples = simulate_ude_model(
            train_samples, t_step, args, params_true=params_true)

        # throw away data outside of SINDy training time span
        sindy_train_samples = []
        for ts in ude_pred_samples:
            t_sindy_indices = np.where(
                (ts.t >= sindy_t_span[0]) & (ts.t <= sindy_t_span[1]))[0]
            sindy_train_samples.append(ts[t_sindy_indices])

        learn_dx = True
        num_hidden_neurons = args.num_hidden_neurons
        activation = args.activation
        prior_step_alias = f'{ude_rhs}_'
        prior_step_alias += '_'.join(str(i) for i in num_hidden_neurons)
        prior_step_alias += f'_{activation}'
    else:
        sindy_train_samples = train_samples
        learn_dx = False
        prior_step_alias = 'baseline'

    # simulate SINDy model on test data
    sindy_search_config = {}
    if ude_rhs == 'hybrid':
        get_latent_learning_config(args.model, sindy_search_config)

        match args.model:
            case 'lotka_volterra':
                growth_rates = np.array([params_true[0], -params_true[3]])

                def recovered_dynamics(t, x, model):
                    return growth_rates * x + model.predict(
                        x[np.newaxis, :])[0]
            case 'repressilator':
                def recovered_dynamics(t, x, model):
                    return model.predict(x[np.newaxis, :])[0] - x

    else:  # ude_rhs in {'none', 'nn'}
        get_full_learning_config(args.model, sindy_search_config)

        def recovered_dynamics(t, x, model):
            return model.predict(x[np.newaxis, :])[0]

    optimizer_types = {'stlsq': ps.STLSQ, 'sr3': ps.SR3}
    optimizer_kwargs = {'stlsq': {'alpha': args.stlsq_alpha}, 'sr3': {}}
    basis_funcs = sindy_search_config['basis_funcs'][basis_name]
    basis_exprs = sindy_search_config['basis_exprs'][basis_name]
    ode_learner = OdeSystemLearner(sindy_train_samples, '', '')
    ode_learner.train(optimizer_type=optimizer_types[optimizer_name],
                      threshold=threshold, learn_dx=learn_dx,
                      normalize_columns=False, basis_funcs=basis_funcs,
                      basis_exprs=basis_exprs,
                      optimizer_kwargs=optimizer_kwargs[optimizer_name],
                      verbose=verbose)

    # plot SINDy data vs test data
    stop_events = [get_lower_bound_checker(-10.0),
                   get_upper_bound_checker(20.0)]
    for event in stop_events:
        event.terminal = True
    integrator_kwargs = {'args': (ode_learner.model, ),
                         'method': 'LSODA',
                         'events': stop_events,
                         'min_step': 1e-8}
    if args.model == 'emt':
        # evaluate EMT model on denser time points
        test_samples = [ts[1:] for ts in test_samples]
        t_test = test_samples[0].t
        # use time points that are powers of 2 to avoid numerical issues
        t_eval = np.arange(t_test[0], t_test[-1] + 1e-8, 2 ** -3)
        x0_eval = [ts.x[0, :] for ts in test_samples]
        ode_learner.eval(t_eval=t_eval, x0_eval=x0_eval, ref_data=test_samples,
                         eval_func=recovered_dynamics,
                         integrator_kwargs=integrator_kwargs,
                         verbose=verbose)
    else:
        ode_learner.eval(eval_data=test_samples, eval_func=recovered_dynamics,
                         integrator_kwargs=integrator_kwargs, verbose=verbose)

    # plot SINDy-predicted data vs test data
    figure_dir = get_figure_dir(args.model)
    sindy_model_alias = f't_step_{t_step:.02f}_pysindy_basis_{basis_name}' \
        f'_unnormalized_opt_{optimizer_name}'
    for arg_name, arg_val in optimizer_kwargs[optimizer_name].items():
        if isinstance(arg_val, numbers.Number):
            sindy_model_alias += f'_{arg_name}_{arg_val:.02f}'
        else:
            sindy_model_alias += f'_{arg_name}_{arg_val}'
    figure_path = f'{noise_type}_noise_{noise_level:.03f}_seed_{seed:04d}' \
        f'_{data_source}_{prior_step_alias}_{sindy_model_alias}_pred_data.pdf'
    figure_path = os.path.join(figure_dir, figure_path)
    model_class = get_model_class(args.model)
    num_vars = model_class.get_num_variables()
    data_colors = get_data_colors(num_vars)
    data_labels = model_class.get_variable_names()
    with PdfPages(figure_path) as pdf:
        for ts_pred, ts_test in zip(ode_learner.pred_data, test_samples):
            plt.figure(figsize=args.figure_size, dpi=args.figure_dpi)
            for i in range(num_vars):
                plt.plot(ts_test.t, ts_test.x[:, i], marker='o', markersize=5,
                         linestyle='none', color=data_colors[i], alpha=0.3,
                         label=data_labels[i])
                plt.plot(ts_pred.t, ts_pred.x[:, i], color=data_colors[i],
                         label=data_labels[i])
            # add a vertical line to indicate the end of training data
            plt.axvline(t_train_span[1], color='k', linestyle='--')
            plt.xlabel(args.x_label)
            plt.xlabel(args.y_label)
            if args.legend:
                plt.legend()
            pdf.savefig(transparent=True)
            plt.close('all')


def get_true_dx(args, params_true, x0, samples, ude_rhs='none'):
    model = get_model_class(args.model)(param_values=params_true, x0=x0)
    true_dynamics = model.equations

    for ts in samples:
        model.t = ts.t
        clean_sample = model.get_samples(1, noise_level=0.0)[0]
        dx = np.array([true_dynamics(t, x) for t, x in
                       zip(clean_sample.t, clean_sample.x)])
        if ude_rhs == 'hybrid':
            if args.model == 'lotka_volterra':
                growth_rates = np.array([params_true[0], -params_true[3]])
            else:  # args.model == 'repressilator'
                growth_rates = -np.ones(model.num_variables)
            dx = dx - growth_rates * clean_sample.x

        ts.dx = dx


def simulate_ude_model(samples, t_step, args, params_true=None,
                       hyperparams=None):
    # unpack arguments
    noise_type = args.noise_type
    noise_level = args.noise_level
    seed = args.seed
    data_source = args.data_source
    ude_rhs = args.ude_rhs
    num_hidden_neurons = args.num_hidden_neurons
    activation = args.activation

    # initialize UDE model
    model_module = get_model_module(args.model)

    if ude_rhs == 'nn':
        neural_dynamics = model_module.get_neural_dynamics(
            num_hidden_neurons=num_hidden_neurons, activation=activation)
    else:  # ude_rhs == 'hybrid'
        match args.model:
            case 'lotka_volterra':
                growth_rates = np.array([params_true[0], -params_true[3]])
                neural_dynamics = model_module.get_hybrid_dynamics(
                    growth_rates, num_hidden_neurons=num_hidden_neurons,
                    activation=activation)
            case 'repressilator':
                neural_dynamics = model_module.get_hybrid_dynamics(
                    num_hidden_neurons=num_hidden_neurons,
                    activation=activation)

    t_eval = [np.arange(ts.t[0], ts.t[-1] + t_step * 1e-3, t_step)
              for ts in samples]
    x0_eval = [ts.x[0, :] for ts in samples]
    if ude_rhs == 'hybrid':
        eval_modules = 'latent'
    else:  # ude_rhs == 'nn'
        eval_modules = 'self'
    nn_config = {'num_hidden_neurons': num_hidden_neurons,
                 'activation': activation}
    output_dir = get_model_selection_dir(
        args.model, noise_type, noise_level, seed, data_source, 'ude', 'ude',
        ude_rhs=ude_rhs, nn_config=nn_config)

    # get the best epoch for UDE model
    ude_model_metric = pd.read_csv(
        os.path.join(output_dir, 'ude_model_metrics.csv'),
        index_col=False)
    if hyperparams is None:
        # get the best hyperparameters
        ude_row = ude_model_metric['best_valid_loss'].idxmin()
        learning_rate = ude_model_metric.loc[ude_row, 'learning_rate']
        window_size = ude_model_metric.loc[ude_row, 'window_size']
        batch_size = ude_model_metric.loc[ude_row, 'batch_size']
        best_epoch = ude_model_metric.loc[ude_row, 'best_epoch']
    else:
        learning_rate = hyperparams['learning_rate']
        window_size = hyperparams['window_size']
        batch_size = hyperparams['batch_size']
        ude_row = (ude_model_metric['learning_rate'] == learning_rate) \
            & (ude_model_metric['window_size'] == window_size) \
            & (ude_model_metric['batch_size'] == batch_size)
        best_epoch = ude_model_metric.loc[ude_row, 'best_epoch'].item()

    # simulate UDE model on training data
    output_prefix = f'lr_{learning_rate:.3f}_window_size_{window_size:02d}'
    output_prefix += f'_batch_size_{batch_size:02d}'
    ts_learner = NeuralDynamicsLearner(samples, output_dir, output_prefix)
    ts_learner.load_model(
        neural_dynamics, output_suffix=f'model_state_epoch_{best_epoch:03d}')
    ts_learner.eval(t_eval=t_eval, x0_eval=x0_eval,
                    integrator_backend='scipy', eval_modules=eval_modules,
                    integrator_kwargs={'method': 'LSODA'},
                    verbose=True, show_progress=True)

    pred_samples = []
    for ts, dx_pred in zip(ts_learner.pred_data,
                           ts_learner.module_pred_data[eval_modules]):
        ts = ts.copy()
        ts.dx = dx_pred
        pred_samples.append(ts)

    return pred_samples


def get_long_data_from_samples(model, samples):
    model_class = get_model_class(model)
    var_names = model_class.get_variable_names()
    long_data = pd.DataFrame(
        columns=['SampleIndex', 'Time', 'Variable', 'Value'])

    for i, s in enumerate(samples):
        for j, t in enumerate(s.t):
            for k, vn in enumerate(var_names):
                long_data.loc[len(long_data), :] = [i, t, vn, s.x[j, k]]

    return long_data


def get_figure_dir(model):
    project_root = get_project_root()
    model_prefix = get_model_prefix(model)
    figure_dir = os.path.join(project_root, 'outputs', f'{model_prefix}-data')

    return figure_dir


def get_data_colors(num_vars):
    return [f'C{i}' for i in range(num_vars)]


def main():
    args = get_args()
    matplotlib.use(args.backend)
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    if args.noise_type == 'none':
        plot_true_data(args)
    else:
        if args.sindy_optimizer != 'none':
            plot_sindy_data(args)
        elif args.ude_rhs != 'none':
            plot_ude_data(args)
        elif args.plot_dx:
            plot_sindy_dx(args)
        else:
            plot_noisy_data(args)


if __name__ == '__main__':
    main()
