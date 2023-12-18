import os
import os.path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import h5py
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from time_series_data import load_dataset_from_h5
from model_helpers import get_model_class, get_model_prefix


def get_args():
    arg_parser = ArgumentParser(
        description='Plot generated data for a dynamical model.')

    arg_parser.add_argument('--model', type=str, required=True,
                            choices=['lotka_volterra', 'repressilator'],
                            help='Dynamical model to generate data from')
    arg_parser.add_argument('--noise_type', type=str, required=True,
                            choices=['none', 'additive', 'multiplicative'],
                            help='Type of noise to add to data')
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
        param_meta_names = tuple(
            pn.upper() for pn in model_class.get_param_names())
        arg_parser.add_argument('--param_values',
                                nargs=len(default_param_values), type=float,
                                default=default_param_values,
                                metavar=param_meta_names,
                                help='Parameter values of the model')

        default_t_span = model_class.get_default_t_span()
        default_t_step = model_class.get_default_t_step()
        if default_t_step is None:
            default_t_step = (default_t_span[1] - default_t_span[0]) / 10
        arg_parser.add_argument('--t_span', nargs=2, type=float,
                                default=default_t_span,
                                metavar=('T0', 'T_END'),
                                help='Time span of test data')
        arg_parser.add_argument('--t_step', type=float, default=default_t_step,
                                metavar='T_STEP',
                                help='Time step of test data')
    else:
        # noisy data
        arg_parser.add_argument('--noise_level', type=float, required=True,
                                help='Noise level of generated data')
        arg_parser.add_argument('--seed', type=int, default=2023,
                                help='Random seed for generating data')
        arg_parser.add_argument('--data_source', type=str, default='raw',
                                help='Source of generated data, e.g. raw')
        arg_parser.add_argument('--dataset', type=str, default='train',
                                choices=['train', 'valid', 'test'],
                                help='Dataset to plot, e.g. train')

    # figure related arguments
    arg_parser.add_argument('--backend', type=str, default='Agg',
                            help='Matplotlib backend to use')
    arg_parser.add_argument('--figure_size', type=float, nargs=2,
                            default=(6, 4), metavar=('WIDTH', 'HEIGHT'),
                            help='Figure size in inches')
    arg_parser.add_argument('--figure_dpi', type=int, default=300,
                            help='Figure resolution in dots per inch')
    arg_parser.add_argument('--x_label', type=str, nargs='?', const='',
                            default='Time', help='Label of x-axis')
    arg_parser.add_argument('--y_label', type=str, nargs='?', const='',
                            default='Value', help='Label of y-axis')

    return arg_parser.parse_args()


def plot_true_data(args):
    x0 = np.array(args.x0)
    param_values = np.array(args.param_values)
    t_span = np.array(args.t_span)
    t = np.arange(t_span[0], t_span[1] + 1e-8, args.t_step)
    model_class = get_model_class(args.model)
    model = model_class(param_values=param_values, x0=x0)
    model_prefix = get_model_prefix(args.model)
    x = model.simulate(t_span, x0, t_eval=t).x

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(project_root, 'outputs', f'{model_prefix}-data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    figure_path = os.path.join(output_dir, 'true_data.pdf')

    plt.figure(figsize=args.figure_size, dpi=args.figure_dpi)
    plt.plot(t, x, label=model.get_variable_names())
    plt.xlabel(args.x_label)
    plt.ylabel(args.y_label)
    plt.savefig(figure_path, transparent=True)
    plt.close('all')


def plot_noisy_data(args):
    # load data
    noise_type = args.noise_type
    noise_level = args.noise_level
    seed = args.seed
    data_source = args.data_source
    dataset = args.dataset

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..'))
    model_prefix = get_model_prefix(args.model)
    data_path = os.path.join(
        project_root, 'data',
        f'{model_prefix}_{noise_type}_noise_{noise_level:.03f}'
        f'_seed_{seed:04d}_{data_source}.h5')
    data_fd = h5py.File(data_path, 'r')
    samples = load_dataset_from_h5(data_fd, args.dataset)
    data_fd.close()
    output_dir = os.path.join(project_root, 'outputs', f'{model_prefix}-data')
    long_data = get_long_data_from_samples(args.model, samples)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    figure_path = f'{noise_type}_noise_{noise_level:.03f}_seed_{seed:04d}'
    figure_path += f'_{data_source}_{dataset}_data.pdf'
    figure_path = os.path.join(output_dir, figure_path)

    plt.figure(figsize=args.figure_size, dpi=args.figure_dpi)
    g = sns.lineplot(data=long_data, x='Time', y='Value', hue='Variable',
                     errorbar=('sd', 3), err_style='bars', legend=False,
                     linewidth=1)
    g.set_xlabel(args.x_label)
    g.set_ylabel(args.y_label)
    plt.savefig(figure_path, transparent=True)
    plt.close('all')


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
        plot_noisy_data(args)


if __name__ == '__main__':
    main()
